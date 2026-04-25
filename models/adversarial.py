# -*- coding: utf-8 -*-
"""
Adversarial Framework: Wasserstein Critic and AdversaryNet (Intruder)
====================================================================
These two neural networks form the adversarial core:
1. WassersteinCritic: Trained with WGAN-GP.
2. AdversaryNet: A learned attacker (U-Net) that tries to destroy bits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. PatchGAN Discriminator (70x70 Markovian)
# =============================================================================

class WassersteinCritic(nn.Module):
    """
    Wasserstein (WGAN-GP) Patch Critic.
    Output: (B, 1, 30, 30) -- patch-level Wasserstein score map
    """

    def __init__(self, in_channels, ndf):
        """
        Args:
            in_channels: Number of input image channels (3 for RGB).
            ndf: Base number of Critic filters. Doubles at each layer.
        """
        super().__init__()

        # IMPORTANT: Use InstanceNorm2d (NOT BatchNorm) throughout.
        # BatchNorm computes stats across the batch, which interferes with
        # the gradient penalty computation that requires per-sample gradients.

        # Layer 1: (B, 3, 256, 256) -> (B, 64, 128, 128) -- No norm on first layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 2: (B, 64, 128, 128) -> (B, 128, 64, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 3: (B, 128, 64, 64) -> (B, 256, 32, 32)
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 4: (B, 256, 32, 32) -> (B, 512, 31, 31) -- stride=1
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output: (B, 512, 31, 31) -> (B, 1, 30, 30)
        # NO Sigmoid -- raw scores are required for Wasserstein Loss
        self.output_layer = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, img):
        """
        img: (B, 3, H, W) tensor -- either original or watermarked image.
        returns: (B, 1, H', W') tensor -- raw Wasserstein score map.
        """
        x = self.layer1(img)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output_layer(x)  # Raw scores -- NO Sigmoid


# =============================================================================
# 2. AdversaryNet (Learned Watermark Removal Intruder)
# =============================================================================

class AdversaryNet(nn.Module):
    """
    Lightweight U-Net Adversary.
    Trained to destroy bits while maintaining quality.
    """
    def __init__(self, in_channels, base_filters, epsilon):
        """
        base_filters: filters count.
        epsilon: max perturbation strength.
        """
        super().__init__()
        self.epsilon = epsilon
        nf = base_filters

        # ===================== ENCODER PATH =====================
        # Stage 1: (B, 3, 256, 256) -> (B, 32, 128, 128)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 2: (B, 32, 128, 128) -> (B, 64, 64, 64)
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 3: (B, 64, 64, 64) -> (B, 128, 32, 32)
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ==================== BOTTLENECK ====================
        # (B, 128, 32, 32) -> (B, 128, 32, 32)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(inplace=True)
        )

        # ===================== DECODER PATH =====================
        # Stage 3: (B, 128+128, 32, 32) -> (B, 64, 64, 64) -- skip from enc3
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(nf * 4 + nf * 4, nf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(inplace=True)
        )

        # Stage 2: (B, 64+64, 64, 64) -> (B, 32, 128, 128) -- skip from enc2
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2 + nf * 2, nf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(inplace=True)
        )

        # Stage 1: (B, 32+32, 128, 128) -> (B, 3, 256, 256) -- skip from enc1
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(nf + nf, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, watermarked_img):
        """
        watermarked_img: (B, 3, 256, 256) -- Watermarked image.
        Returns: (B, 3, 256, 256) -- Attacked image.
        """
        # --- Encoder ---
        e1 = self.enc1(watermarked_img)   # (B, 32, 128, 128)
        e2 = self.enc2(e1)                 # (B, 64, 64, 64)
        e3 = self.enc3(e2)                 # (B, 128, 32, 32)

        # --- Bottleneck ---
        b = self.bottleneck(e3)            # (B, 128, 32, 32)

        # --- Decoder with Skip Connections ---
        d3 = self.dec3(torch.cat([b, e3], dim=1))    # (B, 64, 64, 64)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))   # (B, 32, 128, 128)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))   # (B, 3, 256, 256)

        # --- Bounded Perturbation ---
        # Tanh squashes to [-1, 1], epsilon scales to [-eps, +eps]
        # This mathematically guarantees the attack strength never exceeds
        # the perceptual budget, preserving PSNR > ~28 dB.
        perturbation = torch.tanh(d1) * self.epsilon

        # --- Apply Attack ---
        adversarial_img = torch.clamp(watermarked_img + perturbation, 0.0, 1.0)

        return adversarial_img
