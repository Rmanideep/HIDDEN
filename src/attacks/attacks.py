# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import random
import kornia.augmentation as K
import kornia.filters as F_kornia
from src.engine.math import DiffJPEG

class BenignAugmentationPipeline(nn.Module):
    """
    Differentiable Benign Attack Pipeline.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # JPEG Compression
        self.jpeg_25 = DiffJPEG(quality=25)
        self.jpeg_50 = DiffJPEG(quality=50)
        self.jpeg_75 = DiffJPEG(quality=75)
        
        # NOTE: Geometric distortions (rotation, flip, crop) REMOVED
        # Reason: Spatial-frequency alignment breaks under rotation
        # Benign augmentation now focuses on: JPEG, blur, color jitter, Instagram filters
        # These are sufficient for realistic deepfake detection scenarios
        
        # Filtering
        self.blur_3x3 = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=1.0)
        self.blur_5x5 = K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5), p=1.0)
        
        # Photometric
        self.color_jitter = K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)

    def add_gaussian_noise(self, I, std=0.05):
        noise = torch.randn_like(I) * std
        return torch.clamp(I + noise, 0.0, 1.0)

    def rgb_to_ycbcr(self, I):
        """Convert RGB to YCbCr and back - tests color channel resilience."""
        r, g, b = I[:, 0:1, :, :], I[:, 1:2, :, :], I[:, 2:3, :, :]
        
        # RGB to YCbCr
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = (b - y) / 1.772 + 0.5
        cr = (r - y) / 1.402 + 0.5
        
        # YCbCr back to RGB
        cb_shift = cb - 0.5
        cr_shift = cr - 0.5
        r_out = y + 1.402 * cr_shift
        g_out = y - 0.114 * 1.772 / 0.587 * cb_shift - 0.299 * 1.402 / 0.587 * cr_shift
        b_out = y + 1.772 * cb_shift
        
        rgb = torch.cat([r_out, g_out, b_out], dim=1)
        return torch.clamp(rgb, 0.0, 1.0)

    def apply_instagram_filter(self, I):
        x_contrast = 0.5 - 0.5 * torch.cos(I * 3.14159265)
        mixed = I * 0.5 + x_contrast * 0.5
        r, g, b = mixed[:, 0:1, :, :], mixed[:, 1:2, :, :], mixed[:, 2:3, :, :]
        r_new = r + 0.15 * r * (1.0 - r)
        b_new = b + 0.15 * (1.0 - b) * b
        filtered = torch.cat([r_new, g, b_new], dim=1)
        return torch.clamp(filtered, 0.0, 1.0)

    def forward(self, I_w):
        img = I_w.clone()
        
        # ACTIVE BENIGN TRANSFORMS (Spatial-alignment compatible)
        # ✓ JPEG (Q=25/50/75): Drops high-freq, our low AC[0,1,2] survive
        # ✓ Gaussian Blur: Spatial patterns recoverable with strong signal
        # ✓ Color Jitter: Per-channel independent, alignment preserved
        # ✓ Instagram Filter: Similar to color jitter, preserves alignment
        # ✓ YCbCr Conversion: Tests color channel resilience, preserves positions
        # ✗ REMOVED: Rotation, Geometric Scaling, HFlip, Crop (break alignment)
            
        # Blur or Noise (mutually exclusive)
        p_filter = random.random()
        if p_filter < 0.1:
            img = self.blur_3x3(img)
        elif p_filter < 0.2:
            img = self.blur_5x5(img)
        elif p_filter < 0.3:
            img = self.add_gaussian_noise(img)
        
        # Color space conversion (25% chance)
        if random.random() < 0.25:
            img = self.rgb_to_ycbcr(img)
        
        # Color jitter (30% chance)
        if random.random() < 0.3:
            img = self.color_jitter(img)
        
        # Instagram filter (20% chance)
        if random.random() < 0.2:
            img = self.apply_instagram_filter(img)
            
        # JPEG compression (50% chance, mutually exclusive quality)
        if random.random() < 0.5:
            p_jpeg = random.random()
            if p_jpeg < 0.33:
                img = self.jpeg_25(img)
            elif p_jpeg < 0.66:
                img = self.jpeg_50(img)
            else:
                img = self.jpeg_75(img)
                
        return img


class MalignAttackGenerator(nn.Module):
    """
    Simulates identity-changing deepfakes.
    """
    def __init__(self, device='cpu', feather_kernel=15, feather_sigma=3.0, use_diffusion=True):
        super().__init__()
        self.device = device
        self.feather_kernel = feather_kernel
        self.feather_sigma = feather_sigma
        self.use_diffusion = use_diffusion
        self.pipe = None
        self._sd_load_attempted = False

    def _try_load_diffusion(self):
        if self._sd_load_attempted:
            return
        self._sd_load_attempted = True
        try:
            from diffusers import StableDiffusionInpaintPipeline
            model_id = "runwayml/stable-diffusion-inpainting"
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            self.pipe.set_progress_bar_config(disable=True)
            self.pipe.to(self.device)
            print("[*] SD pipeline loaded.")
        except Exception as e:
            print(f"[!] SD unavailable. Falling back.")
            self.use_diffusion = False

    def soft_mask(self, M):
        return F_kornia.gaussian_blur2d(M, (self.feather_kernel, self.feather_kernel), 
                                        (self.feather_sigma, self.feather_sigma))

    def semantic_face_swap(self, I_w, I_donor, M):
        smooth_mask = self.soft_mask(M)
        attacked = I_w * (1.0 - smooth_mask) + I_donor * smooth_mask
        return torch.clamp(attacked, 0, 1)

    def latent_distorter(self, I_w, M):
        B, C, H, W = I_w.shape
        noise = torch.randn((B, C, H // 4, W // 4), device=self.device)
        noise = torch.nn.functional.interpolate(noise, size=(H, W), mode='bilinear')
        perturbation = noise * 0.4
        smooth_mask = self.soft_mask(M)
        attacked = I_w + (perturbation * smooth_mask)
        return torch.clamp(attacked, 0, 1)

    def generative_inpaint(self, I_w, M):
        if self.use_diffusion and self.pipe is None:
            self._try_load_diffusion()
        if not self.use_diffusion or self.pipe is None:
            return self.latent_distorter(I_w, M)
            
        B = I_w.shape[0]
        result_batch = []
        import kornia
        
        with torch.no_grad():
            for i in range(B):
                img_t = I_w[i].detach().cpu()
                mask_t = M[i].detach().cpu()
                pil_img = kornia.utils.tensor_to_image(img_t)
                pil_img = Image.fromarray((pil_img * 255).astype('uint8'))
                pil_mask = kornia.utils.tensor_to_image(mask_t).squeeze()
                pil_mask = Image.fromarray((pil_mask * 255).astype('uint8'))
                
                prompt = "A high resolution, detailed portrait photograph of a face."
                out = self.pipe(prompt=prompt, image=pil_img, mask_image=pil_mask).images[0]
                out_tensor = kornia.utils.image_to_tensor(out, keepdim=False).float() / 255.0
                out_tensor = out_tensor.to(I_w.device)
                result_batch.append(out_tensor)
                
        if len(result_batch) == B:
            return torch.stack(result_batch, dim=0)
        return self.latent_distorter(I_w, M)

    def forward(self, I_w, I_donor, M):
        val = random.random()
        # Latent distortion: 50% (fast, learnable gradients)
        if val < 0.50:
            return self.latent_distorter(I_w, M)
        # Semantic face swap: 40% (realistic, learnable gradients)
        elif val < 0.90:
            return self.semantic_face_swap(I_w, I_donor, M)
        # Generative inpainting: 10% (expensive, non-learnable but realistic)
        # NOTE: Generative inpaint is non-differentiable (uses Stable Diffusion)
        # but decoder still learns to fail on it via L_RE loss without backprop
        else:
            return self.generative_inpaint(I_w, M)
