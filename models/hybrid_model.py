# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.math import BlockDCT, BlockIDCT
from models.utils import ResBlock

class SpatialEncoder(nn.Module):
    """
    CNN that spatially embeds message bits into facial region.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.conv_in = nn.Conv2d(3 + 1 + n_bits, 64, kernel_size=3, padding=1)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, img, M, bits):
        B, C, H, W = img.shape
        bits_map = bits.view(B, -1, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([img, M, bits_map], dim=1)
        
        x = F.relu(self.conv_in(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        r_s = self.conv_out(x)
        
        return r_s * M

class FrequencyEncoder(nn.Module):
    """
    Embeds bits into AC coefficients of DCT space.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.dct = BlockDCT(block_size=8)
        self.idct = BlockIDCT(block_size=8)
        
        self.ac_residual_conv = nn.Sequential(
            nn.Conv2d(3 + n_bits, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def _get_ac_mask(self, x):
        """
        Batch-safe and Device-safe mask generation.
        Zeroes out DC component of every 8x8 block.
        """
        # Create a mask of 1s on the correct device and same shape as input
        mask = torch.ones_like(x)
        
        # Strided Indexing: [Batch, Channel, Height, Width]
        # Start at 0, skip every 8 pixels. This zeroes the (0,0) of every 8x8 block.
        mask[:, :, 0::8, 0::8] = 0
        
        return mask
        
    def forward(self, img, bits):
        B, C, H, W = img.shape
        
        # 1. Transform to DCT domain
        dct_coeffs = self.dct(img)
        
        # 2. Broadcast message bits
        bits_map = bits.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # 3. Predict frequency residual
        x = torch.cat([dct_coeffs, bits_map], dim=1)
        ac_res = self.ac_residual_conv(x)
        
        # 4. Enforce updates only to AC coefficients using the fixed mask
        ac_mask = self._get_ac_mask(dct_coeffs)
        modified_dct = dct_coeffs + (ac_res * ac_mask)
        
        # 5. Reverse transform and return residual
        freq_out = self.idct(modified_dct)
        return freq_out - img

class HybridEncoder(nn.Module):
    def __init__(self, n_bits, encoder_scale=0.2):
        """
        encoder_scale: controls bit capacity vs invisibility tradeoff.
        """
        super().__init__()
        self.encoder_scale = encoder_scale
        self.spatial_encoder = SpatialEncoder(n_bits=n_bits)
        self.freq_encoder = FrequencyEncoder(n_bits=n_bits)

        # expressive fusion block
        self.fusion = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),  # Final 1x1 projection
        )

    def forward(self, img, M, bits):
        res_s = self.spatial_encoder(img, M, bits)
        res_f = self.freq_encoder(img, bits)

        # Non-linear fusion of both residual streams
        fused_res = self.fusion(torch.cat([res_s, res_f], dim=1))

        # Scale is now config-driven, not hardcoded
        return img + (self.encoder_scale * fused_res)

class SpatialDecoder(nn.Module):
    """
    Decodes bits from spatial domain.
    Uses GroupNorm for WGAN-GP compatibility.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, n_bits)
        
    def forward(self, img):
        x = self.net(img)
        return self.fc(x.view(x.size(0), -1))

class FrequencyDecoder(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.dct = BlockDCT(block_size=8)
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Increased complexity to help with bit recovery
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, n_bits)
        
    def forward(self, img):
        dct_coeffs = self.dct(img)
        x = self.net(dct_coeffs)
        return self.fc(x.view(x.size(0), -1))

class HybridDecoder(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.spatial_decoder = SpatialDecoder(n_bits=n_bits)
        self.freq_decoder = FrequencyDecoder(n_bits=n_bits)
        self.fusion_fc = nn.Sequential(
            nn.Linear(n_bits * 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_bits)
        )
        
    def forward(self, img):
        s_bits = self.spatial_decoder(img)
        f_bits = self.freq_decoder(img)
        out = self.fusion_fc(torch.cat([s_bits, f_bits], dim=1))
        return torch.sigmoid(out)