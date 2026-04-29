# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.engine.math import BlockDCT, BlockIDCT
from src.models.utils import ResBlock

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
        # Dedicated bit injection path
        self.bit_injector = nn.Conv2d(n_bits, 3, kernel_size=1)
        # bit signal strength 0.06 (Refinement phase: reducing intensity now that decoder has acquisition)
        nn.init.constant_(self.bit_injector.weight, 0.06)
        nn.init.constant_(self.bit_injector.bias, 0)
        
    def forward(self, img, M, bits):
        """
        Spatially localized bit embedding (8x8 grid for 64 bits).
        Each bit encoded in specific spatial region for independent decoding.
        """
        B, C, H, W = img.shape
        n_bits = bits.shape[1]
        grid_size = int(n_bits ** 0.5)  # 8 for 64 bits
        
        # Create localized bit embeddings
        polarized_bits = (bits.float() * 2.0) - 1.0  # -1 or +1 per bit
        
        # LOCALIZATION: Create per-region patterns
        # Divide image into grid_size x grid_size regions
        region_h = H // grid_size
        region_w = W // grid_size
        
        # For each bit, create signal in its spatial region
        bits_map = torch.zeros(B, n_bits, H, W, device=img.device)
        for bit_idx in range(n_bits):
            grid_row = bit_idx // grid_size
            grid_col = bit_idx % grid_size
            
            start_h = grid_row * region_h
            end_h = (grid_row + 1) * region_h if grid_row < grid_size - 1 else H
            start_w = grid_col * region_w
            end_w = (grid_col + 1) * region_w if grid_col < grid_size - 1 else W
            
            # Fill region with polarized bit value (broadcast to spatial dims)
            bit_value = polarized_bits[:, bit_idx, None, None]  # [B] -> [B, 1, 1]
            bits_map[:, bit_idx, start_h:end_h, start_w:end_w] = bit_value
        
        # Inject bits via 1x1 conv (now each bit has spatial localization)
        res = self.bit_injector(bits_map) * 0.5
        return res * M

class FrequencyEncoder(nn.Module):
    """
    Embeds bits into AC coefficients of DCT space.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.dct = BlockDCT(block_size=8)
        self.idct = BlockIDCT(block_size=8)
        
        # 192 frequency channels (3 colors * 8 * 8) + n_bits
        self.ac_residual_conv = nn.Sequential(
            nn.Conv2d(192 + n_bits, 256, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 192, kernel_size=1)
        )
        self.bit_injector = nn.Conv2d(n_bits, 192, kernel_size=1)
        # bit signal strength 0.06
        nn.init.constant_(self.bit_injector.weight, 0.06)
        nn.init.constant_(self.bit_injector.bias, 0)

    def _get_ac_mask(self, x):
        """
        Batch-safe and Device-safe mask generation.
        Zeroes out DC component of every 8x8 block.
        """
        mask = torch.ones_like(x)
        mask[:, :, 0::8, 0::8] = 0
        return mask
        
    def forward(self, img, M, bits):
        """
        SPATIALLY-ALIGNED frequency domain embedding.
        Bit (row, col) → DCT block (row, col) + AC coefficients [0,1,2] (R,G,B)
        """
        B, C, H, W = img.shape
        n_bits = bits.shape[1]
        grid_size = int(n_bits ** 0.5)  # 8 for 64 bits
        
        # 1. Transform to DCT domain [B, 3, H, W]
        dct_coeffs = self.dct(img)  # [B, 3, H, W]
        
        # 2. Reshape into frequency channels: [B, 192, H//8, W//8]
        # Where 192 = 3 colors × 64 DCT coefficients per 8×8 block
        x_reshaped = dct_coeffs.view(B, 3, H//8, 8, W//8, 8)
        x_reshaped = x_reshaped.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, 192, H//8, W//8)
        
        # 3. SPATIAL-FREQUENCY ALIGNMENT: Map bits to their corresponding DCT blocks
        polarized_bits = (bits.float() * 2.0) - 1.0  # -1 or +1 per bit
        
        # For each bit at grid position (row, col):
        # - Embed in DCT block at position (row, col)
        # - Use AC coefficients [0, 1, 2] (one per color channel)
        for bit_idx in range(n_bits):
            grid_row = bit_idx // grid_size
            grid_col = bit_idx % grid_size
            
            # SPATIALLY ALIGN frequency bits to the 4x4 DCT block regions
            bit_value = polarized_bits[:, bit_idx:bit_idx+1]  # (B, 1)
            start_r = grid_row * 4
            end_r = (grid_row + 1) * 4
            start_c = grid_col * 4
            end_c = (grid_col + 1) * 4
            
            # Apply to AC coefficients [1] in this DCT region
            x_reshaped[:, 1, start_r:end_r, start_c:end_c] += bit_value.unsqueeze(2) * 0.5
            x_reshaped[:, 65, start_r:end_r, start_c:end_c] += bit_value.unsqueeze(2) * 0.5
            x_reshaped[:, 129, start_r:end_r, start_c:end_c] += bit_value.unsqueeze(2) * 0.5
        
        # 4. Inverse reshape back to [B, 3, H, W]
        ac_res_spatial = x_reshaped.view(B, 3, 8, 8, H//8, W//8)
        ac_res_spatial = ac_res_spatial.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, 3, H, W)
        
        # 5. Enforce updates only to AC coefficients
        ac_mask = self._get_ac_mask(dct_coeffs)
        modified_dct = dct_coeffs + (ac_res_spatial - dct_coeffs) * ac_mask
        
        # 6. Reverse transform and return residual
        freq_out = self.idct(modified_dct)
        
        # 7. Apply face mask to localize frequency bits to face only (consistent with spatial encoder)
        return (freq_out - img) * M

class HybridEncoder(nn.Module):
    def __init__(self, n_bits, encoder_scale=1.0):
        """
        encoder_scale: controls bit capacity vs invisibility tradeoff.
        """
        super().__init__()
        self.encoder_scale = encoder_scale
        self.spatial_encoder = SpatialEncoder(n_bits=n_bits)
        self.freq_encoder = FrequencyEncoder(n_bits=n_bits)

        # expressive fusion block
        self.fusion = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),  # Final 1x1 projection
        )
        
        # Initialize fusion weights to properly blend spatial and frequency signals
        nn.init.kaiming_normal_(self.fusion[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fusion[2].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.normal_(self.fusion[4].weight, std=0.02)  # Final projection: small random init
        for layer in self.fusion:
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)

    def forward(self, img, M, bits):
        res_s = self.spatial_encoder(img, M, bits)  # [B, 3, H, W]
        res_f = self.freq_encoder(img, M, bits)    # [B, 3, H, W]

        # Concatenate spatial and frequency residuals WITH the original image
        concat_res = torch.cat([img, res_s, res_f], dim=1)  # [B, 9, H, W]
        
        # Learn optimal image-adaptive fusion
        fused_res = self.fusion(concat_res)  # [B, 3, H, W]

        # Scale and embed: 
        # 1. Directly add beacons so they aren't scrambled by random fusion init
        # 2. Apply tanh to fused_res and bound it to 0.15. This prevents the 'Clamp Trap' 
        #    where massive noise saturates the image beyond [0.0, 1.0] and kills the L1 gradients.
        bounded_fusion = torch.tanh(fused_res) * 0.15
        watermarked = img + self.encoder_scale * (bounded_fusion + res_s + res_f)
        
        return torch.clamp(watermarked, 0.0, 1.0)

class SpatialDecoder(nn.Module):
    """
    Decodes bits from spatial domain.
    Uses GroupNorm for WGAN-GP compatibility.
    Deepened to 4 strided layers + ResBlocks + 8x8 spatial preservation.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
            ResBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            ResBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),
            ResBlock(256),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
    def forward(self, img):
        x = self.net(img)
        return x

class FrequencyDecoder(nn.Module):
    """
    Decodes bits from frequency domain (DCT block AC coefficients).
    Deepened to 4 strided layers + ResBlocks + 8x8 spatial preservation.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.dct = BlockDCT(block_size=8)
        self.net = nn.Sequential(
            # Input is [B, 192, 32, 32]
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),
            ResBlock(256),
            
            # [B, 256, 32, 32] -> [B, 512, 16, 16]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2),
            ResBlock(512),
            
            # [B, 512, 16, 16] -> [B, 512, 8, 8]
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
    def forward(self, img):
        B, C, H, W = img.shape
        dct_coeffs = self.dct(img)
        
        # Reshape to prevent spectral blindness
        x_reshaped = dct_coeffs.view(B, 3, H//8, 8, W//8, 8)
        x_reshaped = x_reshaped.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, 192, H//8, W//8)
        
        x = self.net(x_reshaped)
        return x

class HybridDecoder(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.grid_size = int(n_bits ** 0.5)  # 8 for 64 bits
        
        self.spatial_decoder = SpatialDecoder(n_bits=n_bits)
        self.freq_decoder = FrequencyDecoder(n_bits=n_bits)
        
        # SPATIAL-AWARE extraction: one local classifier per bit region
        # Each bit gets its own small FC network from its corresponding spatial region
        self.spatial_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 * 1 * 1, 128),  # 512 features from 1x1 avg pool of region
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)  # Single logit for this bit
            ) for _ in range(n_bits)
        ])
        
        self.freq_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 * 1 * 1, 128),  # 512 features from frequency region
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)  # Single logit for this bit
            ) for _ in range(n_bits)
        ])
        
        # Global fusion for bits (aggregate all local signals)
        self.fusion_fc = nn.Sequential(
            nn.Linear(n_bits * 2, 256),  # 64 spatial + 64 freq logits
            nn.ReLU(inplace=True),
            nn.Linear(256, n_bits)  # Final refined bit predictions (logits)
        )
        
    def forward(self, img, return_aux=False):
        B, C, H, W = img.shape
        
        # Extract features
        s_feat = self.spatial_decoder(img)  # [B, 512, 8, 8]
        f_feat = self.freq_decoder(img)    # [B, 512, 8, 8]
        
        # SPATIAL EXTRACTION: Extract bits from their corresponding regions
        # s_feat and f_feat are [B, 512, 8, 8], matching the encoder's 8x8 bit grid
        spatial_logits = []
        freq_logits = []
        
        for bit_idx in range(self.n_bits):
            # Map bit index to 8x8 grid position
            bit_row = bit_idx // self.grid_size  # 0-7
            bit_col = bit_idx % self.grid_size   # 0-7
            
            # Extract the exact spatial cell for this bit
            s_region = s_feat[:, :, bit_row:bit_row+1, bit_col:bit_col+1]  # [B, 512, 1, 1]
            f_region = f_feat[:, :, bit_row:bit_row+1, bit_col:bit_col+1]  # [B, 512, 1, 1]
            
            # Flatten and classify
            s_region_flat = s_region.view(B, -1)  # [B, 512]
            f_region_flat = f_region.view(B, -1)  # [B, 512]
            
            s_logit = self.spatial_extractors[bit_idx](s_region_flat)  # [B, 1]
            f_logit = self.freq_extractors[bit_idx](f_region_flat)    # [B, 1]
            
            spatial_logits.append(s_logit)
            freq_logits.append(f_logit)
        
        # Stack all logits [B, 64] from spatial and frequency
        spatial_logits = torch.cat(spatial_logits, dim=1)  # [B, 64]
        freq_logits = torch.cat(freq_logits, dim=1)        # [B, 64]
        
        # Fuse spatial and frequency predictions
        combined = torch.cat([spatial_logits, freq_logits], dim=1)  # [B, 128]
        out_logits = self.fusion_fc(combined)  # [B, 64] - refined logits
        
        if return_aux:
            return out_logits, spatial_logits + freq_logits  # Auxiliary: sum of spatial+freq
        
        return out_logits  # Return raw logits for BCEWithLogitsLoss (more stable than sigmoid+BCE)