# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BlockDCT(nn.Module):
    """
    Differentiable 2D DCT for 8x8 blocks.
    """
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        D = self._get_dct_matrix(block_size)
        self.register_buffer('D', D)

    def _get_dct_matrix(self, N):
        D = torch.zeros(N, N)
        for i in range(N):
            c = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
            for j in range(N):
                D[i, j] = c * np.cos((2 * j + 1) * i * np.pi / (2 * N))
        return D

    def forward(self, x):
        # Buffer synchronization
        D = self.D.to(device=x.device, dtype=x.dtype)
        
        B, C, H, W = x.shape
        N = self.block_size
        x_blocks = x.view(B, C, H // N, N, W // N, N).permute(0, 1, 2, 4, 3, 5)
        
        # dct_blocks = D * x_blocks * D_T
        dct_blocks = torch.einsum('ik, ...kl, jl -> ...ij', D, x_blocks, D)
        return dct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)


class BlockIDCT(nn.Module):
    """
    Differentiable Inverse 2D DCT for 8x8 blocks.
    """
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        D = self._get_dct_matrix(block_size)
        self.register_buffer('D', D)

    def _get_dct_matrix(self, N):
        D = torch.zeros(N, N)
        for i in range(N):
            c = np.sqrt(1 / N) if i == 0 else np.sqrt(2 / N)
            for j in range(N):
                D[i, j] = c * np.cos((2 * j + 1) * i * np.pi / (2 * N))
        return D

    def forward(self, x):
        # Buffer synchronization
        D = self.D.to(device=x.device, dtype=x.dtype)
        
        B, C, H, W = x.shape
        N = self.block_size
        x_blocks = x.view(B, C, H // N, N, W // N, N).permute(0, 1, 2, 4, 3, 5)
        
        # idct_blocks = D_T * x_blocks * D
        idct_blocks = torch.einsum('ki, ...kl, lj -> ...ij', D, x_blocks, D)
        return idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)


class DiffJPEG(nn.Module):
    """
    Differentiable JPEG Simulation.
    """
    def __init__(self, quality=50):
        super().__init__()
        self.dct = BlockDCT(8)
        self.idct = BlockIDCT(8)
        
        # Standard JPEG Luma Quantization Matrix
        q_luma = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32)
        
        # Scale by perceived quality factor
        scale = 5000 / quality if quality < 50 else 200 - quality * 2
        q_matrix = torch.floor((q_luma * scale + 50) / 100).clamp(min=1)
        
        # Reshape to easily broadcast over B, C, (H/8), (W/8), 8, 8
        self.register_buffer('q_matrix', q_matrix.view(1, 1, 1, 1, 8, 8))

    def forward(self, img_orig):
        # Synchronize q_matrix
        q_matrix = self.q_matrix.to(device=img_orig.device, dtype=img_orig.dtype)
        
        B, C, H, W = img_orig.shape
        
        # 1. Transform to frequency space
        dct_layer = self.dct(img_orig)
        
        # 2. Blockify for quantization
        blocks = dct_layer.view(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 3, 5)
        
        # 3. Quantization with STE
        scaled = blocks / q_matrix
        quantized = scaled + (torch.round(scaled) - scaled).detach()
        
        # 4. De-Quantization
        dequantized = quantized * q_matrix
        
        # Restore to standard shape and IDCT
        dct_restored = dequantized.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        
        return self.idct(dct_restored)


class FunctionalBlockCodec:
    """
    Linear Block Code.
    """
    def __init__(self, k, n, device='cpu'):
        self.k = k
        self.n = n
        self.parity_len = n - k
        
        # Deterministic Parity Matrix
        np.random.seed(42)
        P_np = np.random.randint(0, 2, (k, self.parity_len)).astype(np.float32)
        self.P = torch.tensor(P_np, device=device)
        
        I_k = torch.eye(k, device=device)
        self.G = torch.cat([I_k, self.P], dim=1) # (k, n)
        
        I_p = torch.eye(self.parity_len, device=device)
        self.H = torch.cat([self.P.T, I_p], dim=1) # (parity_len, n)
        
        # Syndrome Mapping
        self.syndrome_table = {}
        for err_index in range(n):
            e = torch.zeros(n, device=device)
            e[err_index] = 1.0
            s = torch.matmul(e, self.H.T) % 2
            s_hash = "".join(map(str, s.int().tolist()))
            self.syndrome_table[s_hash] = err_index

def encode_message_bch(bits, codec):
    """Multiplies bits by Generator Matrix G."""
    return torch.matmul(bits, codec.G) % 2

def decode_message_bch(rx_bits, codec):
    """Systematic Syndrome decoding."""
    B = rx_bits.shape[0]
    corrected = rx_bits.clone()
    syndromes = torch.matmul(rx_bits, codec.H.T) % 2
    
    for i in range(B):
        s_hash = "".join(map(str, syndromes[i].int().tolist()))
        if "1" in s_hash: 
            if s_hash in codec.syndrome_table:
                err_index = codec.syndrome_table[s_hash]
                corrected[i, err_index] = 1 - corrected[i, err_index]
            
    return corrected[:, :codec.k]


