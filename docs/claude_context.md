# Project Context: Hybrid Spatial-Frequency Deepfake Watermarking

## 1. High-Level Objective
We are building a **Semi-Fragile Watermarking System** designed to detect deepfakes and identity theft. 
The system embeds a 32-bit payload (BER target < 2%) into images. It must satisfy the **Robustness-Fragility Paradox**:
*   **Robustness:** The watermark must survive benign transmission augmentations (JPEG compression, Gaussian noise, blur).
*   **Fragility:** The watermark must strictly reside within the **facial landmarks** (defined by an MTCNN mask `M`). If a deepfake replaces the face, the watermark must vanish entirely, resulting in a ~50% BER, which flags the image as forged.

## 2. Neural Architecture
The system employs a 3-way adversarial setup:
1.  **Encoder (`HybridEncoder`):** Takes `img`, mask `M`, and `bits`. Uses a `SpatialEncoder` and a `FrequencyEncoder` (DCT). It fuses them and strictly multiplies by `M` so the watermark only exists on the face.
    *   Output generation: `torch.clamp(img + encoder_scale * torch.tanh(fused_res), 0.0, 1.0)`
2.  **Decoder (`HybridDecoder`):** A deep, 5-stride CNN utilizing `ResBlock` layers and `AdaptiveAvgPool2d((4,4))` (to avoid diluting the signal with the unwatermarked background). 
3.  **Adversary (`AdversaryNet`):** A CNN that actively tries to remove the watermark without destroying image quality.
4.  **Critic (`WassersteinCritic`):** A PatchGAN that enforces visual realism using WGAN-GP.

## 3. The 4-Phase Curriculum
The training (`train_main.py`) uses a BER-gated curriculum:
*   **Phase 1 (Bit Establishment):** No augmentations, no quality losses. Goal is simply to get BER < 15%.
*   **Phase 2a (Aug Robustness):** Linearly ramps up benign augmentations.
*   **Phase 2b (Quality Ramp):** Linearly ramps up image quality losses (L1, SSIM, LPIPS) and ramps down `encoder_scale` to target (0.03).
*   **Phase 3 (Arms Race):** Activates the `AdversaryNet` removal attacks.

## 4. The Current Problem (The "Dead-Clamp" Anomaly)
We are currently stuck in **Phase 1**. 
*   The raw Bit Error Rate (BER) is permanently stuck at ~50% (random chance).
*   The **PSNR is exactly 7.70 dB**, and it does not change across epochs. 

### Our Current Diagnosis
We suspect a **Gradient Collapse / Dead-Clamp** issue. 
In `train_main.py`, the curriculum forcefully overrides the `encoder_scale` to `1.0` during Phase 1 to maximize signal strength.
Because `W_L1` and `W_LPIPS` are `0.0`, the Encoder pushes `fused_res` to extreme values. 
When `torch.clamp(img + 1.0 * torch.tanh(fused_res), 0.0, 1.0)` executes, almost every pixel on the face hits the `0.0` or `1.0` hard boundaries. 
The mask covers ~56% of the image. If 56% of the image is pure white or pure black, the mathematical MSE results in exactly **7.74 dB PSNR**. 
Once clamped, the gradient drops to exactly 0.0. The Encoder's pixels "die," it stops learning, and the Decoder is left staring at a blank, solid-color face, unable to extract any bits (hence 50% BER).

---

## 5. Relevant Code Snippets

### A. The Encoder Fusion (`src/models/hybrid_model.py`)
```python
    def forward(self, img, M, bits):
        res_s = self.spatial_encoder(img, M, bits)
        res_f = self.freq_encoder(img, bits)
        fused_res = self.fusion(torch.cat([res_s, res_f], dim=1))
        
        # Identity-Aware Gate: Watermark strictly restricted to face
        fused_res = fused_res * M
        
        # This is where we suspect the clamp kills the gradients if encoder_scale=1.0
        return torch.clamp(img + self.encoder_scale * torch.tanh(fused_res), 0.0, 1.0)
```

### B. The Curriculum Logic (`train_main.py`)
```python
        if current_phase == 1:
            if epoch <= 15 or last_eval_ber > 0.30:
                # Suspected Bug: Forcing scale to 1.0 causes clamp death
                trainer.encoder.encoder_scale    = 1.0   
                trainer.criterion.W_L1           = 0.0
                trainer.criterion.W_LPIPS        = 0.0
                trainer.criterion.W_SSIM         = 0.0
                trainer.criterion.W_ID           = 0.0
                trainer.criterion.W_BIT_BENIGN   = 15.0  # Only loss active
                trainer.aug_prob                 = 0.0
```

### C. The Base Transform (`src/data/transforms.py`)
```python
def get_base_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), # Maps image to [0.0, 1.0]
    ])
```
