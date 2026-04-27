# Technical Architecture & Implementation Specifications
**Project:** Hybrid Spatial-Frequency Identity-Aware Watermarking

This document provides a highly detailed breakdown of the layer configurations, adversarial curriculum, and mathematical dynamics of the watermarking model. It is designed to be fed into AI models for structural analysis, debugging, and optimization.

---

## 1. Mathematical Dynamics: BER, PSNR, & The "Dead Clamp" Anomaly

### The "Dead Clamp" Gradient Collapse (Resolved)
**Symptom:** During early development, the network was permanently trapped at **50% BER** and exactly **7.70 dB PSNR**.
**Mathematical Cause:** 
The Phase 1 curriculum originally overrode the `encoder_scale` to `1.0` (aiming for maximum signal strength) while disabling visual quality losses (`W_L1 = 0`). The Encoder pushed massive signals (e.g., $+1.0$ or $-1.0$) into `fused_res`. 
The final output equation is:
```python
I_w = torch.clamp(img + encoder_scale * torch.tanh(fused_res), 0.0, 1.0)
```
Because `img` is inherently `[0, 1]`, adding $\pm 1.0$ immediately slammed the pixels into the `0.0` or `1.0` `clamp` boundaries. 
In PyTorch, the derivative (gradient) of a clamped value outside the bounds is **exactly zero**. The gradients collapsed, the Encoder stopped learning instantly, and it painted the entire 56% facial mask solid white or black. 
*(Proof: A solid white/black block covering 56% of an image mathematically yields an MSE resulting in precisely 7.74 dB PSNR).*

**The Fix:** We permanently locked `encoder_scale` to `0.03`. At a maximum shift of $\pm 0.03$, pixels rarely hit the bounds, gradients flow freely, and the baseline **PSNR mathematically cannot fall below ~30.45 dB**.

### Deepfake Fragility & The >70% BER Trick
**Symptom:** SOTA models achieve ~50% BER on deepfakes (random chance). The user target is **>70% BER**.
**Mathematical Solution:**
If the watermark is perfectly destroyed, BER is 50%. To get >70%, the network must actively output the *wrong* bits.
We achieved this by setting the fragility loss to:
```python
loss_bit_fragile = bce(pred_attacked, 1.0 - msg_encoded)
```
During a simulated FaceSwap (`MalignAttackGenerator`), a `soft_mask` (Gaussian Blur) is used to blend the donor face over the original face. 
This forces the Encoder to mathematically hide the **true** message in the center of the face, and the **inverse** message `(1.0 - msg_encoded)` in the feathered edges. When a Deepfake deletes the center, the Decoder reads the surviving edges, outputting the inverse bits and yielding a **near 100% BER**.

---

## 2. Model Architectures & Layer Configurations

### A. HybridEncoder
The Encoder fuses spatial and frequency residuals, strictly gating them within the facial landmarks (`M`).

1. **SpatialEncoder (Inputs: 36 channels `[img, M, bits_map]`)**
   * `Conv2d(36, 64, k=3, p=1)` -> `ReLU`
   * 3x `ResBlock(64)` (Consisting of 2x `Conv2d(64,64)` + `ReLU`)
   * `Conv2d(64, 3, k=3, p=1)` -> Output `r_s`

2. **FrequencyEncoder (Inputs: 35 channels `[img_dct, bits_map]`)**
   * DCT Block Transform (8x8 blocks)
   * `Conv2d(35, 64, k=3, p=1)` -> `ReLU`
   * `Conv2d(64, 3, k=3, p=1)` -> `ac_res`
   * Mask zeroes out DC coefficients -> `ac_res * ac_mask`
   * IDCT Inverse Transform -> Output `r_f`

3. **Fusion & Output Gate**
   * `Conv2d(6, 64)` -> `ReLU` -> `Conv2d(64, 32)` -> `ReLU` -> `Conv2d(32, 3)` -> `fused_res`
   * **Strict Identity Gate:** `fused_res = fused_res * M` (forces fragility)
   * **Output:** `torch.clamp(img + 0.03 * torch.tanh(fused_res), 0, 1)`

### B. HybridDecoder
A highly parameterized decoder to survive Adversary Net distortions. Replaced Global Average Pooling with `AdaptiveAvgPool2d((4,4))` to prevent background noise from drowning out the facial signal.

1. **SpatialDecoder (Inputs: 3 channels)**
   * `Conv2d(3, 64, k=3, stride=2, p=1)` -> `GroupNorm(8)` -> `ReLU` -> `ResBlock(64)`
   * `Conv2d(64, 128, k=3, stride=2, p=1)` -> `GroupNorm(8)` -> `ReLU` -> `ResBlock(128)`
   * `Conv2d(128, 256, k=3, stride=2, p=1)` -> `GroupNorm(8)` -> `ReLU` -> `ResBlock(256)`
   * `Conv2d(256, 512, k=3, stride=2, p=1)` -> `GroupNorm(8)` -> `ReLU` -> `ResBlock(512)`
   * `AdaptiveAvgPool2d((4, 4))` -> Flatten to 8192 parameters.

2. **FrequencyDecoder (Inputs: 3 channels)**
   * Identical architecture to SpatialDecoder, but operates on DCT coefficients.

3. **Fusion Classifier**
   * `Linear(8192 + 8192, 512)` -> `ReLU`
   * `Linear(512, 32)` -> `Sigmoid`

### C. Wasserstein Critic (Discriminator)
A PatchGAN architecture using WGAN-GP (Gradient Penalty) to enforce realism.
* 4x Strided Convolutions (`stride=2`) ramping from 64 to 512 filters.
* Uses `LeakyReLU(0.2)`.
* **Important:** No `BatchNorm` is used (replaced by `GroupNorm` or InstanceNorm), as BatchNorm breaks WGAN-GP independent gradient tracking.

### D. Internal Adversary (Removal Attack Network)
A U-Net style AutoEncoder designed to wipe the watermark.
* **Encoder:** 3 strided layers (32 -> 64 -> 128 filters)
* **Bottleneck:** 2x `ResBlock(128)`
* **Decoder:** 3 Transpose Convolutions (128 -> 64 -> 32 -> 3)
* Bound by `epsilon=0.05`: `torch.clamp(I_w + 0.05 * torch.tanh(perturbation), 0, 1)`

---

## 3. Loss Configuration (The 8-Component Framework)
The Generator (Encoder+Decoder) minimizes a complex weighted sum:
1. `W_L1` (1.0): Pixel-wise Absolute Error.
2. `W_LPIPS` (1.0): VGG-16 Perceptual distance.
3. `W_SSIM` (0.5): Structural Similarity.
4. `W_ID` (0.5): FaceNet Embedding MSE (maintains biometric geometry).
5. `W_BIT_BENIGN` (20.0): BCE Loss for Bit Recovery (Robustness).
6. `W_BIT_FRAGILE` (5.0): Inverse BCE Loss for Bit Destruction (Fragility).
7. `W_DISC` (0.5): Wasserstein Fake Score `-E[C(I_w)]` (Realism).
8. `W_ADV` (0.5): BCE Loss against AdversaryNet (Resilience).
