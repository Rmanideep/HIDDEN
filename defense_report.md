# PhD Defense Report: Hybrid Spatial-Frequency Identity-Aware Watermarking
### In-Depth Technical Analysis of Every Design Decision

> **Prepared for:** Thesis Defense Committee
> **Project:** Hybrid Spatial-Frequency Identity-Aware Watermarking Framework
> **Primary Codebase:** `c:/Users/Harsha/Desktop/watermark/`

---

## Defense Pillar I — The Domain Split: Mathematical Justification

### The Core Hypothesis

The central theoretical claim of this architecture is that **no single signal domain is simultaneously optimal for both robustness and fragility**. These two objectives are, by their information-theoretic nature, antithetical within a unified representation space. The Hybrid Encoder resolves this by decomposing the watermarking problem across two orthogonal representation domains.

### 1.1 The Frequency Branch: Global Robustness via DCT

The `FrequencyEncoder` in `models/hybrid_model.py` (lines 33–81) operates in the **Discrete Cosine Transform (DCT)** domain. The mathematical justification rests on three pillars:

**The DCT Energy Compaction Theorem.** For any natural image `I`, the 2D DCT concentrates the majority of signal energy into a small number of low-frequency coefficients. The transformation is defined as:

```
F(u,v) = c(u)c(v) Σ Σ f(x,y) · cos[(2x+1)uπ/2N] · cos[(2y+1)vπ/2N]
```
*(where `c(0) = √(1/N)` and `c(k) = √(2/N)` for k > 0, as directly implemented in `core/math.py` lines 17–23)*

**Alignment with JPEG's Perceptual Model.** The `DiffJPEG` class (`core/math.py`, lines 69–119) implements a differentiable JPEG pipeline using the *same* 8×8 block quantization framework that the `FrequencyEncoder` embeds into. The critical insight is that **JPEG discards high-frequency coefficients** according to the standard luminance quantization matrix (`q_luma`). By embedding watermark information into the **AC (alternating-current) coefficients** — which correspond to mid-to-high spatial frequencies within each block — the encoder exploits the fact that these coefficients carry image texture information that is perceptually difficult to destroy without visible artifact introduction.

The `_get_ac_mask()` method (lines 48–60) enforces this precisely:

```python
mask[:, :, 0::8, 0::8] = 0  # Zero out DC coefficient of every 8x8 block
```

This strided-zero mask implements a hard constraint: **the DC coefficient (the block's mean luminance) is never perturbed**. Only AC coefficients — which encode local contrast and texture — carry the watermark signal. This is the mathematical reason the frequency branch survives JPEG compression and Gaussian blur: these attacks operate by attenuating high-frequency content, but the encoder learns to embed bits in the coefficient bands that survive the quantization budget.

**Differentiability via Straight-Through Estimation.** The `DiffJPEG` module uses the Straight-Through Estimator (STE) for quantization (line 111):

```python
quantized = scaled + (torch.round(scaled) - scaled).detach()
```

This allows gradients to flow through the otherwise non-differentiable rounding operation during training, enabling the `FrequencyEncoder` to learn embedding strategies that are explicitly robust to JPEG quantization, since the encoder's forward pass experiences the same quantization noise the JPEG codec introduces.

### 1.2 The Spatial Branch: Identity Fragility via Pixel-Domain Sensitivity

The `SpatialEncoder` (lines 8–31) operates directly in the **pixel domain**. Its architectural purpose is the opposite of the frequency branch: to embed bits that are *destroyed* when the subject's face is synthetically manipulated (i.e., a deepfake attack).

The scientific rationale exploits the **local continuity constraint** of pixel-space CNN representations. A deepfake GAN (e.g., FaceSwap, SimSwap) performs a learned, non-linear transformation of the facial texture field:

```
I_deepfake = G_swap(I_watermarked)
```

This transformation is geometrically non-isometric — it does not preserve local pixel neighbourhoods in a one-to-one manner. Any high-frequency spatial signal embedded by `SpatialEncoder` (the 3-layer residual network at lines 15–18) will be **disrupted** at the subpixel level by even a high-quality face-swap, because the swap network's spatial warp introduces aliasing, blending seam artefacts, and illumination re-normalization. These collectively scramble the spatial embedding's bit pattern.

This is precisely verified in `trainer.py` (line 615):

```
BER (Deepfake):   XX.XX%  (Target: > 70%)
```

A BER exceeding 70% after a deepfake attack demonstrates that the spatial payload has been effectively destroyed by the manipulation — constituting a **watermark fragility trigger** that flags the image as synthetic.

### 1.3 The Fusion Block: Non-Linear Integration

The `HybridEncoder.forward()` (lines 102–110) integrates both branches through a **learned non-linear fusion**:

```python
fused_res = self.fusion(torch.cat([res_s, res_f], dim=1))
return img + (self.encoder_scale * fused_res)
```

The fusion is a 3-layer CNN (6→64→32→3 channels via 3×3, 3×3, and 1×1 convolutions) that learns to *arbitrate* between the spatial and frequency residuals. The final 1×1 projection acts as a feature-wise linear recombination, allowing the network to dynamically weight each domain's contribution per-pixel. The scalar `encoder_scale = 0.2` (YAML, line 33) is the **perceptual budget**: it controls the trade-off between embedding capacity (bit recovery) and imperceptibility (PSNR), and is the primary hyperparameter exposed for tuning invisibility without architectural changes.

---

## Defense Pillar II — Identity-Aware Gating: The Role of `r_s * M`

### 2.1 Semantic Gating via MTCNN Landmarks

The gating operation in `SpatialEncoder.forward()` (line 31):

```python
return r_s * M
```

is the single most consequential line in the entire codebase. It implements what can be formally termed a **semantically-tethered payload** — a watermark embedding whose *spatial support* is not globally distributed across the image plane but is instead hard-constrained to the binary or soft mask `M` derived from MTCNN facial landmark detection.

The `M` tensor has shape `(B, 1, H, W)` and is constructed from the CelebA landmarks file (`data/raw/celeba/list_landmarks_celeba.txt`, YAML line 11). The mask creates a Boolean indicator field:

```
M(x, y) = 1,   if (x, y) belongs to the convex hull of detected facial keypoints
M(x, y) = 0,   otherwise
```

The Hadamard product `r_s * M` performs **hard spatial zeroing**: any residual signal the `SpatialEncoder` CNN computes outside the mask is literally multiplied to zero before it is propagated to the fusion block. This is a deterministic, gradient-transparent operation — the gradient of the mask is zero everywhere the mask is zero, so the CNN learns that perturbing pixels outside the facial region **cannot reduce the loss**, and therefore allocates its representational capacity entirely within the facial geometry.

### 2.2 Why Semantic Tethering is Scientifically Superior to Global Watermarking

A global watermark (uniform embedding over the entire image) suffers from several fundamental information-theoretic deficiencies:

**1. Entropy Dilution.** In a 256×256 image, the facial region typically occupies ~15–30% of total pixels. A global watermark distributes its payload across 65,536 pixels. The bulk of those pixels (background, hair edges, clothing) are *not affected by a face swap*. A deepfake attack therefore leaves ~70–85% of the globally-embedded signal intact, which reduces the post-attack BER and makes fragility detection unreliable.

**2. Perceptual Inefficiency.** The LPIPS perceptual loss (`lambda_lpips: 100.0` in YAML) and the FaceNet Identity Loss (`lambda_id: 2.0`) both respond most acutely to perturbations within the facial region. Distributing residual signal into flat background regions wastes perceptual budget by modifying areas where humans (and the VGG feature extractor) are not paying attention.

**3. Causal Misalignment.** A deepfake is, by definition, a manipulation of the *facial identity region*. The fragility signal should be causally located in precisely the region that the attack modifies. The gating `r_s * M` enforces this causal alignment: **the bits that must break are embedded exclusively where the attack will strike**. This is the spatial analogue of the information-theoretic principle that a detector should be matched to the signal subspace.

**4. Lipschitz Consistency with the Critic.** Because the watermarked perturbation is spatially bounded to the facial region, the Wasserstein Critic (`WassersteinCritic`) receives images where pixel-level differences are concentrated and geometrically structured (face-shaped), rather than randomly distributed. This improves the Critic's ability to learn a meaningful Wasserstein distance, as the feature maps from the PatchGAN's 70×70 receptive field will more reliably contain the embedded signal within their local window.

---

## Defense Pillar III — The 3-Way Adversarial Game Theory

### 3.1 The Two Adversarial Networks: Distinct Objectives

The framework implements two distinct adversarial pressures, solving two fundamentally different game-theoretic problems:

| Property | WassersteinCritic | AdversaryNet |
|---|---|---|
| **Adversarial Objective** | Image-level realism | Bit-level destruction |
| **Input** | `I` (original) or `I_w` (watermarked) | `I_w` (watermarked) only |
| **Output** | Real-valued Wasserstein score map `(B, 1, 30, 30)` | Attacked image `(B, 3, H, W)` |
| **Game Played** | Two-player min-max on distributional distance | Two-player bit-error game |
| **Loss Type** | Wasserstein distance + Gradient Penalty | BCE on inverted bit labels |
| **Architectural Class** | PatchGAN Markovian discriminator | U-Net with skip connections |
| **Training Step** | Step 1 (`_train_discriminator`) | Step 2 (`_train_adversary`) |
| **Effect on Encoder** | Forces statistical indistinguishability of `I` and `I_w` | Forces embedding robustness to intelligent removal attacks |

**The Wasserstein Critic** is a *distributional* adversary. It asks: "Is the *distribution* of watermarked images statistically indistinguishable from the distribution of original images?" This is a generative modelling objective — the Encoder must learn to match $p_{data}$ rather than simply pass a binary test.

**The AdversaryNet** is a *content* adversary. It asks: "Can I destroy the specific bits embedded in *this specific image* with a bounded perturbation?" This is a worst-case robustness objective — the Encoder must learn to embed bits in a manner that survives the strongest possible learned image perturbation within the epsilon ball.

These two games are complementary and non-redundant: the Critic could assign a perfect realism score to `I_w` while the AdversaryNet still trivially erases the bits, or vice versa. Their *simultaneous* optimization via the `WatermarkLoss` weighted sum is what forces the Encoder to achieve both imperceptibility and adversarial resilience concurrently.

### 3.2 WGAN-GP vs. Standard BCE: Why Wasserstein Distance?

The standard GAN discriminator uses **Binary Cross-Entropy (BCE)**:

```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
```

This formulates the discriminator as a classifier and the generator's objective as **minimizing JS (Jensen-Shannon) divergence** between `p_data` and `p_generator`. The JS divergence has two critical pathological properties:

**1. Vanishing Gradients.** When `p_data` and `p_generator` have non-overlapping support (which is common in the early phase of training when the Encoder produces low-quality watermarked images), the JS divergence is a constant (log 2), independent of the distance between the distributions. This yields **zero gradient signal** to the Encoder — a phenomenon known as *mode collapse* or *gradient starvation*. In the context of this project, this would manifest as the Encoder ceasing to learn from the Critic signal and relying entirely on the fidelity losses, resulting in visible watermark artefacts.

**2. BCE Loss Saturation.** At discriminator optimality under BCE, `log(1 - D(G(z))) → 0` because `D(G(z)) → 1`. The gradient of the **generator's** BCE loss vanishes precisely when the discriminator is most informative — a paradox that makes training inherently unstable.

**The Wasserstein Distance (Earth Mover's Distance)** resolves both pathologies:

```
W(p_r, p_g) = inf_{γ ∈ Π(p_r, p_g)} E_{(x,y)~γ}[||x - y||]
```

By the **Kantorovich-Rubinstein duality**, this equals:

```
W(p_r, p_g) = sup_{||f||_L ≤ 1} E_{x~p_r}[f(x)] - E_{x~p_g}[f(x)]
```

The supremum is over all 1-Lipschitz functions `f`. The Wasserstein Critic approximates this `f`. The Critic's training objective (implemented in `_train_discriminator`, lines 371–377):

```python
loss_wasserstein = pred_fake.mean() - pred_real.mean()
```

is a direct discreté Monte Carlo approximation of the difference in expectations under the two distributions. Crucially, the gradient of this quantity with respect to `p_g` is **non-zero and continuous** even when `p_r` and `p_g` have disjoint support, providing a meaningful learning signal throughout the entire training trajectory.

**The 1-Lipschitz Constraint.** For the Critic to approximate a valid Wasserstein metric, it must be constrained to the set of 1-Lipschitz functions. The original WGAN enforced this via weight clipping, which introduces gradient flow pathologies and limits model capacity. **WGAN-GP** (Gulrajani et al., 2017) replaces this with a **gradient penalty**, directly as implemented in `_compute_gradient_penalty` (lines 271–314):

```python
# x_hat = alpha * real + (1-alpha) * fake, alpha ~ U[0,1]
interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs.detach()).requires_grad_(True)
gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
```

This penalizes any deviation of the Critic's gradient norm from exactly 1.0 at points sampled from the *straight line* between the real and fake distributions — the minimal convex constraint required to enforce the Lipschitz condition. The penalty coefficient `lambda_gp: 10.0` follows the exact recommendation from Gulrajani et al.

The generator (Encoder) loss contribution from the Critic (`WatermarkLoss.forward`, line 149):

```python
loss_disc = -disc_pred_fake.mean()
```

is the generator half of the Wasserstein game: **the Encoder maximises the Critic's score on `I_w`**, which is equivalent to minimizing the Earth Mover's Distance between `p_{I_w}` and `p_I`.

---

## Defense Pillar IV — Numerical Engineering: The V100 Optimization Stack

### 4.1 Gradient Accumulation: High-Fidelity GAN Gradients

GANs are notoriously sensitive to **effective batch size**. The gradient noise in mini-batch stochastic gradient descent is proportional to:

```
Var[∇L] ∝ 1 / N_batch
```

For GAN training specifically, the **Wasserstein distance estimate** is:

```
W_hat = (1/N) Σ C(x_i) - (1/M) Σ C(G(z_j))
```

With `N = M = batch_size = 8` (YAML line 18), the variance of this Monte Carlo estimate is high, meaning the Critic receives a noisy estimate of the true distributional divergence. This destabilizes the Critic's learning of the Lipschitz function `f`.

**Gradient Accumulation** (`accumulation_steps: 4`, YAML line 43) synthesizes an effective batch size of `8 × 4 = 32` without requiring additional GPU memory:

```python
# _step_optimizer, trainer.py lines 334-341
loss = loss / self.accumulation_steps      # Normalize the loss
scaler.scale(loss).backward()              # Accumulate gradients
if is_last:
    scaler.step(optimizer)                 # Update only every 4 steps
    scaler.update()
    optimizer.zero_grad()
```

The loss normalization `loss / accumulation_steps` ensures the **effective gradient magnitude** is equivalent to a true batch-32 forward pass: since the backward pass accumulates gradients additively across 4 micro-batches, each contributing `loss/4`, the total is equal to the single-pass gradient from a batch-32 computation. This is not merely a computational trick — it improves GAN stability by reducing the variance of the Wasserstein distance estimate by a factor of 4.

The `is_last` flag (line 490):

```python
is_last = (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader)
```

also handles the **epoch boundary** gracefully: it ensures the final partial accumulation window is always flushed, preventing gradient state from spilling across epochs.

### 4.2 FP32 BCELoss Wrapping: Numerical Stability for Bit Loss

PyTorch's Automatic Mixed Precision (AMP) computes forward and backward passes primarily in **FP16** (half precision), which offers 2× throughput on V100 Tensor Cores. However, FP16 has a dynamic range of `[6e-5, 65504]` and represents values with only **10 bits of mantissa**, versus FP32's 23 bits.

The `BCELoss` function computes:

```
L_BCE = -[y · log(p) + (1-y) · log(1-p)]
```

When `p` is very close to 0 or 1 (i.e., the encoder is confidently predicting bits), the argument of `log` approaches zero. In FP16, values below `6e-5` **underflow to zero**, making `log(p)` numerically undefined. This causes:
- Silent `NaN` propagation through the loss computation
- `Inf` values after the backward pass
- GradScaler overflow, triggering a step skip

The framework avoids this via an explicit FP32 context at every bit loss computation (e.g., `trainer.py`, lines 411–412):

```python
with torch.cuda.amp.autocast(enabled=False):
    loss_adv = self.bce(pred_bits.float(), (1.0 - msg_encoded).float())
```

The `.float()` cast ensures that regardless of the outer `autocast()` context, the BCE computation is always performed in 32-bit arithmetic, where log-domain numerical stability is guaranteed. This is also applied consistently in the fragile bit loss (trainer.py lines 132–139) and the encoder-decoder's generator step.

Crucially, the **WGAN-GP gradient penalty** is also computed entirely **without AMP** (`_train_discriminator`, note the *absence* of `with autocast()`). The `create_graph=True` parameter in `torch.autograd.grad` computes second-order gradients that are highly sensitive to numerical precision — FP16 rounding in the Hessian computation would corrupt the gradient norm estimate, yielding an invalid Lipschitz constraint.

### 4.3 GroupNorm: Enabling WGAN-GP–Compatible Normalization

The `SpatialDecoder` in `hybrid_model.py` (lines 119–130) uses `nn.GroupNorm(8, 64)` rather than `nn.BatchNorm2d`. This is not merely a preference — it is a **mathematical requirement** imposed by the gradient penalty computation.

**The Problem with BatchNorm in WGAN-GP.** The gradient penalty requires computing:

```python
gradients = torch.autograd.grad(
    outputs=critic_interpolated,
    inputs=interpolated,
    create_graph=True, ...
)[0]
```

`BatchNorm2d` computes statistics (mean, variance) across the **entire batch dimension**:

```
μ_batch = (1/N) Σ x_i,   σ_batch² = (1/N) Σ (x_i - μ_batch)²
```

When `torch.autograd.grad` computes per-sample gradients for the penalty, the BatchNorm statistics depend on **all other samples in the batch** — the per-sample gradient is not independent. This creates a **gradient entanglement**: the penalty for sample `i` leaks information from samples `j ≠ i`, causing the gradient norm estimate to be biased. Furthermore, BatchNorm's batch statistics make the Critic's function non-deterministic with respect to a single input (it depends on the batch composition), violating the requirement that the Critic be a well-defined function for the Kantorovich-Rubinstein duality to hold.

**GroupNorm** (`G=8` groups) computes normalization statistics **independently per sample**, within each group of `C/G` channels:

```
μ_gn = mean over (C/G, H, W) dimensions for each (batch, group)
```

This makes each sample's forward pass **batch-independent**, which is a prerequisite for valid per-sample gradient computation in the WGAN-GP penalty. The gradient of `C(x_hat_i)` with respect to `x_hat_i` is now a pure function of `x_hat_i` alone, making the Lipschitz constraint correctly enforced.

> [!NOTE]
> The `WassersteinCritic` itself uses `InstanceNorm2d` (lines 45, 52, 59) for the same mathematical reason — InstanceNorm is GroupNorm with `G = C` (one group per channel), providing the strongest form of batch-independence. The Decoder's GroupNorm choice (`G=8`) balances this mathematical requirement against the statistical stability benefits of normalizing larger feature groups.

---

## Defense Pillar V — The Curriculum Learning Strategy: The 3-Phase Convergence Protocol

### 5.1 The Mathematical Collapse Problem

To understand why the 3-phase curriculum is necessary, one must first understand the failure mode it prevents: **Mathematical Collapse of the Encoder**.

The Encoder's total training objective (from `WatermarkLoss.forward`) is:

```
L_total = λ_L1·L_L1 + λ_LPIPS·L_LPIPS + λ_SSIM·L_SSIM + λ_ID·L_ID
        + λ_bit·L_BCE + λ_disc·L_WGAN + λ_adv·L_adv_bit
```

If **all loss terms are activated from epoch 1**, the gradient landscape the Encoder must navigate has **competing and contradictory attraction basins**:

- `L_WGAN` pulls the Encoder output toward the manifold of natural face images
- `L_BCE_bit` pulls the Encoder to produce high-energy embeddings (strong bits)
- `L_L1 + L_LPIPS` pull the Encoder output toward the original image (zero perturbation)

The Wasserstein Critic (which is untrained at epoch 1) initially assigns **random scores**, providing a nonsensical gradient signal that simultaneously conflicts with the fidelity losses. The resulting gradient `∇_θ L_total` points in a high-variance, near-random direction in the Encoder's parameter space.

In this regime, the most energetically favorable solution for the Encoder is the **trivial fixed point**: `I_w = I` (zero residual, identity mapping). This satisfies `L_L1 = 0` and `L_LPIPS = 0` but completely destroys `L_BCE_bit`. The Encoder has "collapsed" — it has found a stable minimum that is mathematically valid but scientifically useless.

### 5.2 Phase 1 (Epochs 1–2): Establishing the Canvas

```python
trainer.criterion.W_BIT_BENIGN  = 0.0
trainer.criterion.W_BIT_FRAGILE = 0.0
trainer.criterion.W_DISC        = 0.0
trainer.criterion.W_ADV         = 0.0
```

In Phase 1, all bit and adversarial loss terms are **zeroed out**. The Encoder trains exclusively on the fidelity objective:

```
L_phase1 = λ_L1·L_L1 + λ_LPIPS·L_LPIPS + λ_SSIM·L_SSIM + λ_ID·L_ID
```

With `λ_L1 = 150.0`, `λ_LPIPS = 100.0`, `λ_SSIM = 50.0` (YAML lines 49–51), the effective loss landscape is **unimodal**: the global minimum is the identity mapping (`I_w = I`), and the gradient field is smooth, convex-like, and non-conflicting.

This phase achieves a critical prerequisite: the Encoder learns the **geometry of the perceptual manifold** — the regions of parameter space where the network's output is simultaneously close in L1, perceptual (VGG), structural (SSIM), and identity (FaceNet) distance to the original. The network establishes what can be called a **canonical representation baseline** or, in JPEG parlance, a zero-perturbation "canvas."

By constraining the Encoder to first learn the identity function with high precision, Phase 1 ensures that subsequent phases begin from a well-defined initialization point near the origin of the perturbation space, rather than from random noise.

### 5.3 Phase 2 (Epochs 3–7): Bit Channel Carving

```python
trainer.criterion.W_BIT_BENIGN  = 40.0
trainer.criterion.W_BIT_FRAGILE = 0.0
trainer.criterion.W_DISC        = 0.0
trainer.criterion.W_ADV         = 0.0
```

Phase 2 activates only the **benign bit loss** at a weight of `40.0`. The Encoder now has a precise mission: starting from the near-identity initialization established in Phase 1, it must learn to embed bits in a manner that survives benign augmentations (JPEG compression, Gaussian blur, etc.) while maintaining fidelity.

This phase can be understood as **channel carving**: the Encoder discovers the specific dimensions of the DCT coefficient space (via the FrequencyEncoder) and the facial texture space (via the SpatialEncoder, gated by `M`) that are:
1. Sufficiently orthogonal to the fidelity loss gradients that they can encode information without degrading PSNR
2. Sufficiently robust to survive the BenignAugmentationPipeline

Note that `W_BIT_FRAGILE = 0.0` in Phase 2. This is deliberate: the fragile bit objective (maximizing BER *after* deepfake) directly opposes the robust bit objective (minimizing BER after benign augmentation). Activating both simultaneously in the early learning phase would produce conflicting gradient signals and prevent the network from establishing a stable bit channel. Phase 2 solves the robustness problem first.

### 5.4 Phase 3 (Epoch 8+): The Arms Race — Full Adversarial Learning

```python
trainer.criterion.W_BIT_BENIGN  = config['loss_config']['lambda_bit_benign']   # 25.0
trainer.criterion.W_BIT_FRAGILE = config['loss_config']['lambda_bit_fragile']  # 10.0
trainer.criterion.W_DISC        = config['loss_config']['lambda_disc']          # 5.0
trainer.criterion.W_ADV         = config['loss_config']['lambda_adv']           # 0.1
```

Phase 3 activates all loss components. At this stage:

- The **Encoder** has a well-initialized, identity-close, bit-capable parameters from Phases 1 and 2
- The **Discriminator** begins receiving a non-trivial distribution of watermarked images (not random noise) from which it can begin learning a meaningful Wasserstein distance
- The **AdversaryNet** similarly receives plausible watermarked images, allowing the learned U-Net attack to be practically relevant rather than noise-targeting

The formal description of Phase 3 is a **3-player simultaneous game** with the following Stackelberg structure per training step:

1. **(Step 1) Critic Leadership:** Given current Encoder weights, the Critic updates toward the optimal Wasserstein discriminator, receiving its own dedicated optimizer with `betas=(0.0, 0.9)` and learning rate `lr * disc_lr_mult = 5e-5 * 0.5 = 2.5e-5`
2. **(Step 2) Adversary Leadership:** Given current Encoder/Decoder weights, the AdversaryNet updates toward maximizing BER at the epsilon-bounded perturbation limit
3. **(Step 3) Encoder-Decoder Response:** Given updated Critic and Adversary, the Encoder-Decoder updates to minimize the full 8-component loss

This sequential update rule — Critic, then Adversary, then Generator — ensures that each player responds to the **most current** version of its adversaries, rather than to stale gradient estimates from a previous iteration. It is the discrete-time implementation of the continuous Nash equilibrium update dynamics.

### 5.5 The QoS Score: A Scientifically Principled Checkpointing Metric

The final checkpointing decision (train.py, line 138):

```python
current_score = (1.0 - ber_benign) * (psnr / 40.0)
```

implements a **multiplicative Quality of Service objective**. This is superior to a simple additive combination (`0.5 * accuracy + 0.5 * psnr`) for two information-theoretic reasons:

1. **Multiplicative form enforces joint optimality.** If BER is high (`ber_benign ≈ 0.5`, i.e., chance-level), then `(1 - ber_benign) ≈ 0.5`, and the score is penalized regardless of PSNR. Conversely, a model achieving `PSNR = 40dB` but BER = 50% would score `0.5 × 1.0 = 0.5` — no better than a model with `PSNR = 20dB` and BER = 0%. An additive form would rank the high-PSNR model better.

2. **The 40dB normalization anchor.** `psnr / 40.0` normalizes PSNR to a [0, 1] scale where 1.0 represents the research community's accepted threshold for "imperceptible" watermarking. A score above 1.0 is achievable only if PSNR > 40dB, which is the target referenced in the evaluation report (`eval_epoch`, line 617).

---

## Summary Table: Defense Pillars at a Glance

| Pillar | Key Mechanism | Code Location | Design Rationale |
|---|---|---|---|
| **Domain Split** | DCT AC-only embedding + spatial residual | `hybrid_model.py:48–81` | Frequency branch survives JPEG; spatial branch is fragile to face swaps |
| **Identity Gating** | `r_s * M` Hadamard product | `hybrid_model.py:31` | Semantically-tethered payload; causal alignment with deepfake attack region |
| **Critic vs. Adversary** | WGAN-GP score vs. U-Net perturbation | `adversarial.py:18–177` | Distributional realism vs. content-bit destruction — orthogonal objectives |
| **WGAN-GP** | Earth Mover's Distance + gradient penalty | `trainer.py:271–385` | Non-vanishing gradients; valid Wasserstein metric via Lipschitz constraint |
| **Gradient Accumulation** | `loss / 4` + `is_last` step control | `trainer.py:332–341` | Synthesizes batch-32 for lower-variance Wasserstein estimate on batch-8 GPU |
| **FP32 BCE Wrapping** | `autocast(enabled=False)` around BCELoss | `trainer.py:411–412` | Prevents FP16 log-underflow NaN at confident bit predictions |
| **GroupNorm** | `GroupNorm(8, C)` instead of `BatchNorm` | `hybrid_model.py:121–128` | Batch-independent statistics preserve per-sample gradient validity for WGAN-GP |
| **Curriculum (Ph1)** | Zero bit/adv weights for 2 epochs | `train.py:105–109` | Establishes identity-mapping canvas; prevents mathematical encoder collapse |
| **Curriculum (Ph2)** | Benign bit loss only, weight=40 | `train.py:112–116` | Carves robust bit channel before introducing adversarial noise |
| **Curriculum (Ph3)** | Full 8-component loss, all weights restored | `train.py:119–123` | Arms race begins from stable initialization; Critic and Adversary are meaningful |
