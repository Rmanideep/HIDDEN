# -*- coding: utf-8 -*-
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import lpips
from facenet_pytorch import InceptionResnetV1
import kornia.metrics as K_metrics
import kornia.losses as K_losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.hybrid_model import HybridEncoder, HybridDecoder
from src.models.adversarial import WassersteinCritic, AdversaryNet
from src.engine.math import encode_message_bch, decode_message_bch, FunctionalBlockCodec
from src.attacks.attacks import BenignAugmentationPipeline, MalignAttackGenerator


class WatermarkLoss(nn.Module):
    """
    Research-Grade 8-Component Weighted Objective Function.
    
    Balances eight competing objectives across four categories:
    
    FIDELITY (Image Quality):
        1. L1 Loss           - Pixel-level fidelity between I and I'
        2. SSIM Loss          - Structural integrity via luminance/contrast/structure
        3. LPIPS Loss         - Deep perceptual similarity (VGG-based)
        4. Identity Loss      - FaceNet embedding preservation (VGGFace2)
    
    SECURITY (Bit Recovery):
        5. Benign Bit Loss    - BER after benign augmentations (robustness)
        6. Fragile Bit Loss   - BER after deepfake attacks (fragility trigger)
    
    ADVERSARIAL (GAN Realism):
        7. Discriminator Loss - PatchGAN adversarial realism (fool the critic)
    
    RESILIENCE (Attack Resistance):
        8. Adversary Loss     - BER after learned AdversaryNet removal attack
    
    Weighting:
        Total = L1 + LPIPS + SSIM + ID + Bits + Disc + Adv
    """

    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config['training_config']['device'])

        # --- Loss Weights from Config (no hardcoding) ---
        lc = config['loss_config']
        self.W_L1          = lc['lambda_l1']
        self.W_LPIPS       = lc['lambda_lpips']
        self.W_SSIM        = lc['lambda_ssim']
        self.W_ID          = lc['lambda_id']
        self.W_BIT_BENIGN  = lc['lambda_bit_benign']
        self.W_BIT_FRAGILE = lc['lambda_bit_fragile']
        self.W_DISC        = lc['lambda_disc']
        self.W_ADV         = lc['lambda_adv']

        # --- Pixel & Structural Losses ---
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = K_losses.SSIMLoss(window_size=11, reduction='mean')

        # --- Perceptual Loss ---
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device).eval()

        # --- Identity Consistency Loss ---
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        for param in self.facenet.parameters():
            param.requires_grad = False

        # --- Bit-Level Losses ---
        self.bce = nn.BCELoss()

        # --- Discriminator Loss (WGAN-GP style) ---
        # Wasserstein: L = E[C(fake)] - E[C(real)]
        # Generator objective: maximize E[C(fake)] = minimize -E[C(fake)]
        # Note: NO BCEWithLogitsLoss here. Critic outputs raw scores.
        # BCE is removed -- Wasserstein distance is used instead.
        self.wgan_lambda_gp = config['loss_config'].get('lambda_gp', 10.0)

    def get_identity_mse(self, img1, img2):
        """Computes MSE between FaceNet (VGGFace2) embeddings."""
        resize = lambda x: F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        normalize = lambda x: (resize(x) * 2.0) - 1.0
        feat1 = self.facenet(normalize(img1))
        feat2 = self.facenet(normalize(img2))
        return F.mse_loss(feat1, feat2)

    def forward(self, I, I_w, msg_encoded,
                msg_pred_benign, msg_pred_attacked=None,
                disc_pred_fake=None, msg_pred_adv=None):
        """
        Computes the full 8-component loss.

        Args:
            I:          (B, 3, H, W) Original host image I.
            I_w:             (B, 3, H, W) Watermarked image I'.
            msg_encoded:       (B, N) Ground-truth encoded message bits.
            msg_pred_benign:   (B, N) Decoded bits after benign augmentations.
            msg_pred_attacked: (B, N) Decoded bits after deepfake/malign attack.
            disc_pred_fake:    (B, 1, H', W') Discriminator logits on I' (optional).
            msg_pred_adv:      (B, N) Decoded bits after AdversaryNet attack (optional).

        Returns:
            total_loss: Scalar weighted sum of all active loss components.
            metrics:    Dict of individual loss values for TensorBoard logging.
        """
        metrics = {}

        # ===================== FIDELITY =====================

        # 1. L1 Loss - Pixel-level absolute difference
        loss_l1 = self.l1_loss(I_w, I)
        metrics['l1'] = loss_l1.item()

        # 2. SSIM Loss - Structural similarity (luminance, contrast, structure)
        loss_ssim = self.ssim_loss(I_w, I)
        metrics['ssim'] = loss_ssim.item()

        # 3. LPIPS Loss - Deep perceptual similarity (VGG feature distance)
        loss_lpips = self.lpips_vgg((I * 2.0) - 1.0, (I_w * 2.0) - 1.0).mean()
        metrics['lpips'] = loss_lpips.item()

        # 4. Identity Loss - FaceNet embedding distance
        loss_id = self.get_identity_mse(I, I_w)
        metrics['id'] = loss_id.item()

        # ===================== SECURITY =====================
        # Use autocast(enabled=False) for BCE to prevent precision issues
        with torch.cuda.amp.autocast(enabled=False):
            # 5. Benign Bit Loss - Recover bits after benign edits (robustness)
            loss_bit_benign = self.bce(msg_pred_benign.float(), msg_encoded.float())
            metrics['bit_benign'] = loss_bit_benign.item()

            # 6. Fragile Bit Loss - Force bit destruction after deepfake attack
            loss_bit_fragile = torch.tensor(0.0, device=self.device)
            if msg_pred_attacked is not None:
                loss_bit_fragile = self.bce(msg_pred_attacked.float(), (1.0 - msg_encoded).float())
            metrics['bit_fragile'] = loss_bit_fragile.item()

        # ===================== ADVERSARIAL =====================

        # 7. Critic (Wasserstein) Loss -- Generator objective
        # Goal: Fool the Critic. Wasserstein generator loss = -E[C(I_w)]
        # (Encoder wants Critic to output HIGH scores for watermarked images)
        loss_disc = torch.tensor(0.0, device=self.device)
        if disc_pred_fake is not None:
            # Generator tries to MAXIMIZE Critic score on fake -- so minimize negative mean
            loss_disc = -disc_pred_fake.mean()
        metrics['disc'] = loss_disc.item()

        # ===================== RESILIENCE =====================

        # 8. Adversary Loss - Recover bits even after learned AdversaryNet attack
        loss_adv = torch.tensor(0.0, device=self.device)
        if msg_pred_adv is not None:
            with torch.cuda.amp.autocast(enabled=False):
                loss_adv = self.bce(msg_pred_adv.float(), msg_encoded.float())
        metrics['adv'] = loss_adv.item()

        # ===================== WEIGHTED SUM =====================
        total_loss = (
            self.W_L1          * loss_l1 +
            self.W_LPIPS       * loss_lpips +
            self.W_SSIM        * loss_ssim +
            self.W_ID          * loss_id +
            self.W_BIT_BENIGN  * loss_bit_benign +
            self.W_BIT_FRAGILE * loss_bit_fragile +
            self.W_DISC        * loss_disc +
            self.W_ADV         * loss_adv
        )
        metrics['total'] = total_loss.item()

        return total_loss, metrics


class WatermarkTrainer:
    """
    3-Way Adversarial Training Framework.

    Implements a complex min-max game between three competing networks:

    1. Encoder-Decoder (Generator): Minimizes the 8-component Total_Loss.
       Learns to hide bits invisibly while resisting both enemies.

    2. Discriminator (PatchGAN): Maximizes detection of watermarked vs. original.
       Pushes the Encoder toward statistical indistinguishability.

    3. Adversary (Intruder): Maximizes Bit Error Rate after attacking I'.
       Forces the Encoder to embed bits that survive intelligent removal.

    Training Order (each step):
        Step 1: Train Discriminator   - Classify Real(I) vs Fake(I')
        Step 2: Train Adversary       - Attack I' to destroy bits
        Step 3: Train Encoder-Decoder - Resist both enemies + maintain quality
    """

    def __init__(self, config, encoder, decoder, discriminator, adversary,
                 train_loader, val_loader=None):
        self.config = config
        self.device = torch.device(config['training_config']['device'])

        # --- Core Networks ---
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.adversary = adversary.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # --- Attack Pipelines ---
        self.benign_pipeline = BenignAugmentationPipeline(device=self.device)
        self.malign_attack = MalignAttackGenerator(device=self.device)

        # Read from config - no hardcoding
        self.n_bits = config['model_config']['n_bits']
        self.log_freq = config['training_config']['log_freq']

        # =====================================================================
        # BCH CODEC INITIALIZATION
        # =====================================================================
        k = config['model_config']['watermark_length']
        self.codec = FunctionalBlockCodec(k=k, n=self.n_bits, device=self.device)

        # =====================================================================
        # THREE SEPARATE OPTIMIZERS - One per competing network
        # =====================================================================
        lr = config['training_config']['lr']
        betas = tuple(config['training_config']['betas'])
        disc_lr_mult = config['training_config']['disc_lr_mult']

        # Optimizer 1: Encoder-Decoder (the "Generator" in GAN terms)
        self.opt_encdec = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr, betas=betas
        )

        # Optimizer 2: Wasserstein Critic
        self.opt_disc = optim.Adam(
            self.discriminator.parameters(),
            lr=lr * disc_lr_mult, betas=(0.0, 0.9)  # WGAN-GP uses betas=(0, 0.9)
        )

        # Optimizer 3: Adversary
        self.opt_adv = optim.Adam(
            self.adversary.parameters(),
            lr=lr, betas=betas
        )

        # --- LR Scheduling (Plateau-based, not timer-based) ---
        # ReduceLROnPlateau only decays LR when BER stops improving,
        # unlike CosineAnnealingLR which was decaying during Phase 1 and
        # actively fighting the curriculum.
        self.scheduler_encdec = ReduceLROnPlateau(
            self.opt_encdec, mode='min', factor=0.3, patience=12, min_lr=1e-6, verbose=True
        )
        self.scheduler_disc = ReduceLROnPlateau(
            self.opt_disc, mode='min', factor=0.3, patience=12, min_lr=1e-6, verbose=True
        )
        self.scheduler_adv = ReduceLROnPlateau(
            self.opt_adv, mode='min', factor=0.3, patience=12, min_lr=1e-6, verbose=True
        )

        # Read lambda_gp from config
        self.lambda_gp = config['loss_config'].get('lambda_gp', 10.0)

        # =====================================================================
        # MIXED PRECISION - Separate GradScalers per optimizer for A100
        # =====================================================================
        self.scaler_encdec = GradScaler()
        self.scaler_disc = GradScaler()
        self.scaler_adv = GradScaler()

        # --- Loss & Logging ---
        self.criterion = WatermarkLoss(config=config)
        self.disc_loss_fn = nn.BCEWithLogitsLoss()  # Kept only for adversary BCE steps
        self.bce = nn.BCELoss()
        self.writer = SummaryWriter(config['path_config']['log_dir'])
        self.checkpoint_dir = config['path_config']['checkpoint_dir']

        # --- Refactor: Gradient Accumulation ---
        tc = config['training_config']
        self.accumulation_steps = tc.get('accumulation_steps', 1)
        self.eval_freq = tc.get('eval_freq', 1)
        self.max_eval_steps = tc.get('max_eval_steps', None)

        # --- Augmentation Probability (set by train.py curriculum) ---
        self.aug_prob = 0.0  # 0.0 = no augmentations, 1.0 = always augment

    # =====================================================================
    # WGAN-GP: GRADIENT PENALTY COMPUTATION
    # =====================================================================
    def _compute_gradient_penalty(self, real_imgs, fake_imgs):
        """
        WGAN-GP Gradient Penalty (Gulrajani et al., 2017).

        Enforces the Lipschitz constraint on the Wasserstein Critic by penalizing
        the gradient norm at random interpolations between real and fake images.

        Mathematically:
            x_hat = epsilon * real + (1 - epsilon) * fake,  epsilon ~ Uniform(0,1)
            GP = E[(||grad_C(x_hat)||_2 - 1)^2]

        This replaces weight clipping (original WGAN) and is more stable.
        Gradient penalty ensures the Critic remains 1-Lipschitz continuous,
        which is required for the Wasserstein distance to be a valid metric.

        Returns:
            gradient_penalty: scalar tensor, added to Critic loss.
        """
        B = real_imgs.size(0)
        # Random interpolation coefficient alpha ~ Uniform[0,1] for each sample
        alpha = torch.rand(B, 1, 1, 1, device=self.device)
        # x_hat: random point on the line between real and fake
        interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs.detach()).requires_grad_(True)

        # Get Critic score at interpolated point
        critic_interpolated = self.discriminator(interpolated)

        # Compute gradient of Critic output w.r.t. interpolated input
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,   # Need second-order gradients for autograd
            retain_graph=True,
            only_inputs=True
        )[0]

        # Reshape to (B, -1) and compute L2 norm per sample
        gradients = gradients.view(B, -1)
        gradient_norm = gradients.norm(2, dim=1)   # ||grad||_2

        # Penalize deviation from 1-Lipschitz (norm should be exactly 1.0)
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
        return gradient_penalty

    # =====================================================================
    # UTILITY: Freeze / Unfreeze parameter groups
    # =====================================================================
    @staticmethod
    def _freeze(model):
        for p in model.parameters():
            p.requires_grad = False

    @staticmethod
    def _unfreeze(model):
        for p in model.parameters():
            p.requires_grad = True

    # =====================================================================
    # UTILITY: Step Optimizer with Accumulation
    # =====================================================================
    def _step_optimizer(self, optimizer, scaler, loss, is_last, params=None):
        """Helper to handle gradient scaling, accumulation, and stepping."""
        # Normalize loss for accumulation
        loss = loss / self.accumulation_steps
        scaler.scale(loss).backward()
        
        if is_last:
            if params is not None:
                # Unscale prior to clipping to ensure true gradient magnitudes are used
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # =====================================================================
    # STEP 1: TRAIN DISCRIMINATOR
    # =====================================================================
    def _train_discriminator(self, img_orig, I_w, is_last=True):
        """
        Wasserstein Critic objective:
            L_critic = E[C(I_w)] - E[C(I)] + lambda_gp * GP

        The Critic wants to assign LOW scores to watermarked images (fake)
        and HIGH scores to original images (real). The gradient penalty
        enforces the Lipschitz constraint required for Wasserstein distance.

        Note: Autocast is DISABLED for Critic training. WGAN-GP gradient
        penalty requires precise FP32 gradient norms. Mixed precision can
        corrupt the gradient norm computation inside the penalty calculation.
        """
        self._unfreeze(self.discriminator)
        self._freeze(self.encoder)
        self._freeze(self.decoder)
        self._freeze(self.adversary)

        self.opt_disc.zero_grad()

        # FP32 required for accurate gradient penalty computation
        # autocast is intentionally NOT used here
        pred_real = self.discriminator(img_orig.detach())
        pred_fake = self.discriminator(I_w.detach())

        # Wasserstein Critic Loss: maximize E[C(real)] - E[C(fake)]
        # i.e. minimize E[C(fake)] - E[C(real)]
        loss_wasserstein = pred_fake.mean() - pred_real.mean()

        # Gradient Penalty (enforces 1-Lipschitz continuity)
        gp = self._compute_gradient_penalty(img_orig, I_w)
        loss_critic = loss_wasserstein + self.lambda_gp * gp

        # NOTE: No GradScaler here -- gradient penalty needs raw FP32 gradients
        loss_critic.backward()
        if is_last:
            self.opt_disc.step()
            self.opt_disc.zero_grad()

        return loss_critic.item(), loss_wasserstein.item(), gp.item()

    # =====================================================================
    # STEP 2: TRAIN ADVERSARY (INTRUDER)
    # =====================================================================
    def _train_adversary(self, I_w, msg_encoded, is_last=True):
        """
        Adversary objective: Maximize BER after attacking I'.
        The adversary learns to perturb I' to destroy the embedded bits.
        """
        self._unfreeze(self.adversary)
        self._freeze(self.encoder)
        self._freeze(self.decoder)
        self._freeze(self.discriminator)

        self.opt_adv.zero_grad()

        with autocast():
            # Attack the watermarked image
            img_attacked = self.adversary(I_w.detach())

            # Decode bits from the attacked image
            pred_bits = self.decoder(img_attacked)

            # Adversary wants the decoded bits to be the INVERSE of the truth
            # This maximizes the Bit Error Rate
            with torch.cuda.amp.autocast(enabled=False):
                loss_adv = self.bce(pred_bits.float(), (1.0 - msg_encoded).float())

        self._step_optimizer(self.opt_adv, self.scaler_adv, loss_adv, is_last, params=self.adversary.parameters())

        return loss_adv.item()

    # =====================================================================
    # STEP 3: TRAIN ENCODER-DECODER (GENERATOR)
    # =====================================================================
    def _train_encoder_decoder(self, I, M, msg_encoded,
                                I_donor, is_last=True, epoch=1):
        """
        Generator objective: Minimize the full 8-component loss.
        The encoder-decoder must simultaneously:
          - Keep the image invisible (L1, SSIM, LPIPS, Identity)
          - Recover bits after benign edits (Benign BER)
          - Break bits after deepfake attacks (Fragile BER)
          - Fool the discriminator (Disc loss)
          - Resist the adversary (Adv BER)
        """
        self._unfreeze(self.encoder)
        self._unfreeze(self.decoder)
        self._freeze(self.discriminator)
        self._freeze(self.adversary)

        self.opt_encdec.zero_grad()

        with autocast():
            # --- Forward: Encode ---
            I_w = self.encoder(I, M, msg_encoded)

            # --- Benign Path (Robustness) ---
            # Augmentation probability controlled by curriculum (train.py sets self.aug_prob)
            if self.aug_prob > 0 and random.random() < self.aug_prob:
                benign_w = self.benign_pipeline(I_w)
                pred_benign = self.decoder(benign_w)
            else:
                pred_benign = self.decoder(I_w)

            # --- Malign Path (Fragility) --- GATED: skip when weight is 0
            pred_attacked = None
            if self.criterion.W_BIT_FRAGILE > 0:
                attacked_w = self.malign_attack(I_w, I_donor, M)
                pred_attacked = self.decoder(attacked_w)

            # --- Discriminator Signal (Realism) --- GATED: skip when weight is 0
            disc_pred_fake = None
            if self.criterion.W_DISC > 0:
                disc_pred_fake = self.discriminator(I_w)

            # --- Adversary Signal (Resilience) --- GATED: skip when weight is 0
            pred_adv = None
            if self.criterion.W_ADV > 0:
                img_adv = self.adversary(I_w)
                pred_adv = self.decoder(img_adv)

            # --- Compute Full 8-Component Loss ---
            total_loss, metrics = self.criterion(
                I, I_w, msg_encoded,
                pred_benign, pred_attacked,
                disc_pred_fake, pred_adv
            )

        # --- Decoder Output Diagnostics ---
        with torch.no_grad():
            metrics['dec_mean'] = pred_benign.mean().item()
            metrics['dec_std'] = pred_benign.std().item()

        params_encdec = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self._step_optimizer(self.opt_encdec, self.scaler_encdec, total_loss, is_last, params=params_encdec)

        return total_loss.item(), metrics, I_w

    # =====================================================================
    # MAIN TRAINING LOOP
    # =====================================================================
    def train_epoch(self, epoch):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        self.adversary.train()

        running_loss = 0.0
        
        # Zero gradients at start of epoch
        self.opt_encdec.zero_grad()
        self.opt_disc.zero_grad()
        self.opt_adv.zero_grad()

        for i, (I, M, m_gt, I_donor) in enumerate(self.train_loader):
            I, M, m_gt, I_donor = I.to(self.device), M.to(self.device), m_gt.to(self.device), I_donor.to(self.device)
            msg_encoded = encode_message_bch(m_gt, self.codec)

            # Multi-batch Gradient Accumulation Logic
            is_last = (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader)

            # ============================================
            # STEP 1: Train Wasserstein Critic -- GATED
            # ============================================
            loss_d, loss_w, loss_gp = 0.0, 0.0, 0.0
            if self.criterion.W_DISC > 0:
                with torch.no_grad():
                    img_w_for_disc = self.encoder(I, M, msg_encoded)
                loss_d, loss_w, loss_gp = self._train_discriminator(I, img_w_for_disc, is_last=is_last)

            # ============================================
            # STEP 2: Train Adversary -- GATED
            # ============================================
            loss_a = 0.0
            if self.criterion.W_ADV > 0:
                with autocast():
                    img_w_for_adv = self.encoder(I, M, msg_encoded)
                loss_a = self._train_adversary(img_w_for_adv, msg_encoded, is_last=is_last)

            # ============================================
            # STEP 3: Train Encoder-Decoder (Generator)
            # ============================================
            loss_g, metrics, I_w = self._train_encoder_decoder(
                I, M, msg_encoded, I_donor, is_last=is_last, epoch=epoch
            )

            running_loss += loss_g

            # --- Logging ---
            global_step = epoch * len(self.train_loader) + i
            if i % self.log_freq == 0:
                print(
                    f"Epoch {epoch}, Step {i:>4d} | "
                    f"G: {metrics['total']:.4f} | "
                    f"W_dist: {loss_w:.4f} | "
                    f"GP: {loss_gp:.4f} | "
                    f"A: {loss_a:.4f} | "
                    f"L1: {metrics['l1']:.4f} | "
                    f"LPIPS: {metrics['lpips']:.4f} | "
                    f"BER_b: {metrics['bit_benign']:.4f} | "
                    f"BER_adv: {metrics['adv']:.4f} | "
                    f"Dec: mu={metrics.get('dec_mean', 0):.3f} sig={metrics.get('dec_std', 0):.3f}"
                )
                # TensorBoard -- float() safely converts to Python scalar
                self.writer.add_scalar('Train/Generator_Total', float(metrics['total']), global_step)
                self.writer.add_scalar('Train/Wasserstein_Distance', float(loss_w), global_step)
                self.writer.add_scalar('Train/Gradient_Penalty', float(loss_gp), global_step)
                self.writer.add_scalar('Train/Critic_Total', float(loss_d), global_step)
                self.writer.add_scalar('Train/Adversary', float(loss_a), global_step)
                for key, val in metrics.items():
                    self.writer.add_scalar(f'Train/Loss_{key}', float(val), global_step)

        # --- LR scheduling is driven by val BER, called from train.py eval block ---
        # (ReduceLROnPlateau.step() requires the monitored metric — called in train.py)

    # =====================================================================
    # EVALUATION LOOP
    # =====================================================================
    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.encoder.eval()
        self.decoder.eval()
        self.adversary.eval()

        total_ber_benign = 0.0
        total_ber_attacked = 0.0
        total_ber_adv = 0.0
        total_raw_ber = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_id = 0.0
        batches = 0
        viz_saved = False

        for i, (I, M, m_gt, I_donor) in enumerate(self.val_loader):
            if self.max_eval_steps is not None and i >= self.max_eval_steps:
                break
            I = I.to(self.device)
            M = M.to(self.device)
            m_gt = m_gt.to(self.device)
            I_donor = I_donor.to(self.device)
            msg_encoded = encode_message_bch(m_gt, self.codec)

            with autocast():
                I_w = self.encoder(I, M, msg_encoded)
                # Match training condition: only augment if aug_prob > 0
                if self.aug_prob > 0:
                    benign_w = self.benign_pipeline(I_w)
                else:
                    benign_w = I_w  # Clean eval during Phase 1 (matches training)

                pred_benign = self.decoder(benign_w)

                # Gate malign path -- only run if fragile loss is active
                pred_attacked = None
                if self.criterion.W_BIT_FRAGILE > 0:
                    attacked_w = self.malign_attack(I_w, I_donor, M)
                    pred_attacked = self.decoder(attacked_w)
                else:
                    # Placeholder: random chance (untrained path)
                    pred_attacked = torch.empty_like(pred_benign).fill_(0.5)

                # Gate adversary path -- only run if adv loss is active
                pred_adv = None
                if self.criterion.W_ADV > 0:
                    img_adv = self.adversary(I_w)
                    pred_adv = self.decoder(img_adv)
                else:
                    # Placeholder: random chance (untrained path)
                    pred_adv = torch.empty_like(pred_benign).fill_(0.5)

                # Perceptual metrics (always computed for logging)
                loss_lpips = self.criterion.lpips_vgg((I * 2.0) - 1.0, (I_w * 2.0) - 1.0).mean()
                loss_id = self.criterion.get_identity_mse(I, I_w)

            # BER Calculations
            raw_ber_b = ((pred_benign > 0.5).float() != msg_encoded).float().mean()
            
            bch_pred_b = decode_message_bch((pred_benign > 0.5).float(), self.codec)
            bch_pred_a = decode_message_bch((pred_attacked > 0.5).float(), self.codec)
            bch_pred_adv = decode_message_bch((pred_adv > 0.5).float(), self.codec)

            ber_benign = (bch_pred_b != m_gt).float().mean()
            ber_attacked = (bch_pred_a != m_gt).float().mean()
            ber_adv = (bch_pred_adv != m_gt).float().mean()

            # PSNR & SSIM
            psnr = K_metrics.psnr(I_w.float(), I.float(), max_val=1.0).mean()
            ssim = K_metrics.ssim(I_w.float(), I.float(), window_size=11, max_val=1.0).mean()

            total_raw_ber += raw_ber_b.item()
            total_ber_benign += ber_benign.item()
            total_ber_attacked += ber_attacked.item()
            total_ber_adv += ber_adv.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            total_lpips += loss_lpips.item()
            total_id += loss_id.item()
            batches += 1

            if not viz_saved:
                self.visualize_batch(I, M, I_w, epoch)
                viz_saved = True

        # Averages
        avg_ber_b = total_ber_benign / batches
        avg_raw_ber = total_raw_ber / batches
        avg_ber_a = total_ber_attacked / batches
        avg_ber_adv = total_ber_adv / batches
        avg_psnr = total_psnr / batches
        avg_ssim = total_ssim / batches
        avg_lpips = total_lpips / batches
        avg_id = total_id / batches

        print(f"\n{'='*55}")
        print(f"  EPOCH {epoch} EVALUATION REPORT")
        print(f"{'='*55}")
        print(f"  RAW Neural BER:   {avg_raw_ber*100:>6.2f}%")
        print(f"  BER (BCH Benign): {avg_ber_b*100:>6.2f}%  (Target: < 2%)")
        print(f"  BER (Deepfake):   {avg_ber_a*100:>6.2f}%  (Target: > 70%)")
        print(f"  BER (Adversary):  {avg_ber_adv*100:>6.2f}%  (Target: < 5%)")
        print(f"  PSNR:             {avg_psnr:>6.2f} dB (Target: > 35dB)")
        print(f"  SSIM:             {avg_ssim:>6.4f}   (Target: > 0.95)")
        print(f"  LPIPS:            {avg_lpips:>6.4f}   (Target: < 0.1)")
        print(f"  ID_Loss:          {avg_id:>6.6f} (Target: < 0.01)")
        print(f"{'='*55}\n")

        # TensorBoard -- float() for scalars
        self.writer.add_scalar('Val/BER_Benign',    float(avg_ber_b),   epoch)
        self.writer.add_scalar('Val/BER_Deepfake',  float(avg_ber_a),   epoch)
        self.writer.add_scalar('Val/BER_Adversary', float(avg_ber_adv), epoch)
        self.writer.add_scalar('Val/PSNR',          float(avg_psnr),    epoch)
        self.writer.add_scalar('Val/SSIM',          float(avg_ssim),    epoch)
        self.writer.add_scalar('Val/LPIPS',         float(avg_lpips),   epoch)
        self.writer.add_scalar('Val/Identity',      float(avg_id),      epoch)

        # Free memory at the end of the evaluation epoch to prevent fragmentation on V100
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_ber_b, avg_ber_a, avg_ber_adv, avg_psnr, avg_ssim, avg_lpips, avg_id

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    def visualize_batch(self, I, M, I_w, epoch):
        get_img = lambda x: x[0].cpu().float().numpy().transpose(1, 2, 0).clip(0, 1)

        orig_img = get_img(I)
        mask_np = M[0].cpu().numpy().transpose(1, 2, 0)
        masked_region = orig_img * mask_np
        w_img = get_img(I_w)
        residual = get_img(torch.abs(I - I_w) * 10)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Original", "Masked Region", "Watermarked", "Residual x10"]
        images = [orig_img, masked_region, w_img, residual]

        for ax, title, img in zip(axes, titles, images):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        viz_dir = os.path.join(self.config['path_config']['log_dir'], 'viz')
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, f"epoch_{epoch}.png"), bbox_inches='tight')
        plt.close(fig)

    # =====================================================================
    # CHECKPOINTING - Saves all 4 networks
    # =====================================================================
    def save_checkpoint(self, name):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(),
                    os.path.join(self.checkpoint_dir, f'encoder_{name}.pt'))
        torch.save(self.decoder.state_dict(),
                    os.path.join(self.checkpoint_dir, f'decoder_{name}.pt'))
        torch.save(self.discriminator.state_dict(),
                    os.path.join(self.checkpoint_dir, f'discriminator_{name}.pt'))
        torch.save(self.adversary.state_dict(),
                    os.path.join(self.checkpoint_dir, f'adversary_{name}.pt'))
