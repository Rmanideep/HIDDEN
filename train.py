# -*- coding: utf-8 -*-
import yaml
import argparse
import os
import torch
from models.hybrid_model import HybridEncoder, HybridDecoder
from models.adversarial import WassersteinCritic, AdversaryNet
from data.dataset import get_dataloader
from core.trainer import WatermarkTrainer

def load_config(config_path):
    """
    Loads YAML config and performs numerical safety casting to prevent PyTorch Optimizer errors.
    Ensures scientific notation (e.g., 1e-4) in YAML is treated as float, not string.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- NUMERICAL SAFETY: CAST STRINGS TO FLOATS ---
    tc = config['training_config']
    tc['lr'] = float(tc['lr'])
    tc['disc_lr_mult'] = float(tc['disc_lr_mult'])
    tc['betas'] = [float(b) for b in tc['betas']]
    
    mc = config['model_config']
    mc['adversary_epsilon'] = float(mc['adversary_epsilon'])
    
    # Cast all loss weights
    lc = config['loss_config']
    for key in lc:
        if key.startswith('lambda_'):
            lc[key] = float(lc[key])
            
    return config

def main():
    parser = argparse.ArgumentParser(description="Deepfake Watermarking Research System")
    # Default to your V100 optimized config
    parser.add_argument('--config', type=str, default='configs/v100_train.yaml', help='Path to config file')
    args = parser.parse_args()

    # 1. Load and clean config types
    config = load_config(args.config)

    # V100 OPTIMIZATION: Enable CuDNN auto-tuner for static input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 2. GPU Isolation Logic
    device_str = config['training_config']['device']
    if 'cuda:' in device_str:
        gpu_id = device_str.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config['training_config']['device'] = 'cuda:0' 
        print(f"[*] Isolated GPU {gpu_id}. Using cuda:0 internally.")

    # 3. Directories
    os.makedirs(config['path_config']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['path_config']['log_dir'], exist_ok=True)

    # 4. Architecture Initialization
    n_bits = config['model_config']['n_bits']
    ndf = config['model_config']['discriminator_ndf']
    adv_f = config['model_config']['adversary_base_filters']
    eps = config['model_config']['adversary_epsilon']

    encoder = HybridEncoder(n_bits=n_bits, encoder_scale=config['model_config'].get('encoder_scale', 0.2))
    decoder = HybridDecoder(n_bits=n_bits)
    discriminator = WassersteinCritic(in_channels=3, ndf=ndf)
    adversary = AdversaryNet(in_channels=3, base_filters=adv_f, epsilon=eps)

    # 5. Resume Capability
    ckpt_dir = config['path_config']['checkpoint_dir']
    resume_path = os.path.join(ckpt_dir, 'best_weights')
    if os.path.exists(os.path.join(ckpt_dir, 'encoder_best_weights.pt')):
        encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, 'encoder_best_weights.pt'), map_location='cpu'))
        decoder.load_state_dict(torch.load(os.path.join(ckpt_dir, 'decoder_best_weights.pt'), map_location='cpu'))
        print('[INFO] Resuming training from previous state.')

    print(f"[*] Architecture Loaded. Initializing Dataloaders...")
    print(f"[*] Config:        {args.config}")
    print(f"[*] n_bits:        {n_bits} | ndf: {ndf} | adv_filters: {adv_f} | epsilon: {eps}")

    # 5. Data Loading (Synchronized with your dataset.py signature)
    train_loader = get_dataloader(config, is_val=False)
    val_loader = get_dataloader(config, is_val=True) 
    
    # 6. Build the 3-way adversarial trainer
    trainer = WatermarkTrainer(
        config, encoder, decoder, discriminator, adversary,
        train_loader, val_loader
    )
    
    epochs = config['training_config']['epochs']
    best_score = -1.0
    
    # Early Stopping
    patience = config['training_config'].get('early_stop_patience', 7)
    no_improve_count = 0
    
    print(f"\nStarting SOTA Training Loop for {epochs} Epochs on {config['training_config']['device']}...")
    print(f"  (Using Gradient Accumulation: {trainer.accumulation_steps} steps)")
    print(f"  (Early Stopping Patience: {patience} evals)")
    
    for epoch in range(1, epochs + 1):
        # --- 3-PHASE PROGRESSIVE CURRICULUM ---
        # Forces convergence by isolating objectives in stages
        target_bit_benign = config['loss_config']['lambda_bit_benign']
        lc = config['loss_config']
        
        if epoch <= 3:
            # Phase 1: BIT-FIRST (break identity deadlock)
            # Suppress quality weights to 1/3 so bit gradients can overcome them.
            # The encoder MUST embed signal before quality refinement begins.
            trainer.criterion.W_L1          = lc['lambda_l1'] / 3.0
            trainer.criterion.W_LPIPS       = lc['lambda_lpips'] / 3.0
            trainer.criterion.W_SSIM        = lc['lambda_ssim'] / 3.0
            trainer.criterion.W_ID          = lc['lambda_id'] / 3.0
            trainer.criterion.W_BIT_BENIGN  = 80.0
            trainer.criterion.W_BIT_FRAGILE = 0.0
            trainer.criterion.W_DISC        = 0.0
            trainer.criterion.W_ADV         = 0.0
            phase_info = "Phase 1: Bit-First (quality suppressed)"
        elif epoch <= 10:
            # Phase 2: Restore quality weights + ramp bit weight to target
            # Linearly restore quality from 1/3 -> full over epochs 4-10
            q_progress = (epoch - 3) / 7.0  # epoch 4->1/7, epoch 10->7/7
            q_scale = (1.0 / 3.0) + (2.0 / 3.0) * q_progress
            trainer.criterion.W_L1          = lc['lambda_l1'] * q_scale
            trainer.criterion.W_LPIPS       = lc['lambda_lpips'] * q_scale
            trainer.criterion.W_SSIM        = lc['lambda_ssim'] * q_scale
            trainer.criterion.W_ID          = lc['lambda_id'] * q_scale
            # Ramp bit weight from 80 -> target
            b_progress = (epoch - 3) / 7.0
            w_bit = 80.0 + (target_bit_benign - 80.0) * b_progress
            trainer.criterion.W_BIT_BENIGN  = w_bit
            trainer.criterion.W_BIT_FRAGILE = 0.0
            trainer.criterion.W_DISC        = 0.0
            trainer.criterion.W_ADV         = 0.0
            phase_info = f"Phase 2: Quality Restore (q={q_scale:.2f}, bit={w_bit:.1f})"
        else:
            # Phase 3: The Arms Race (Full Adversarial Learning)
            trainer.criterion.W_L1          = lc['lambda_l1']
            trainer.criterion.W_LPIPS       = lc['lambda_lpips']
            trainer.criterion.W_SSIM        = lc['lambda_ssim']
            trainer.criterion.W_ID          = lc['lambda_id']
            trainer.criterion.W_BIT_BENIGN  = target_bit_benign
            trainer.criterion.W_BIT_FRAGILE = lc['lambda_bit_fragile']
            trainer.criterion.W_DISC        = lc['lambda_disc']
            trainer.criterion.W_ADV         = lc['lambda_adv']
            phase_info = "Phase 3: The Arms Race (Full Adversarial)"

        print(f"\n{'-'*60}")
        print(f"[*] {phase_info} | Epoch {epoch}")
        print(f"{'-'*60}")
        
        trainer.train_epoch(epoch)
        
        # 7. Evaluation & Checkpointing (Respects configured eval_freq)
        if epoch % trainer.eval_freq == 0:
            eval_results = trainer.eval_epoch(epoch)
            ber_benign, ber_attacked, ber_adv, psnr, ssim, lpips, id_loss = eval_results
            
            # --- QUALITY OF SERVICE (QoS) SCORE ---
            # Scientifically balances bit recovery (1-BER) with image quality (PSNR relative to 40dB)
            current_score = (1.0 - ber_benign) * (psnr / 40.0)
            
            # Checkpointing based on QoS Score
            if current_score > best_score:
                print(f"  [++] New optimal weights! Score improved from {best_score:.4f} to {current_score:.4f}")
                print(f"      (BER: {ber_benign*100:.2f}% | PSNR: {psnr:.2f}dB)")
                best_score = current_score
                no_improve_count = 0
                trainer.save_checkpoint('best_weights')
            else:
                no_improve_count += 1
                print(f"  [--] No improvement ({no_improve_count}/{patience}). Best QoS: {best_score:.4f}")
                if no_improve_count >= patience:
                    print(f"\n[!] Early stopping triggered at epoch {epoch} (no improvement for {patience} evals).")
                    print(f"    Best QoS Score: {best_score:.4f}")
                    break
            
if __name__ == "__main__":
    main()
