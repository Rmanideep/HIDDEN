# -*- coding: utf-8 -*-
import yaml
import argparse
import os
import torch
import torch.optim as optim
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure the root directory is in the path to allow absolute imports from 'src'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.hybrid_model import HybridEncoder, HybridDecoder
from src.models.adversarial import WassersteinCritic, AdversaryNet
from src.data.dataset import get_dataloader
from src.engine.trainer import WatermarkTrainer

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

    encoder = HybridEncoder(n_bits=n_bits, encoder_scale=config['model_config'].get('encoder_scale', 1.0))
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
    print(f"[*] watermark_length: {config['model_config']['watermark_length']}")
    print(f"[*] encoder_scale: {config['model_config'].get('encoder_scale', 1.0)}")

    # 5. Data Loading (Synchronized with your dataset.py signature)
    train_loader = get_dataloader(config, is_val=False)
    val_loader = get_dataloader(config, is_val=True) 
    
    # 6. Build the 3-way adversarial trainer
    trainer = WatermarkTrainer(
        config, encoder, decoder, discriminator, adversary,
        train_loader, val_loader
    )

    # Extract optimizer hyperparameters from config
    base_lr = config['training_config']['lr']
    disc_lr_mult = config['training_config']['disc_lr_mult']
    betas = tuple(config['training_config']['betas'])

    trainer.opt_encdec = optim.Adam([
        {'params': trainer.encoder.parameters(), 'lr': base_lr, 'name': 'encoder'},
        {'params': trainer.decoder.parameters(), 'lr': base_lr * 2.0, 'name': 'decoder'},
    ], betas=betas)
    
    trainer.scheduler_encdec = ReduceLROnPlateau(
        trainer.opt_encdec, mode='min', factor=0.3, patience=12, min_lr=1e-6, verbose=True
    )
    print(f"Joint training: enc_lr={base_lr:.5f}, dec_lr={base_lr:.5f}")
    print(f"scheduler_encdec initialized.")
    
    epochs = config['training_config']['epochs']
    best_score = -1.0
    patience = config['training_config'].get('early_stop_patience', 20)
    no_improve_count = 0
    
    print(f"\n{'='*60}")
    print(f"  Hybrid Domain Semi-Fragile Watermarking System")
    print(f"  L_RE Loss: L1(benign) - L1(malicious) from Epoch 1")
    print(f"  Epochs: {epochs} | Device: {config['training_config']['device']}")
    print(f"  LR: {base_lr} | Batch Size: {config['data_config']['batch_size']}")
    print(f"  Patience: {patience} | All networks active from Epoch 1")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # =================================================================
        # HYBRID DOMAIN APPROACH: No all networks from Epoch 1
        # =================================================================
        
        # Set all weights to their active values (from config)
        trainer.encoder.encoder_scale    = config['model_config'].get('encoder_scale', 1.0)
        trainer.aug_prob                 = 0.5  # Reduced to 0.5 to stabilize signal acquisition
        trainer.criterion.W_L1           = config['loss_config']['lambda_l1']
        trainer.criterion.W_LPIPS        = config['loss_config']['lambda_lpips']
        trainer.criterion.W_SSIM         = config['loss_config']['lambda_ssim']
        trainer.criterion.W_ID           = config['loss_config']['lambda_id']
        trainer.criterion.W_BIT_BENIGN   = config['loss_config']['lambda_bit_benign']
        trainer.criterion.W_BIT_FRAGILE  = config['loss_config']['lambda_bit_fragile']  # L_RE loss weight
        trainer.criterion.W_DISC         = config['loss_config']['lambda_disc']
        trainer.criterion.W_ADV          = config['loss_config']['lambda_adv']
        
        phase = f"HYBRID DOMAIN: All losses active (L_RE semi-fragile loss)"

        print(f"\n{'-'*60}")
        print(f"[*] {phase} | Epoch {epoch}/{epochs}")
        print(f"{'-'*60}")
        
        trainer.train_epoch(epoch)
        
        # 7. Evaluation & Checkpointing
        if epoch % trainer.eval_freq == 0:
            eval_results = trainer.eval_epoch(epoch)
            ber_benign, ber_attacked, ber_adv, psnr, ssim, lpips, id_loss = eval_results

            # Cache latest BER for phase gate
            last_eval_ber = ber_benign

            # --- Step LR schedulers with current BER (ReduceLROnPlateau) ---
            trainer.scheduler_encdec.step(ber_benign)
            if trainer.criterion.W_DISC > 0:
                trainer.scheduler_disc.step(ber_benign)
            if trainer.criterion.W_ADV > 0:
                trainer.scheduler_adv.step(ber_benign)

            # --- HYBRID DOMAIN SCORING: All networks active from start ---
            # Goal: Maximize robustness (benign BER) while maintaining imperceptibility (PSNR/SSIM)
            # Score: (1 - benign_BER) * (PSNR/48), where 48dB is typical baseline
            baseline_psnr = 48.0
            current_score = (1.0 - ber_benign) * (psnr / baseline_psnr)
            
            # Checkpointing
            if current_score > best_score:
                print(f"\n{'+'*60}")
                print(f"  [BEST] New best score! {best_score:.4f} -> {current_score:.4f}")
                print(f"  Benign BER: {ber_benign*100:.1f}% | Malign BER: {ber_attacked*100:.1f}%")
                print(f"  PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f}")
                print(f"{'+'*60}\n")
                best_score = current_score
                no_improve_count = 0
                trainer.save_checkpoint('best_weights')
            else:
                no_improve_count += 1
                print(f"\n{'-'*60}")
                print(f"  [NO IMPROVE] ({no_improve_count}/{patience}) | Best Score: {best_score:.4f}")
                print(f"  Current: BER={ber_benign*100:.1f}% | PSNR={psnr:.2f}dB")
                print(f"{'-'*60}\n")
                if no_improve_count >= patience:
                    print(f"\n{'!'*60}")
                    print(f"[EARLY STOPPING] No improvement for {patience} epochs at Epoch {epoch}.")
                    print(f"[FINAL BEST SCORE] {best_score:.4f}")
                    print(f"{'!'*60}")
                    break

if __name__ == "__main__":
    main()

