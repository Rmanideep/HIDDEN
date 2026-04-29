# -*- coding: utf-8 -*-
"""
Final Research Evaluation & Benchmarking Script.
Generates all metrics, tables, and visualization grids for the research paper.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import yaml
import kornia.metrics as K_metrics
from PIL import Image

from src.models.hybrid_model import HybridEncoder, HybridDecoder
from src.data.dataset import get_dataloader
from src.engine.math import DiffJPEG
from src.attacks.attacks import BenignAugmentationPipeline, MalignAttackGenerator

def save_image(tensor, path):
    """Saves a torch tensor [3, H, W] to file."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def create_viz_grid(original, watermarked, mask, attacked, diff_factor=10.0):
    """Creates a comparison grid for the paper."""
    diff = torch.abs(watermarked - original) * diff_factor
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    imgs = [original, watermarked, diff, mask, attacked]
    titles = ['Original', 'Watermarked', f'Residual (x{diff_factor})', 'Face Mask', 'Attacked (JPEG)']
    
    for i, (img, title) in enumerate(zip(imgs, titles)):
        # If mask (1 channel)
        if img.shape[0] == 1:
            axes[i].imshow(img[0].detach().cpu().numpy(), cmap='gray')
        else:
            axes[i].imshow(img.detach().cpu().numpy().transpose(1, 2, 0))
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

@torch.no_grad()
def run_evaluation():
    # 1. Load Configuration
    with open('configs/v100_train.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results/visuals', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    
    # 2. Initialize Models
    n_bits = config['model_config']['n_bits']
    encoder = HybridEncoder(n_bits=n_bits).to(device)
    decoder = HybridDecoder(n_bits=n_bits).to(device)
    
    # Load Best Weights
    ckpt_dir = config['path_config']['checkpoint_dir']
    encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, 'encoder_best_weights.pt'), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(ckpt_dir, 'decoder_best_weights.pt'), map_location=device))
    encoder.eval()
    decoder.eval()
    
    print("[*] Models Loaded. Initializing Test Dataloader...")
    
    # 3. Setup Test Dataloader
    test_loader = get_dataloader(config, is_val=True)
    
    # 4. Results Accumulators
    results = {
        'Condition': [],
        'PSNR': [],
        'SSIM': [],
        'LPIPS': [],
        'BRA (%)': []  # Bit Recovery Accuracy (1 - BER)
    }
    
    # For AUC Calculation
    all_scores = []  # BER values used as "tamper scores"
    all_labels = []  # 0 for benign, 1 for deepfake
    
    # Attack sets for sweep
    jpeg_qualities = [90, 70, 50, 30]
    malign_attacks = MalignAttackGenerator(device=device)
    
    # Perceptual Loss for metrics
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    print(f"[*] Starting Evaluation on {len(test_loader)} batches...")
    
    # 5. Core Evaluation Loop
    batch_idx = 0
    for I, M, m_gt, I_donor in tqdm(test_loader):
        # Full evaluation on all test images for maximum accuracy
        
        I, M, m_gt, I_donor = I.to(device), M.to(device), m_gt.to(device), I_donor.to(device)
        msg_encoded = m_gt 
        
        # --- Baseline (Clean) ---
        I_w = encoder(I, M, msg_encoded)
        
        # Fidelity Metrics
        psnr = K_metrics.psnr(I_w, I, max_val=1.0).mean().item()
        ssim = K_metrics.ssim(I_w, I, window_size=11, max_val=1.0).mean().item()
        with torch.no_grad():
            lpips_val = lpips_fn(I_w * 2 - 1, I * 2 - 1).mean().item()
        
        # --- Test Scenarios ---
        scenarios = {
            'Benign (Clean)': I_w,
        }
        for q in jpeg_qualities:
            jpeg_layer = DiffJPEG(quality=q).to(device)
            scenarios[f'JPEG_Q{q}'] = jpeg_layer(I_w)
            
        scenarios['Deepfake (Swap)'] = malign_attacks.semantic_face_swap(I_w, I_donor, M)
        scenarios['Deepfake (Latent)'] = malign_attacks.latent_distorter(I_w, M)
        
        # Run scenarios
        for name, img_attacked in scenarios.items():
            logits = decoder(img_attacked)
            
            # BER and BRA (Bit Recovery Accuracy)
            raw_ber = ((logits > 0.0).float() != msg_encoded).float().mean().item()
            bra = (1.0 - raw_ber) * 100
            
            results['Condition'].append(name)
            results['PSNR'].append(psnr)
            results['SSIM'].append(ssim)
            results['LPIPS'].append(lpips_val)
            results['BRA (%)'].append(bra)
            
            # AUC Tracking
            if name == 'Benign (Clean)':
                all_scores.append(raw_ber)
                all_labels.append(0)
            elif 'Deepfake' in name:
                all_scores.append(raw_ber)
                all_labels.append(1)
            
        # 6. Save Visualizations for the first batch
        if batch_idx == 0:
            for i in range(min(5, I.shape[0])):
                grid = create_viz_grid(I[i], I_w[i], M[i], scenarios['JPEG_Q50'][i])
                grid.savefig(f'results/visuals/paper_grid_{i}.png')
                plt.close(grid)
                save_image(I[i], f'results/visuals/sample_{i}_orig.png')
                save_image(I_w[i], f'results/visuals/sample_{i}_watermarked.png')
                save_image(torch.abs(I_w[i]-I[i])*15.0, f'results/visuals/sample_{i}_residual_x15.png')
                save_image(scenarios['Deepfake (Swap)'][i], f'results/visuals/sample_{i}_deepfaked.png')

        batch_idx += 1

    # 7. Aggregate and Save Results Table
    df = pd.DataFrame(results)
    summary = df.groupby('Condition').mean().reset_index()
    summary.to_csv('results/tables/test_metrics_summary.csv', index=False)
    
    # 8. Calculate AUC & Fragility Gap
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(all_labels, all_scores)
    
    # Calculate Contrast Metrics
    avg_benign_bra = summary[~summary['Condition'].str.contains('Deepfake')]['BRA (%)'].mean()
    avg_malign_bra = summary[summary['Condition'].str.contains('Deepfake')]['BRA (%)'].mean()
    fragility_gap = avg_benign_bra - avg_malign_bra
    
    # Generate LaTeX Table snippet
    print("\n" + "="*60)
    print("      RESEARCH PAPER RESULTS (Hybrid Domain System)")
    print("="*60)
    print(f" Detection AUC:      {auc_score:.4f}")
    print(f" Avg. Benign BRA:    {avg_benign_bra:.2f}% (Robustness)")
    print(f" Avg. Malicious BRA: {avg_malign_bra:.2f}% (Fragility)")
    print(f" Fragility Gap:      {fragility_gap:.2f}%")
    print("-" * 60)
    print(summary.to_latex(index=False, float_format="%.4f"))
    print("="*60)
    
    print(f"\n[*] Evaluation Complete! Results saved to 'results/' directory.")

if __name__ == "__main__":
    run_evaluation()
