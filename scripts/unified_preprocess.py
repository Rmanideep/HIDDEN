import os
import shutil
import pandas as pd
import numpy as np
import random
import multiprocessing
import yaml
from PIL import Image
from tqdm import tqdm

def process_image(task):
    img_name, src_dir, dst_dir, target_size, quality = task
    src_path = os.path.join(src_dir, img_name)
    dst_path = os.path.join(dst_dir, img_name)
    
    try:
        with Image.open(src_path) as img:
            # Convert to RGB to ensure jpeg compatibility
            img = img.convert('RGB')
            # LANCZOS resizing to 256x256
            if hasattr(Image, 'Resampling'):
                resample_filter = Image.Resampling.LANCZOS
            else:
                resample_filter = Image.LANCZOS
                
            img_resized = img.resize((target_size, target_size), resample_filter)
            img_resized.save(dst_path, format='JPEG', quality=quality)
        return True
    except Exception as e:
        return False

def main():
    print("=" * 60)
    print("UNIFIED PREPROCESSING SCRIPT")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/v100_train.yaml')
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(root_dir, args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"[*] Loaded Config: {config_path}")
    
    raw_img_dir = os.path.join(root_dir, config['path_config']['raw_path'], 'img_align_celeba')
    identity_file = os.path.join(root_dir, config['path_config']['identity_file'])
    landmarks_file = os.path.join(root_dir, config['path_config']['landmarks_file'])
    
    processed_base_dir = os.path.join(root_dir, config['path_config']['processed_path'])
    processed_img_dir = os.path.join(processed_base_dir, 'images')
    
    # Task 1: Environment Reset
    print("[*] Resetting environment...")
    if os.path.exists(processed_base_dir):
        shutil.rmtree(processed_base_dir)
    os.makedirs(processed_img_dir, exist_ok=True)
    
    # Task 2: Identity Mapping
    print(f"[*] Reading Identity Mapping: {identity_file}")
    if not os.path.exists(identity_file):
        print(f"[Error]: {identity_file} not found.")
        return
        
    identities_df = pd.read_csv(identity_file, sep=r'\s+', names=['image_id', 'person_id'])
    
    # Group images by person_id
    person_image_map = {}
    for _, row in identities_df.iterrows():
        person_image_map.setdefault(row['person_id'], []).append(row['image_id'])
        
    unique_persons = list(person_image_map.keys())
    random.seed(42)
    random.shuffle(unique_persons)
    
    # Task 3: Identity-Based Research Split
    TARGET_TRAIN = config['data_config']['target_train']
    TARGET_VAL = config['data_config']['target_val']
    TARGET_TEST = config['data_config']['target_test']
    
    train_ids = set()
    val_ids = set()
    test_ids = set()
    
    train_images = []
    val_images = []
    test_images = []

    print("[*] Allocating identities to strict Disjoint splits...")
    
    for pid in unique_persons:
        imgs = person_image_map[pid]
        
        # Test Bucket
        if len(test_images) < TARGET_TEST:
            needed = TARGET_TEST - len(test_images)
            test_ids.add(pid)
            test_images.extend(imgs[:needed])
            
        # Val Bucket
        elif len(val_images) < TARGET_VAL:
            needed = TARGET_VAL - len(val_images)
            val_ids.add(pid)
            val_images.extend(imgs[:needed])
            
        # Train Bucket
        elif len(train_images) < TARGET_TRAIN:
            needed = TARGET_TRAIN - len(train_images)
            train_ids.add(pid)
            train_images.extend(imgs[:needed])
        else:
            # We reached 100k exact
            break

    # Build final selected dataframe
    split_records = []
    for img in train_images: split_records.append({'image_id': img, 'split': 'train', 'person_id': [k for k,v in person_image_map.items() if img in v][0]})
    for img in val_images: split_records.append({'image_id': img, 'split': 'val', 'person_id': [k for k,v in person_image_map.items() if img in v][0]})
    for img in test_images: split_records.append({'image_id': img, 'split': 'test', 'person_id': [k for k,v in person_image_map.items() if img in v][0]})
    
    final_split_df = pd.DataFrame(split_records)
    selected_images = final_split_df['image_id'].tolist()
    
    # Task 4: Parallel Image Processing
    target_sz = config['data_config']['image_size']
    qual = config['model_config']['preprocessing_quality']
    
    print(f"[*] Resizing {len(selected_images)} images to {target_sz}x{target_sz} via exact multiprocessing...")
    tasks = [(img, raw_img_dir, processed_img_dir, target_sz, qual) for img in selected_images]
    
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    # Using Pool
    successful_count = 0
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for result in tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks)):
            if result:
                successful_count += 1
                
    if successful_count != len(selected_images):
        print(f"[Warning]: Only {successful_count}/{len(selected_images)} images found/processed.")
        
    # Task 5: Unified Metadata Generation
    print(f"[*] Normalizing Landmarks from {landmarks_file}")
    
    if os.path.exists(landmarks_file):
        try:
            # CelebA landmark format: Line 0 is count, Line 1 is header
            landmarks_df = pd.read_csv(landmarks_file, sep=r'\s+', skiprows=1)
            
            # If the header doesn't match standard or has no image_id column name, handle it:
            if 'lefteye_x' in landmarks_df.columns and len(landmarks_df.columns) == 10:
                # pandas read_csv with space delim uses index as first column if header is shorter
                landmarks_df = pd.read_csv(landmarks_file, sep=r'\s+', skiprows=2, names=['image_id', 'le_x', 'le_y', 're_x', 're_y', 'nose_x', 'nose_y', 'lm_x', 'lm_y', 'rm_x', 'rm_y'])
            else:
                # If parsed gracefully
                landmarks_df = landmarks_df.reset_index()
                landmarks_df.columns = ['image_id', 'le_x', 'le_y', 're_x', 're_y', 'nose_x', 'nose_y', 'lm_x', 'lm_y', 'rm_x', 'rm_y']

            # Inner join with our 100k generated splits
            merged_df = pd.merge(final_split_df, landmarks_df, on='image_id', how='inner')
            
            # Normalize coordinates to [0, 1] relative to aligned raw size (178w x 218h)
            merged_df['le_x'] = merged_df['le_x'] / 178.0
            merged_df['le_y'] = merged_df['le_y'] / 218.0
            merged_df['re_x'] = merged_df['re_x'] / 178.0
            merged_df['re_y'] = merged_df['re_y'] / 218.0
            merged_df['nose_x'] = merged_df['nose_x'] / 178.0
            merged_df['nose_y'] = merged_df['nose_y'] / 218.0
            merged_df['lm_x'] = merged_df['lm_x'] / 178.0
            merged_df['lm_y'] = merged_df['lm_y'] / 218.0
            merged_df['rm_x'] = merged_df['rm_x'] / 178.0
            merged_df['rm_y'] = merged_df['rm_y'] / 218.0
            
            # Save metadata
            csv_out = os.path.join(processed_base_dir, 'metadata_final.csv')
            merged_df.to_csv(csv_out, index=False)
            print(f"[Success]: Exported unified metadata to {csv_out}")
            
        except Exception as e:
             print(f"[Error] processing landmarks: {e}")
    else:
        print(f"[Error]: {landmarks_file} not found. Ensure this V100 dataset has the aligned landmarks file.")

    # Task 6: CLI Summary Check
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_train_val = train_ids.intersection(val_ids)
    
    print("\n" + "="*40)
    print("PREPROCESSING SUMMARY")
    print("="*40)
    print(f"Total Unique Identities: {len(train_ids) + len(val_ids) + len(test_ids)}")
    print(f"Train Images: {len(train_images)} (Identities: {len(train_ids)})")
    print(f"Val Images:   {len(val_images)} (Identities: {len(val_ids)})")
    print(f"Test Images:  {len(test_images)} (Identities: {len(test_ids)})")
    print(f"Train/Test Overlap: {len(overlap_train_test)} identities")
    print(f"Train/Val Overlap:  {len(overlap_train_val)} identities")
    print("="*40)
    if len(overlap_train_test) == 0 and len(overlap_train_val) == 0:
        print("[OK] Identity leakage test passed! 0 overlaps.")
    else:
        print("[CRITICAL]: Identity overlap detected.")

if __name__ == '__main__':
    main()
