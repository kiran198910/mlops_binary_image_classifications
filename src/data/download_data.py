"""
Data Download Script for Cats vs Dogs Dataset
Downloads the dataset from Kaggle or uses a public URL
"""

import os
import zipfile
import shutil
import urllib.request
from pathlib import Path
import yaml
import argparse


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(data_dir: str = "data/raw") -> str:
    """
    Download the Cats vs Dogs dataset.
    Uses Microsoft's public dataset URL.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Microsoft's Cats vs Dogs dataset (subset)
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    zip_path = os.path.join(data_dir, "cats_dogs.zip")
    
    if os.path.exists(zip_path):
        print(f"Dataset already downloaded at {zip_path}")
        return zip_path
    
    print(f"Downloading dataset from {url}...")
    print("This may take a few minutes...")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        print(f"Failed to download from Microsoft URL: {e}")
        print("Creating sample dataset for demonstration...")
        create_sample_dataset(data_dir)
        return None
    
    return zip_path


def extract_dataset(zip_path: str, extract_dir: str = "data/raw") -> str:
    """Extract the downloaded zip file."""
    if zip_path is None:
        return None
        
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Extracted to {extract_dir}")
    return extract_dir


def create_sample_dataset(data_dir: str, num_samples: int = 100) -> None:
    """
    Create a sample dataset with placeholder images for testing.
    This is used when the actual dataset cannot be downloaded.
    """
    from PIL import Image
    import numpy as np
    
    print("Creating sample dataset for demonstration...")
    
    cats_dir = os.path.join(data_dir, "PetImages", "Cat")
    dogs_dir = os.path.join(data_dir, "PetImages", "Dog")
    
    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create cat-like images (more blue/gray tones)
        cat_img = np.random.randint(100, 180, (150, 150, 3), dtype=np.uint8)
        cat_img[:, :, 0] = np.clip(cat_img[:, :, 0] - 30, 0, 255)  # Less red
        cat_pil = Image.fromarray(cat_img)
        cat_pil.save(os.path.join(cats_dir, f"cat_{i}.jpg"))
        
        # Create dog-like images (more brown/warm tones)
        dog_img = np.random.randint(100, 200, (150, 150, 3), dtype=np.uint8)
        dog_img[:, :, 2] = np.clip(dog_img[:, :, 2] - 40, 0, 255)  # Less blue
        dog_pil = Image.fromarray(dog_img)
        dog_pil.save(os.path.join(dogs_dir, f"dog_{i}.jpg"))
    
    print(f"Created {num_samples} sample images for cats and dogs")


def organize_dataset(raw_dir: str, config: dict) -> dict:
    """
    Organize dataset into train/validation/test splits.
    """
    import random
    
    processed_dir = config['data']['processed_data_path']
    train_dir = config['data']['train_path']
    val_dir = config['data']['validation_path']
    test_dir = config['data']['test_path']
    
    # Create directories
    for split in [train_dir, val_dir, test_dir]:
        for cls in ['cats', 'dogs']:
            os.makedirs(os.path.join(split, cls), exist_ok=True)
    
    # Find source directories
    pet_images_dir = os.path.join(raw_dir, "PetImages")
    
    if not os.path.exists(pet_images_dir):
        print(f"PetImages directory not found at {pet_images_dir}")
        return {}
    
    cats_src = os.path.join(pet_images_dir, "Cat")
    dogs_src = os.path.join(pet_images_dir, "Dog")
    
    val_split = config['data']['validation_split']
    test_split = config['data']['test_split']
    train_split = 1.0 - val_split - test_split
    
    stats = {'train': {'cats': 0, 'dogs': 0}, 
             'validation': {'cats': 0, 'dogs': 0},
             'test': {'cats': 0, 'dogs': 0}}
    
    for class_name, src_dir, dst_name in [("Cat", cats_src, "cats"), ("Dog", dogs_src, "dogs")]:
        if not os.path.exists(src_dir):
            print(f"Source directory {src_dir} not found")
            continue
            
        files = [f for f in os.listdir(src_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        
        # Calculate split indices
        n_train = int(len(files) * train_split)
        n_val = int(len(files) * val_split)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copy files to respective directories
        for split_name, split_files, split_dir in [
            ('train', train_files, train_dir),
            ('validation', val_files, val_dir),
            ('test', test_files, test_dir)
        ]:
            for f in split_files:
                src_path = os.path.join(src_dir, f)
                dst_path = os.path.join(split_dir, dst_name, f)
                try:
                    shutil.copy2(src_path, dst_path)
                    stats[split_name][dst_name] += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
    
    print("\nDataset organization complete!")
    print(f"Train: {stats['train']}")
    print(f"Validation: {stats['validation']}")
    print(f"Test: {stats['test']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Cats vs Dogs dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading and only organize existing data")
    parser.add_argument("--sample-only", action="store_true",
                        help="Create sample dataset only (for testing)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    raw_dir = config['data']['raw_data_path']
    
    if args.sample_only:
        create_sample_dataset(raw_dir)
    elif not args.skip_download:
        zip_path = download_dataset(raw_dir)
        if zip_path:
            extract_dataset(zip_path, raw_dir)
    
    # Organize into train/val/test
    stats = organize_dataset(raw_dir, config)
    
    print("\nData preparation complete!")
    return stats


if __name__ == "__main__":
    main()
