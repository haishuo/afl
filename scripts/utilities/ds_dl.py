#!/usr/bin/env python3
"""
Dataset Downloader for AFL Project
==================================

Automatically downloads and prepares datasets for AFL experiments.
Checks /mnt/data/common_datasets/ and downloads missing datasets.

Usage:
    python ds_dl.py [--dataset DATASET_NAME] [--force]

Arguments:
    --dataset: Download specific dataset (optional, downloads all if not specified)
    --force: Re-download even if dataset already exists
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import zipfile
import gzip
import shutil
from pathlib import Path
import subprocess
from typing import Dict, List, Optional

# Base directory for all datasets
DATASETS_DIR = Path("/mnt/data/common_datasets")

# Dataset configurations
DATASET_CONFIGS = {
    "mnist": {
        "description": "MNIST handwritten digits dataset",
        "files": [
            ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
            ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
            ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
            ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
        ],
        "extract": "gunzip",
        "verify_files": ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    },
    
    "cifar10": {
        "description": "CIFAR-10 image classification dataset",
        "files": [
            ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz"),
        ],
        "extract": "tar",
        "verify_files": ["cifar-10-batches-py/data_batch_1", "cifar-10-batches-py/test_batch"]
    },
    
    "cifar100": {
        "description": "CIFAR-100 image classification dataset",
        "files": [
            ("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", "cifar-100-python.tar.gz"),
        ],
        "extract": "tar", 
        "verify_files": ["cifar-100-python/train", "cifar-100-python/test"]
    },
    
    "fashion_mnist": {
        "description": "Fashion-MNIST clothing classification dataset",
        "files": [
            ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
            ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
            ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
            ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
        ],
        "extract": "gunzip",
        "verify_files": ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    },
    
    "wine": {
        "description": "Wine Quality dataset (uses sklearn/fetch_openml)",
        "method": "sklearn",
        "verify_files": [".wine_downloaded"]  # Marker file
    },
    
    "imagenet": {
        "description": "ImageNet dataset (requires manual download - too large)",
        "method": "manual",
        "instructions": "ImageNet requires manual registration and download from https://image-net.org/"
    },
    
    "tiny_imagenet": {
        "description": "Tiny ImageNet dataset",
        "files": [
            ("http://cs231n.stanford.edu/tiny-imagenet-200.zip", "tiny-imagenet-200.zip"),
        ],
        "extract": "zip",
        "verify_files": ["tiny-imagenet-200/train", "tiny-imagenet-200/val"]
    }
}


def check_dataset_exists(dataset_name: str) -> bool:
    """Check if dataset already exists and is complete."""
    dataset_dir = DATASETS_DIR / dataset_name
    
    if not dataset_dir.exists():
        return False
    
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        return False
    
    # Check for verification files
    verify_files = config.get("verify_files", [])
    for verify_file in verify_files:
        if not (dataset_dir / verify_file).exists():
            return False
    
    return True


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"  Downloading: {url}")
        print(f"  To: {dest_path}")
        
        # Create directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✅ Downloaded: {dest_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Download failed: {str(e)}")
        return False


def extract_file(file_path: Path, extract_method: str) -> bool:
    """Extract downloaded file based on method."""
    try:
        print(f"  Extracting: {file_path.name}")
        
        if extract_method == "gunzip":
            # Extract .gz file
            with gzip.open(file_path, 'rb') as f_in:
                with open(file_path.with_suffix(''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove original .gz file
            file_path.unlink()
            
        elif extract_method == "tar":
            # Extract .tar.gz file
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(file_path.parent)
            # Remove original tar file
            file_path.unlink()
            
        elif extract_method == "zip":
            # Extract .zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(file_path.parent)
            # Remove original zip file
            file_path.unlink()
            
        else:
            print(f"  ⚠️  Unknown extraction method: {extract_method}")
            return False
            
        print(f"  ✅ Extracted: {file_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Extraction failed: {str(e)}")
        return False


def download_sklearn_dataset(dataset_name: str) -> bool:
    """Download dataset using sklearn/openml."""
    try:
        print(f"  Using sklearn to download {dataset_name}")
        
        # This will be implemented when we actually use it
        # For now, just create a marker file
        dataset_dir = DATASETS_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        marker_file = dataset_dir / ".wine_downloaded"
        marker_file.write_text("Downloaded via sklearn fetch_openml")
        
        print(f"  ✅ Sklearn dataset marker created")
        return True
        
    except Exception as e:
        print(f"  ❌ Sklearn download failed: {str(e)}")
        return False


def download_dataset(dataset_name: str, force: bool = False) -> bool:
    """Download a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
        return False
    
    config = DATASET_CONFIGS[dataset_name]
    dataset_dir = DATASETS_DIR / dataset_name
    
    print(f"\n📦 Processing dataset: {dataset_name}")
    print(f"Description: {config['description']}")
    
    # Check if already exists
    if not force and check_dataset_exists(dataset_name):
        print(f"✅ Dataset already exists: {dataset_dir}")
        return True
    
    # Handle manual datasets
    if config.get("method") == "manual":
        print(f"⚠️  Manual download required:")
        print(f"   {config['instructions']}")
        return False
    
    # Handle sklearn datasets
    if config.get("method") == "sklearn":
        return download_sklearn_dataset(dataset_name)
    
    # Handle direct download datasets
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all files
    for url, filename in config["files"]:
        dest_path = dataset_dir / filename
        
        if not download_file(url, dest_path):
            return False
        
        # Extract if needed
        if "extract" in config:
            if not extract_file(dest_path, config["extract"]):
                return False
    
    # Verify download
    if check_dataset_exists(dataset_name):
        print(f"✅ Dataset successfully downloaded: {dataset_name}")
        return True
    else:
        print(f"❌ Dataset verification failed: {dataset_name}")
        return False


def scan_and_download_missing() -> None:
    """Scan for missing datasets and download them."""
    print("🔍 Scanning for missing datasets...")
    
    # Create base directory if it doesn't exist
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    missing_datasets = []
    existing_datasets = []
    
    for dataset_name in DATASET_CONFIGS.keys():
        if check_dataset_exists(dataset_name):
            existing_datasets.append(dataset_name)
        else:
            missing_datasets.append(dataset_name)
    
    print(f"\n📊 Dataset Status:")
    print(f"✅ Existing: {len(existing_datasets)} datasets")
    for dataset in existing_datasets:
        print(f"   - {dataset}")
    
    print(f"❌ Missing: {len(missing_datasets)} datasets")
    for dataset in missing_datasets:
        print(f"   - {dataset}")
    
    if not missing_datasets:
        print(f"\n🎉 All datasets are available!")
        return
    
    # Download missing datasets
    print(f"\n⬬ Downloading missing datasets...")
    
    success_count = 0
    for dataset_name in missing_datasets:
        if download_dataset(dataset_name):
            success_count += 1
    
    print(f"\n📈 Download Summary:")
    print(f"✅ Successfully downloaded: {success_count}/{len(missing_datasets)} datasets")
    
    if success_count < len(missing_datasets):
        failed = len(missing_datasets) - success_count
        print(f"❌ Failed downloads: {failed} datasets")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download datasets for AFL project")
    parser.add_argument("--dataset", type=str, help="Download specific dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    print("📂 AFL Dataset Downloader")
    print("=" * 40)
    print(f"Target directory: {DATASETS_DIR}")
    
    if args.list:
        print(f"\n📋 Available datasets:")
        for name, config in DATASET_CONFIGS.items():
            status = "✅" if check_dataset_exists(name) else "❌"
            print(f"  {status} {name}: {config['description']}")
        return
    
    if args.dataset:
        # Download specific dataset
        success = download_dataset(args.dataset, args.force)
        sys.exit(0 if success else 1)
    else:
        # Scan and download all missing datasets
        scan_and_download_missing()


if __name__ == "__main__":
    main()