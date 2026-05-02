#!/usr/bin/env python3
"""
Verification script to check if everything is set up correctly
Run this before training to ensure all paths and dependencies are correct
"""

import os
import sys

def check_files():
    """Check if all required files exist"""
    print("="*70)
    print("CHECKING FILES")
    print("="*70)
    
    required_files = [
        'improved_model.py',
        'improved_dataset.py',
        'improved_training.py',
        'compare_models.py',
        'analysis_utils.py',
        'README.md',
        'IMPROVEMENT_GUIDE.md'
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_imports():
    """Check if all required packages can be imported"""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM'
    }
    
    all_imported = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            all_imported = False
    
    return all_imported

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*70)
    print("CHECKING GPU")
    print("="*70)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ CUDA is NOT available - training will be slow on CPU")
        
        return cuda_available
    except:
        print("✗ Could not check CUDA")
        return False

def check_dataset():
    """Check if dataset exists"""
    print("\n" + "="*70)
    print("CHECKING DATASET")
    print("="*70)
    
    data_dir = "../erdes"
    splits_dir = os.path.join(data_dir, "splits", "macula_detached_vs_intact")
    
    # Check data directory
    if os.path.exists(data_dir):
        print(f"✓ Data directory exists: {data_dir}")
    else:
        print(f"✗ Data directory NOT FOUND: {data_dir}")
        print(f"  Please update DATA_DIR in improved_training.py")
        return False
    
    # Check splits directory
    if os.path.exists(splits_dir):
        print(f"✓ Splits directory exists: {splits_dir}")
    else:
        print(f"✗ Splits directory NOT FOUND: {splits_dir}")
        return False
    
    # Check CSV files
    csv_files = ['train.csv', 'val.csv', 'test.csv']
    all_exist = True
    
    for csv_file in csv_files:
        csv_path = os.path.join(splits_dir, csv_file)
        if os.path.exists(csv_path):
            # Count lines
            with open(csv_path, 'r') as f:
                num_samples = len(f.readlines()) - 1  # Subtract header
            print(f"✓ {csv_file}: {num_samples} samples")
        else:
            print(f"✗ {csv_file} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_disk_space():
    """Check available disk space"""
    print("\n" + "="*70)
    print("CHECKING DISK SPACE")
    print("="*70)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/content")
        
        free_gb = free / (1024**3)
        print(f"Free space: {free_gb:.2f} GB")
        
        if free_gb < 5:
            print(f"⚠ WARNING: Low disk space (< 5 GB)")
            print(f"  Model checkpoints may require several GB")
            return False
        else:
            print(f"✓ Sufficient disk space")
            return True
    except:
        print("⚠ Could not check disk space")
        return True

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "IMPROVED CLASSIFIER SETUP VERIFICATION" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    results = {
        'Files': check_files(),
        'Dependencies': check_imports(),
        'GPU': check_cuda(),
        'Dataset': check_dataset(),
        'Disk Space': check_disk_space()
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
        if not passed and check in ['Files', 'Dependencies', 'Dataset']:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nTo start training, run:")
        print("  python improved_training.py")
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print("\nCommon fixes:")
        print("  - Update DATA_DIR in improved_training.py if dataset is elsewhere")
        print("  - Install missing packages: pip install <package>")
    
    print("="*70 + "\n")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
