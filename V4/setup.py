#!/usr/bin/env python3
"""
Setup Script for ASIC-CNN Hybrid Benchmark

This script handles:
1. Dependency installation
2. Directory structure creation
3. Configuration validation
4. Optional dataset download

Run: python setup.py

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              ASIC-CNN HYBRID BENCHMARK - SETUP                           ║
║                                                                          ║
║    Automated setup for chest X-ray pathology detection benchmark         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


def check_python_version():
    """Ensure Python 3.8+."""
    print("\n[1/6] Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required packages."""
    print("\n[2/6] Installing dependencies...")
    
    packages = [
        'torch',
        'torchvision', 
        'numpy',
        'pillow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'requests'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package} (already installed)")
        except ImportError:
            print(f"  → Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package, '-q'
            ])
            print(f"  ✓ {package}")
    
    return True


def create_directories():
    """Create required directory structure."""
    print("\n[3/6] Creating directory structure...")
    
    base_dir = Path(__file__).parent
    
    directories = [
        base_dir / 'data' / 'train' / 'NORMAL',
        base_dir / 'data' / 'train' / 'PNEUMONIA',
        base_dir / 'data' / 'val' / 'NORMAL',
        base_dir / 'data' / 'val' / 'PNEUMONIA',
        base_dir / 'data' / 'test' / 'NORMAL',
        base_dir / 'data' / 'test' / 'PNEUMONIA',
        base_dir / 'results' / 'models',
        base_dir / 'results' / 'figures',
        base_dir / 'results' / 'metrics',
        base_dir / 'results' / 'attention_cache'
    ]
    
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"  ✓ Created {len(directories)} directories")
    return True


def check_gpu():
    """Check for GPU availability."""
    print("\n[4/6] Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ CUDA available: {device_name} ({memory:.1f} GB)")
            return True
        else:
            print("  ⚠ No CUDA GPU detected - will use CPU")
            print("    Training will be slower but functional")
            return True
            
    except Exception as e:
        print(f"  ⚠ Could not check GPU: {e}")
        return True


def check_asic_config():
    """Validate ASIC configuration."""
    print("\n[5/6] Checking ASIC configuration...")
    
    try:
        from config import ASIC_CONFIG
        
        if ASIC_CONFIG['enabled']:
            print(f"  → ASIC enabled: {ASIC_CONFIG['host']}:{ASIC_CONFIG['port']}")
            print("    Will attempt hardware connection during training")
        else:
            print("  → ASIC disabled - using software SHA-256")
            print("    Enable in config.py to use LV06 hardware")
        
        if ASIC_CONFIG['fallback_to_software']:
            print("  ✓ Software fallback enabled")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ Could not load config: {e}")
        return True


def setup_dataset():
    """Offer to download dataset."""
    print("\n[6/6] Dataset setup...")
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    
    # Check if data exists
    train_normal = data_dir / 'train' / 'NORMAL'
    has_data = train_normal.exists() and any(train_normal.glob('*'))
    
    if has_data:
        count = len(list(train_normal.glob('*')))
        print(f"  ✓ Dataset already exists ({count} images in train/NORMAL)")
        return True
    
    print("  ⚠ No dataset found")
    print("\n  Options:")
    print("    1. Download from Kaggle (requires kaggle API)")
    print("    2. Create synthetic dataset (for testing only)")
    print("    3. Skip (download manually later)")
    
    choice = input("\n  Enter choice [1/2/3]: ").strip()
    
    if choice == '1':
        print("\n  Attempting Kaggle download...")
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                'paultimothymooney/chest-xray-pneumonia',
                path=str(data_dir),
                unzip=True
            )
            print("  ✓ Dataset downloaded")
        except ImportError:
            print("  ✗ Kaggle package not installed")
            print("    Run: pip install kaggle")
            print("    Then configure ~/.kaggle/kaggle.json")
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
    
    elif choice == '2':
        print("\n  Creating synthetic dataset...")
        try:
            from dataset import create_synthetic_dataset
            create_synthetic_dataset(data_dir, num_samples=500)
            print("  ✓ Synthetic dataset created")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    else:
        print("\n  Skipped. Download manually from:")
        print("  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    
    return True


def print_next_steps():
    """Print instructions for next steps."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                           SETUP COMPLETE!                                ║
╚══════════════════════════════════════════════════════════════════════════╝

Next Steps:
───────────────────────────────────────────────────────────────────────────

1. VERIFY DATASET
   Ensure data/ contains train/val/test folders with NORMAL and PNEUMONIA

2. CONFIGURE ASIC (Optional)
   Edit config.py to set your LV06 IP address

3. TRAIN MODELS
   python main.py --mode train --models all

4. RUN BENCHMARK
   python main.py --mode benchmark

5. VIEW RESULTS
   Check results/metrics/benchmark_report.md

───────────────────────────────────────────────────────────────────────────

Quick Commands:
  python main.py --mode config      # View configuration
  python main.py --mode test-asic   # Test ASIC connection
  python main.py --mode full        # Run complete pipeline

For detailed instructions, see QUICKSTART.md

───────────────────────────────────────────────────────────────────────────
    """)


def main():
    """Run setup."""
    print_banner()
    
    steps = [
        ("Python version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Directories", create_directories),
        ("GPU check", check_gpu),
        ("ASIC config", check_asic_config),
        ("Dataset", setup_dataset)
    ]
    
    all_passed = True
    
    for name, func in steps:
        try:
            if not func():
                all_passed = False
                print(f"\n[ERROR] {name} check failed")
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            all_passed = False
    
    if all_passed:
        print_next_steps()
        return 0
    else:
        print("\n[WARNING] Setup completed with warnings")
        print("Review errors above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
