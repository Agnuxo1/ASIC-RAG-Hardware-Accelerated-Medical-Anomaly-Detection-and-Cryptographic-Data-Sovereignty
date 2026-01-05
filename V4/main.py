#!/usr/bin/env python3
"""
ASIC-CNN Hybrid Benchmark: Main Entry Point

Complete system for comparing standard CNN vs ASIC-enhanced hybrid models
for chest X-ray pathology detection.

Usage:
    # Create synthetic dataset (for testing)
    python main.py --mode dataset --synthetic
    
    # Download real dataset from Kaggle
    python main.py --mode dataset --kaggle
    
    # Train all models
    python main.py --mode train --models all
    
    # Train specific model
    python main.py --mode train --models hybrid_single
    
    # Run benchmark on test set
    python main.py --mode benchmark
    
    # Generate visualizations
    python main.py --mode visualize
    
    # Full pipeline
    python main.py --mode full

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
Date: December 2024
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Import configuration
from config import (
    BASE_DIR, DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    DATASET_CONFIG, ASIC_CONFIG, MODEL_CONFIG, TRAINING_CONFIG,
    AUGMENTATION_CONFIG, BENCHMARK_CONFIG, get_device, print_config
)


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def prepare_dataset(args) -> bool:
    """Prepare dataset for training."""
    from dataset import (
        download_kaggle_dataset,
        download_huggingface_dataset,
        create_synthetic_dataset
    )
    
    print("\n" + "=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)
    
    if args.synthetic:
        print("\n[INFO] Creating synthetic dataset for testing...")
        return create_synthetic_dataset(DATA_DIR, num_samples=args.samples)
    
    elif hasattr(args, 'huggingface') and args.huggingface:
        print("\n[INFO] Downloading from Hugging Face...")
        # Check if dataset already exists to avoid re-download
        if (DATA_DIR / 'train' / 'NORMAL').exists():
            print("[INFO] Dataset directory exists. Skipping download (delete 'data' folder to force refresh).")
            return True
        return download_huggingface_dataset(DATA_DIR)

    elif args.kaggle:
        print("\n[INFO] Downloading from Kaggle...")
        success = download_kaggle_dataset(
            DATASET_CONFIG['kaggle_dataset'],
            DATA_DIR
        )
        if not success:
            print("[INFO] Kaggle download failed. Try --huggingface or --synthetic")
            return False
        return success
    
    else:
        # Check if data exists
        if (DATA_DIR / 'train' / 'NORMAL').exists():
            print("[INFO] Dataset already exists")
            return True
        else:
            print("[WARNING] No dataset found. Use --huggingface, --synthetic or --kaggle")
            return False


def train_models(args):
    """Train selected models."""
    import torch
    from dataset import ChestXRayDataset, get_transforms, create_data_loaders
    from models import create_model
    from trainer import train_model
    from asic_interface import create_attention_generator
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    # Check dataset
    if not (DATA_DIR / 'train').exists():
        print("[ERROR] Training data not found. Run with --mode dataset first.")
        return False
    
    # Setup device
    device = get_device()
    print(f"\n[INFO] Using device: {device}")
    
    # Create attention generator
    attention_generator = None
    if ASIC_CONFIG['enabled'] or ASIC_CONFIG['fallback_to_software']:
        attention_generator = create_attention_generator(ASIC_CONFIG)
    
    # Get transforms
    transforms_dict = get_transforms(AUGMENTATION_CONFIG)
    
    # Determine which models to train
    if args.models == 'all':
        models_to_train = list(MODEL_CONFIG['models'].keys())
    else:
        models_to_train = [args.models]
    
    print(f"\n[INFO] Models to train: {models_to_train}")
    
    # Training loop for each model
    for model_name in models_to_train:
        print(f"\n{'#' * 70}")
        print(f"Training: {model_name}")
        print(f"{'#' * 70}")
        
        model_cfg = MODEL_CONFIG['models'].get(model_name)
        if not model_cfg:
            print(f"[WARNING] Unknown model: {model_name}, skipping...")
            continue
        
        # Create model
        model = create_model(
            model_name, 
            backbone_name=MODEL_CONFIG['backbone'],
            num_classes=DATASET_CONFIG['num_classes']
        )
        
        # Determine if this model uses attention
        use_attention = model_cfg.get('use_asic_attention', False)
        
        # Create datasets with or without attention
        train_dataset = ChestXRayDataset(
            DATA_DIR, split='train',
            transform=transforms_dict['train'],
            attention_generator=attention_generator if use_attention else None,
            return_attention=use_attention
        )
        
        val_dataset = ChestXRayDataset(
            DATA_DIR, split='val',
            transform=transforms_dict['val'],
            attention_generator=attention_generator if use_attention else None,
            return_attention=use_attention
        )
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Train model
        history = train_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TRAINING_CONFIG,
            save_dir=MODELS_DIR,
            attention_generator=attention_generator if use_attention else None
        )
    
    print("\n[INFO] Training complete!")
    return True


def run_benchmark(args):
    """Run benchmark evaluation."""
    import torch
    from dataset import ChestXRayDataset, get_transforms
    from benchmark import BenchmarkRunner, plot_benchmark_results
    from asic_interface import create_attention_generator
    
    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION")
    print("=" * 70)
    
    # Check test data
    if not (DATA_DIR / 'test').exists():
        print("[ERROR] Test data not found. Run with --mode dataset first.")
        return False
    
    # Create attention generator
    attention_generator = None
    if ASIC_CONFIG['enabled'] or ASIC_CONFIG['fallback_to_software']:
        attention_generator = create_attention_generator(ASIC_CONFIG)
    
    # Get transforms
    transforms_dict = get_transforms(AUGMENTATION_CONFIG)
    
    # Create test dataset (with attention for hybrid models)
    test_dataset = ChestXRayDataset(
        DATA_DIR, split='test',
        transform=transforms_dict['test'],
        attention_generator=attention_generator,
        return_attention=True
    )
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Run benchmark
    runner = BenchmarkRunner(MODELS_DIR, BENCHMARK_CONFIG)
    results = runner.run_benchmark(
        test_loader=test_loader,
        model_configs=MODEL_CONFIG['models'],
        attention_generator=attention_generator
    )
    
    # Generate report
    report_path = runner.generate_report(METRICS_DIR)
    
    # Generate visualizations
    plot_benchmark_results(results, FIGURES_DIR)
    
    print("\n[INFO] Benchmark complete!")
    print(f"[INFO] Report: {report_path}")
    print(f"[INFO] Figures: {FIGURES_DIR}")
    
    return True


def visualize_results(args):
    """Generate visualizations from existing results."""
    import json
    from benchmark import plot_benchmark_results
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Load results
    results_file = METRICS_DIR / 'benchmark_results.json'
    
    if not results_file.exists():
        print("[ERROR] No benchmark results found. Run --mode benchmark first.")
        return False
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate plots
    plot_benchmark_results(results, FIGURES_DIR)
    
    print(f"\n[INFO] Visualizations saved to {FIGURES_DIR}")
    return True


def run_full_pipeline(args):
    """Run complete pipeline: dataset → train → benchmark → visualize."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE EXECUTION")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Prepare dataset
    print("\n[STEP 1/4] Preparing dataset...")
    if not (DATA_DIR / 'train').exists():
        # Default to Hugging Face if no specific flag, else Synthetic
        if not args.synthetic and not args.kaggle:
            args.huggingface = True
        
        # If still no source selected (shouldn't happen with above logic but safe), use synthetic
        if not (args.huggingface or args.kaggle):
            args.synthetic = True
            
        if not prepare_dataset(args):
            return False
    else:
        print("[INFO] Dataset already exists, skipping...")
    
    # 2. Train models
    print("\n[STEP 2/4] Training models...")
    args.models = 'all'
    if not train_models(args):
        return False
    
    # 3. Run benchmark
    print("\n[STEP 3/4] Running benchmark...")
    if not run_benchmark(args):
        return False
    
    # 4. Generate visualizations
    print("\n[STEP 4/4] Generating visualizations...")
    if not visualize_results(args):
        return False
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {elapsed / 60:.2f} minutes")
    print(f"\nResults saved to:")
    print(f"  - Models:   {MODELS_DIR}")
    print(f"  - Metrics:  {METRICS_DIR}")
    print(f"  - Figures:  {FIGURES_DIR}")
    
    return True


def test_asic_connection(args):
    """Test connection to LV06 ASIC."""
    from asic_interface import LV06Interface, test_attention_generation
    
    print("\n" + "=" * 70)
    print("ASIC CONNECTION TEST")
    print("=" * 70)
    
    if not ASIC_CONFIG['enabled']:
        print("\n[INFO] ASIC is disabled in config. Testing software fallback...")
        test_attention_generation()
        return True
    
    print(f"\n[INFO] Testing connection to {ASIC_CONFIG['host']}...")
    
    asic = LV06Interface(
        host=ASIC_CONFIG['host'],
        port=ASIC_CONFIG['port'],
        stratum_port=ASIC_CONFIG['stratum_port'],
        timeout=ASIC_CONFIG['timeout']
    )
    
    if asic.connect():
        print("[SUCCESS] Connected to LV06!")
        
        stats = asic.get_stats()
        if stats:
            print(f"\nDevice Statistics:")
            print(f"  Hash Rate:  {stats.hash_rate:.2f} GH/s")
            print(f"  Temperature: {stats.temperature}°C")
            print(f"  Power:      {stats.power}W")
            print(f"  Uptime:     {stats.uptime}s")
    else:
        print(f"[FAILED] Could not connect: {asic.last_error}")
        print("\n[INFO] Testing with software fallback...")
    
    # Test attention generation
    test_attention_generation()
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='ASIC-CNN Hybrid Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode dataset --synthetic    Create synthetic test dataset
  python main.py --mode train --models all     Train all models
  python main.py --mode benchmark              Run benchmark evaluation
  python main.py --mode full                   Run complete pipeline
  python main.py --mode test-asic              Test ASIC connection
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['dataset', 'train', 'benchmark', 'visualize', 
                                'full', 'test-asic', 'config'],
                        help='Operation mode')
    
    # Dataset options
    parser.add_argument('--synthetic', action='store_true',
                        help='Create synthetic dataset')
    parser.add_argument('--kaggle', action='store_true',
                        help='Download from Kaggle')
    parser.add_argument('--huggingface', action='store_true',
                        help='Download from Hugging Face')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of samples for dataset')
    
    # Training options
    parser.add_argument('--models', type=str, default='all',
                        choices=['all', 'standard_cnn', 'hybrid_single', 'hybrid_multi'],
                        help='Models to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    
    # ASIC options
    parser.add_argument('--asic-host', type=str, default=None,
                        help='Override ASIC host IP')
    parser.add_argument('--no-asic', action='store_true',
                        help='Disable ASIC, use software only')
    parser.add_argument('--enable-asic', action='store_true',
                        help='Enable ASIC (overrides config)')
    
    args = parser.parse_args()
    
    # Apply overrides
    if args.epochs:
        TRAINING_CONFIG['num_epochs'] = args.epochs
    if args.batch_size:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    if args.asic_host:
        ASIC_CONFIG['host'] = args.asic_host
    if args.no_asic:
        ASIC_CONFIG['enabled'] = False
    if args.enable_asic:
        ASIC_CONFIG['enabled'] = True
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     █████╗ ███████╗██╗ ██████╗      ██████╗███╗   ██╗███╗   ██╗         ║
║    ██╔══██╗██╔════╝██║██╔════╝     ██╔════╝████╗  ██║████╗  ██║         ║
║    ███████║███████╗██║██║          ██║     ██╔██╗ ██║██╔██╗ ██║         ║
║    ██╔══██║╚════██║██║██║          ██║     ██║╚██╗██║██║╚██╗██║         ║
║    ██║  ██║███████║██║╚██████╗     ╚██████╗██║ ╚████║██║ ╚████║         ║
║    ╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝      ╚═════╝╚═╝  ╚═══╝╚═╝  ╚═══╝         ║
║                                                                          ║
║           HYBRID BENCHMARK: Standard CNN vs ASIC-Enhanced Models         ║
║                                                                          ║
║    Author: Francisco Angulo de Lafuente                                  ║
║    GitHub: https://github.com/Agnuxo1                                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Set random seed
    setup_seed(TRAINING_CONFIG.get('seed', 42))
    
    # Execute mode
    if args.mode == 'config':
        print_config()
        return 0
    
    elif args.mode == 'dataset':
        success = prepare_dataset(args)
        return 0 if success else 1
    
    elif args.mode == 'train':
        success = train_models(args)
        return 0 if success else 1
    
    elif args.mode == 'benchmark':
        success = run_benchmark(args)
        return 0 if success else 1
    
    elif args.mode == 'visualize':
        success = visualize_results(args)
        return 0 if success else 1
    
    elif args.mode == 'full':
        success = run_full_pipeline(args)
        return 0 if success else 1
    
    elif args.mode == 'test-asic':
        success = test_asic_connection(args)
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
