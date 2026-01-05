"""
Dataset Module for ASIC-CNN Hybrid Benchmark

Handles downloading and loading of chest X-ray datasets.

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
"""

import os
import shutil
import zipfile
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# DATASET DOWNLOAD
# =============================================================================

def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> bool:
    """
    Download dataset from Kaggle.
    
    Requires kaggle API credentials (~/.kaggle/kaggle.json)
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'paultimothymooney/chest-xray-pneumonia')
        output_dir: Directory to save dataset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import kaggle
        
        print(f"[INFO] Downloading {dataset_name} from Kaggle...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True
        )
        
        print(f"[INFO] Dataset downloaded to {output_dir}")
        return True
        
    except ImportError:
        print("[WARNING] kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        return False


def download_huggingface_dataset(output_dir: Path) -> bool:
    """
    Download dataset from Hugging Face (keremberke/chest-xray-pneumonia).
    
    Args:
        output_dir: Directory to save dataset
        
    Returns:
        True if successful
    """
    try:
        from datasets import load_dataset
        import numpy as np
        
        print("[INFO] Downloading 'mmenendezg/pneumonia_x_ray' from Hugging Face...")
        
        # Load dataset
        dataset = load_dataset("mmenendezg/pneumonia_x_ray", name="default")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map HF splits to local folders
        # HF structure: train, validation, test
        # Local structure: train/NORMAL, train/PNEUMONIA, etc.
        
        # Labels: 0 = NORMAL, 1 = PNEUMONIA
        label_map = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        for split in ['train', 'validation', 'test']:
            hf_split = dataset[split]
            local_split = 'val' if split == 'validation' else split
            
            print(f"[INFO] Processing {split} split ({len(hf_split)} images)...")
            
            for i, item in enumerate(hf_split):
                image = item['image']
                label = item['label']
                
                class_name = label_map[label]
                
                # Create directory
                target_dir = output_dir / local_split / class_name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Save image
                # Convert to grayscale if needed to match pipeline expectation
                if image.mode != 'L':
                    image = image.convert('L')
                
                image_path = target_dir / f"hf_{split}_{i:05d}.png"
                image.save(image_path)
                
        print(f"[INFO] Hugging Face dataset downloaded to {output_dir}")
        return True
        
    except ImportError:
        print("[ERROR] 'datasets' library required. Install with: pip install datasets huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download from Hugging Face: {e}")
        return False


        return False


def create_synthetic_dataset(output_dir: Path, num_samples: int = 500) -> bool:
    """
    Create synthetic chest X-ray-like images for testing.
    
    These are NOT real medical images - only for code validation.
    
    Args:
        output_dir: Directory to save dataset
        num_samples: Total number of samples to create
        
    Returns:
        True if successful
    """
    if not HAS_PIL:
        print("[ERROR] PIL required for synthetic dataset. Install with: pip install pillow")
        return False
    
    print(f"[INFO] Creating synthetic dataset ({num_samples} images)...")
    print("[WARNING] These are NOT real medical images - for code testing only!")
    
    np.random.seed(42)
    
    # Create directory structure
    splits = {
        'train': 0.8,
        'val': 0.1,
        'test': 0.1
    }
    
    samples_per_class = num_samples // 2
    
    for split_name, split_ratio in splits.items():
        n_split = int(samples_per_class * split_ratio)



def create_synthetic_dataset(output_dir: Path, num_samples: int = 500) -> bool:
    """
    Create synthetic chest X-ray-like images for testing.
    
    These are NOT real medical images - only for code validation.
    
    Args:
        output_dir: Directory to save dataset
        num_samples: Total number of samples to create
        
    Returns:
        True if successful
    """
    if not HAS_PIL:
        print("[ERROR] PIL required for synthetic dataset. Install with: pip install pillow")
        return False
    
    print(f"[INFO] Creating synthetic dataset ({num_samples} images)...")
    print("[WARNING] These are NOT real medical images - for code testing only!")
    
    np.random.seed(42)
    
    # Create directory structure
    splits = {
        'train': 0.8,
        'val': 0.1,
        'test': 0.1
    }
    
    samples_per_class = num_samples // 2
    
    for split_name, split_ratio in splits.items():
        n_split = int(samples_per_class * split_ratio)
        
        for cls in ['NORMAL', 'PNEUMONIA']:
            cls_dir = output_dir / split_name / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(n_split):
                # Generate synthetic image
                if cls == 'NORMAL':
                    img = _generate_synthetic_normal(seed=hash(f"{split_name}_{cls}_{i}") % 2**32)
                else:
                    img = _generate_synthetic_pneumonia(seed=hash(f"{split_name}_{cls}_{i}") % 2**32)
                
                # Save
                img_path = cls_dir / f"{cls.lower()}_{i:04d}.png"
                img.save(img_path)
    
    print(f"[INFO] Synthetic dataset created at {output_dir}")
    return True


def _generate_synthetic_normal(size: Tuple[int, int] = (224, 224), seed: int = 42) -> Image.Image:
    """Generate synthetic 'normal' chest X-ray."""
    np.random.seed(seed)
    
    # Create base gradient (darker lungs, lighter center)
    y, x = np.mgrid[0:size[0], 0:size[1]]
    center_x, center_y = size[1] // 2, size[0] // 2
    
    # Lung fields (darker)
    dist = np.sqrt((x - center_x)**2 / 3000 + (y - center_y)**2 / 5000)
    base = 70 + 40 * np.exp(-dist)
    
    # Add ribs (lighter horizontal bands)
    for i in range(6):
        rib_y = 30 + i * 30
        rib_mask = np.abs(y - rib_y) < 5
        base[rib_mask] += 20
    
    # Add noise
    noise = np.random.normal(0, 8, size)
    img_array = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array, mode='L')


def _generate_synthetic_pneumonia(size: Tuple[int, int] = (224, 224), seed: int = 42) -> Image.Image:
    """Generate synthetic 'pneumonia' chest X-ray with opacity."""
    # Start with normal
    img = _generate_synthetic_normal(size, seed)
    img_array = np.array(img).astype(np.float32)
    
    np.random.seed(seed + 1000)
    
    # Add opacity region (brighter patch)
    opacity_x = np.random.randint(size[1] // 4, 3 * size[1] // 4)
    opacity_y = np.random.randint(size[0] // 3, 2 * size[0] // 3)
    opacity_size = np.random.randint(30, 60)
    
    y, x = np.mgrid[0:size[0], 0:size[1]]
    dist = np.sqrt((x - opacity_x)**2 + (y - opacity_y)**2)
    
    opacity_mask = dist < opacity_size
    opacity_intensity = 40 + np.random.randint(-10, 10)
    
    # Gaussian falloff
    opacity_addition = opacity_intensity * np.exp(-dist**2 / (2 * opacity_size**2))
    img_array += opacity_addition
    
    # Maybe add a second smaller opacity
    if np.random.random() > 0.5:
        opacity_x2 = np.random.randint(size[1] // 4, 3 * size[1] // 4)
        opacity_y2 = np.random.randint(size[0] // 3, 2 * size[0] // 3)
        opacity_size2 = np.random.randint(15, 35)
        dist2 = np.sqrt((x - opacity_x2)**2 + (y - opacity_y2)**2)
        opacity_addition2 = 30 * np.exp(-dist2**2 / (2 * opacity_size2**2))
        img_array += opacity_addition2
    
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array, mode='L')


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images.
    
    Supports:
    - Normal (label 0) and Pneumonia (label 1) classification
    - Image augmentation
    - Optional ASIC attention generation
    """
    
    def __init__(self, root_dir: Path, split: str = 'train',
                 transform: Optional[Callable] = None,
                 attention_generator: Optional[object] = None,
                 return_attention: bool = False):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing train/val/test folders
            split: One of 'train', 'val', 'test'
            transform: Optional torchvision transforms
            attention_generator: Optional ASICAttentionGenerator
            return_attention: Whether to return attention maps
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.attention_generator = attention_generator
        self.return_attention = return_attention
        
        # Find all images
        self.samples = []
        self.classes = ['NORMAL', 'PNEUMONIA']
        
        split_dir = self.root_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                print(f"[WARNING] Class directory not found: {class_dir}")
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.samples.append((img_path, class_idx))
        
        print(f"[INFO] Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get sample by index.
        
        Returns:
            Dictionary with 'image', 'label', and optionally 'attention'
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Generate attention if needed
        attention = None
        if self.return_attention and self.attention_generator:
            img_array = np.array(image)
            # Use hardware if generator has an ASIC interface connected
            use_hw = getattr(self.attention_generator, 'asic', None) is not None
            attention = self.attention_generator.generate_multiscale_attention(
                img_array, scales=[4, 8, 16], use_asic=use_hw
            )
            attention = torch.from_numpy(attention).float().unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'label': label,
            'path': str(img_path)
        }
        
        if attention is not None:
            result['attention'] = attention
        
        return result


# =============================================================================
# DATA TRANSFORMS
# =============================================================================

def get_transforms(config: Dict) -> Dict[str, Callable]:
    """
    Get transforms for each split.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Dictionary of transforms for train/val/test
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required. Install with: pip install torch torchvision")
    
    train_config = config.get('train', {})
    val_config = config.get('val', {})
    test_config = config.get('test', val_config)
    
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(train_config.get('resize', 256)),
        transforms.RandomCrop(train_config.get('crop_size', 224)),
        transforms.RandomHorizontalFlip(p=train_config.get('horizontal_flip', 0.5)),
        transforms.RandomRotation(degrees=train_config.get('rotation', 10)),
        transforms.ColorJitter(
            brightness=train_config.get('brightness', 0.1),
            contrast=train_config.get('contrast', 0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=train_config.get('normalize_mean', [0.485]),
            std=train_config.get('normalize_std', [0.229])
        )
    ])
    
    # Validation/test transforms (no augmentation)
    eval_transforms = transforms.Compose([
        transforms.Resize(val_config.get('resize', 256)),
        transforms.CenterCrop(val_config.get('crop_size', 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=val_config.get('normalize_mean', [0.485]),
            std=val_config.get('normalize_std', [0.229])
        )
    ])
    
    return {
        'train': train_transforms,
        'val': eval_transforms,
        'test': eval_transforms
    }


# =============================================================================
# DATA LOADERS
# =============================================================================

def create_data_loaders(data_dir: Path, config: Dict,
                        attention_generator: Optional[object] = None) -> Dict[str, DataLoader]:
    """
    Create data loaders for all splits.
    
    Args:
        data_dir: Root data directory
        config: Configuration dictionary
        attention_generator: Optional attention generator
        
    Returns:
        Dictionary of DataLoaders
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")
    
    # Get transforms
    transforms_dict = get_transforms(config.get('augmentation', {}))
    
    # Training config
    batch_size = config.get('training', {}).get('batch_size', 16)
    num_workers = config.get('training', {}).get('num_workers', 4)
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = ChestXRayDataset(
                root_dir=data_dir,
                split=split,
                transform=transforms_dict[split],
                attention_generator=attention_generator,
                return_attention=(attention_generator is not None)
            )
            
            shuffle = (split == 'train')
            
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
        except Exception as e:
            print(f"[WARNING] Could not create {split} loader: {e}")
    
    return loaders


# =============================================================================
# MAIN / TEST
# =============================================================================

def main():
    """Main function for dataset operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset operations')
    parser.add_argument('--action', type=str, default='synthetic',
                        choices=['download', 'synthetic', 'info'],
                        help='Action to perform')
    parser.add_argument('--output', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of samples')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.action == 'download':
        print("[INFO] Attempting Kaggle download...")
        success = download_kaggle_dataset(
            'paultimothymooney/chest-xray-pneumonia',
            output_dir
        )
        if not success:
            download_alternative_dataset(output_dir, args.samples)
    
    elif args.action == 'synthetic':
        create_synthetic_dataset(output_dir, args.samples)
    
    elif args.action == 'info':
        print(f"\n[INFO] Dataset directory: {output_dir}")
        
        if output_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = output_dir / split
                if split_dir.exists():
                    for cls in ['NORMAL', 'PNEUMONIA']:
                        cls_dir = split_dir / cls
                        if cls_dir.exists():
                            count = len(list(cls_dir.glob('*')))
                            print(f"  {split}/{cls}: {count} images")
        else:
            print("[WARNING] Dataset directory does not exist")


if __name__ == "__main__":
    main()
