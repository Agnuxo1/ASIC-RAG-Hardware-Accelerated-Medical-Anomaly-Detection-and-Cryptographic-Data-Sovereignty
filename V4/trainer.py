"""
Training Module for ASIC-CNN Hybrid Benchmark

Handles training loop, validation, and model checkpointing.

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from tqdm import tqdm


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricTracker:
    """
    Tracks training metrics across epochs.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def update(self, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, lr: float):
        """Update metrics for current epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
    
    def get_best_epoch(self, metric: str = 'val_acc') -> int:
        """Get epoch with best metric value."""
        values = self.history.get(metric, [])
        if not values:
            return 0
        
        if 'loss' in metric:
            return int(np.argmin(values))
        return int(np.argmax(values))
    
    def save(self, path: Path):
        """Save history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: Path):
        """Load history from JSON."""
        with open(path, 'r') as f:
            self.history = json.load(f)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """
    Handles model training and validation.
    """
    
    def __init__(self, model: nn.Module, config: Dict,
                 attention_generator: Optional[object] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            attention_generator: Optional ASIC attention generator
            device: Device to train on
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.model = model
        self.config = config
        self.attention_generator = attention_generator
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-5)
        
        optimizer_name = config.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.get('scheduler_patience', 5),
            factor=config.get('scheduler_factor', 0.5)
        )
        
        # Early stopping
        self.early_stopping = None
        if config.get('early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=config.get('early_stopping_patience', 10),
                min_delta=config.get('early_stopping_min_delta', 0.001)
            )
        
        # Metrics
        self.metrics = MetricTracker()
        
        # Best model state
        self.best_model_state = None
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Get attention if available
            attention = batch.get('attention')
            if attention is not None:
                attention = attention.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Model-specific forward
            if hasattr(self.model, 'forward') and attention is not None:
                # Hybrid models
                if hasattr(self.model, 'attention_modules'):
                    # Multi-scale: create pyramid
                    attention_pyramid = self._create_attention_pyramid(attention)
                    outputs = self.model(images, attention_pyramid)
                else:
                    outputs = self.model(images, attention)
            else:
                # Standard CNN
                outputs = self.model(images)
            
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                attention = batch.get('attention')
                if attention is not None:
                    attention = attention.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and attention is not None:
                    if hasattr(self.model, 'attention_modules'):
                        attention_pyramid = self._create_attention_pyramid(attention)
                        outputs = self.model(images, attention_pyramid)
                    else:
                        outputs = self.model(images, attention)
                else:
                    outputs = self.model(images)
                
                logits = outputs['logits']
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_dir: Optional[Path] = None) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        print(f"\n{'=' * 60}")
        print(f"Training on {self.device}")
        print(f"{'=' * 60}\n")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.metrics.update(train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Print results
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f"  ✓ New best model (val_acc: {val_acc:.4f})")
                
                if save_dir:
                    torch.save(self.best_model_state, save_dir / 'best_model.pth')
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    print(f"\n[INFO] Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Periodic checkpoint
            if save_dir and (epoch + 1) % self.config.get('save_frequency', 5) == 0:
                torch.save(
                    self.model.state_dict(),
                    save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                )
        
        training_time = time.time() - start_time
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final metrics
        if save_dir:
            self.metrics.save(save_dir / 'training_history.json')
        
        print(f"\n{'=' * 60}")
        print(f"Training complete!")
        print(f"  Total time: {training_time / 60:.2f} minutes")
        print(f"  Best val accuracy: {self.best_val_acc:.4f}")
        print(f"{'=' * 60}")
        
        return self.metrics.history
    
    def _create_attention_pyramid(self, attention: torch.Tensor) -> List[torch.Tensor]:
        """
        Create attention pyramid from single attention map.
        
        ResNet-18 feature map sizes: 56x56, 28x28, 14x14, 7x7
        """
        import torch.nn.functional as F
        
        pyramid = []
        sizes = [(56, 56), (28, 28), (14, 14), (7, 7)]
        
        for h, w in sizes:
            resized = F.interpolate(
                attention.unsqueeze(1) if attention.dim() == 3 else attention,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            pyramid.append(resized)
        
        return pyramid


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model_name: str, model: nn.Module, 
                train_loader: DataLoader, val_loader: DataLoader,
                config: Dict, save_dir: Path,
                attention_generator: Optional[object] = None) -> Dict:
    """
    Train a single model.
    
    Args:
        model_name: Name of the model (for logging)
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        save_dir: Directory to save results
        attention_generator: Optional attention generator
        
    Returns:
        Training history
    """
    print(f"\n{'#' * 60}")
    print(f"Training: {model_name}")
    print(f"{'#' * 60}")
    
    # Create model-specific save directory
    model_save_dir = save_dir / model_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        attention_generator=attention_generator
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 50),
        save_dir=model_save_dir
    )
    
    return history


# =============================================================================
# TEST
# =============================================================================

def test_trainer():
    """Test training module with dummy data."""
    if not HAS_TORCH:
        print("[ERROR] PyTorch required for testing")
        return
    
    print("\n" + "=" * 60)
    print("Testing Trainer Module")
    print("=" * 60)
    
    # Import model
    from models import StandardCNN
    
    # Create dummy model
    model = StandardCNN(num_classes=2)
    
    # Dummy config
    config = {
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'early_stopping': False,
        'num_epochs': 2
    }
    
    # Create trainer
    trainer = Trainer(model, config)
    
    print(f"Trainer created successfully on {trainer.device}")
    print("=" * 60)


if __name__ == "__main__":
    test_trainer()
