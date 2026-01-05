"""
Benchmark Module for ASIC-CNN Hybrid Evaluation

Comprehensive evaluation and comparison of trained models.

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, roc_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from tqdm import tqdm


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities for positive class (for AUC)
        
    Returns:
        Dictionary of metrics
    """
    if not HAS_SKLEARN:
        # Basic metrics without sklearn
        accuracy = np.mean(y_true == y_pred)
        return {
            'accuracy': accuracy,
            'note': 'Install sklearn for full metrics'
        }
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
    
    # AUC-ROC if probabilities available
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception as e:
            metrics['auc_roc'] = None
            metrics['auc_error'] = str(e)
    
    return metrics


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    """
    Evaluates trained models on test data.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            device: Device for evaluation
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader,
                 use_attention: bool = False,
                 attention_generator: Optional[object] = None) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            use_attention: Whether to use ASIC attention
            attention_generator: Attention generator (for hybrid models)
            
        Returns:
            Dictionary of metrics
        """
        all_labels = []
        all_preds = []
        all_probs = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                labels = batch['label'].numpy()
                
                attention = batch.get('attention')
                if attention is not None:
                    attention = attention.to(self.device)
                
                # Measure inference time
                start_time = time.perf_counter()
                
                # Forward pass
                if use_attention and attention is not None:
                    if hasattr(self.model, 'attention_modules'):
                        attention_pyramid = self._create_attention_pyramid(attention)
                        outputs = self.model(images, attention_pyramid)
                    else:
                        outputs = self.model(images, attention)
                else:
                    outputs = self.model(images)
                
                elapsed = (time.perf_counter() - start_time) * 1000  # ms
                inference_times.append(elapsed / images.size(0))  # per image
                
                logits = outputs['logits']
                probs = outputs['probabilities']
                
                _, predicted = logits.max(1)
                
                all_labels.extend(labels.tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())
        
        # Compute metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Add timing metrics
        metrics['inference_time_ms'] = {
            'mean': np.mean(inference_times),
            'std': np.std(inference_times),
            'min': np.min(inference_times),
            'max': np.max(inference_times)
        }
        
        return metrics
    
    def _create_attention_pyramid(self, attention: torch.Tensor) -> List[torch.Tensor]:
        """Create attention pyramid from single attention map."""
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
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """
    Runs comprehensive benchmark comparing multiple models.
    """
    
    def __init__(self, models_dir: Path, config: Dict):
        """
        Initialize benchmark runner.
        
        Args:
            models_dir: Directory containing trained model weights
            config: Benchmark configuration
        """
        self.models_dir = Path(models_dir)
        self.config = config
        self.results = {}
    
    def run_benchmark(self, test_loader: DataLoader,
                      model_configs: Dict,
                      attention_generator: Optional[object] = None) -> Dict:
        """
        Run benchmark on all models.
        
        Args:
            test_loader: Test data loader
            model_configs: Dictionary of model configurations
            attention_generator: Optional attention generator
            
        Returns:
            Benchmark results
        """
        from models import create_model
        
        print("\n" + "=" * 70)
        print("ASIC-CNN HYBRID BENCHMARK")
        print("=" * 70)
        
        results = {}
        
        for model_name, model_cfg in model_configs.items():
            print(f"\n{'─' * 60}")
            print(f"Evaluating: {model_name}")
            print(f"Description: {model_cfg.get('description', 'N/A')}")
            print(f"{'─' * 60}")
            
            # Create model
            from config import MODEL_CONFIG
            model = create_model(
                model_name, 
                backbone_name=MODEL_CONFIG.get('backbone', 'resnet50'),
                num_classes=2
            )
            
            # Load weights if available
            weights_path = self.models_dir / model_name / 'best_model.pth'
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                print(f"  Loaded weights from {weights_path}")
            else:
                print(f"  [WARNING] No weights found, using random initialization")
            
            # Create evaluator
            evaluator = ModelEvaluator(model)
            
            # Evaluate
            use_attention = model_cfg.get('use_asic_attention', False)
            metrics = evaluator.evaluate(
                test_loader,
                use_attention=use_attention,
                attention_generator=attention_generator
            )
            
            results[model_name] = metrics
            
            # Print results
            self._print_metrics(model_name, metrics)
        
        self.results = results
        return results
    
    def _print_metrics(self, model_name: str, metrics: Dict):
        """Print metrics for a model."""
        print(f"\n  Results for {model_name}:")
        print(f"    Accuracy:    {metrics.get('accuracy', 0):.4f}")
        print(f"    Sensitivity: {metrics.get('sensitivity', 0):.4f}")
        print(f"    Specificity: {metrics.get('specificity', 0):.4f}")
        print(f"    Precision:   {metrics.get('precision', 0):.4f}")
        print(f"    F1 Score:    {metrics.get('f1_score', 0):.4f}")
        
        if metrics.get('auc_roc'):
            print(f"    AUC-ROC:     {metrics['auc_roc']:.4f}")
        
        timing = metrics.get('inference_time_ms', {})
        if timing:
            print(f"    Inference:   {timing.get('mean', 0):.2f} ± {timing.get('std', 0):.2f} ms")
    
    def generate_report(self, output_dir: Path) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_dir: Directory to save report
            
        Returns:
            Path to report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results as JSON
        json_path = output_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            results_json = {}
            for model_name, metrics in self.results.items():
                results_json[model_name] = {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in metrics.items()
                }
            json.dump(results_json, f, indent=2)
        
        # Generate markdown report
        report_path = output_dir / 'benchmark_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# ASIC-CNN Hybrid Benchmark Results\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Summary Comparison\n\n")
            f.write("| Model | Accuracy | Sensitivity | Specificity | F1 Score | AUC-ROC | Inference (ms) |\n")
            f.write("|-------|----------|-------------|-------------|----------|---------|----------------|\n")
            
            for model_name, metrics in self.results.items():
                timing = metrics.get('inference_time_ms', {})
                auc = metrics.get('auc_roc')
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                
                f.write(f"| {model_name} | "
                       f"{metrics.get('accuracy', 0):.4f} | "
                       f"{metrics.get('sensitivity', 0):.4f} | "
                       f"{metrics.get('specificity', 0):.4f} | "
                       f"{metrics.get('f1_score', 0):.4f} | "
                       f"{auc_str} | "
                       f"{timing.get('mean', 0):.2f} |\n")
            
            # Detailed results for each model
            f.write("\n## Detailed Results\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"### {model_name}\n\n")
                
                # Confusion matrix
                cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                f.write("**Confusion Matrix:**\n\n")
                f.write("| | Predicted Normal | Predicted Pneumonia |\n")
                f.write("|---|---|---|\n")
                f.write(f"| **True Normal** | {cm[0][0]} | {cm[0][1]} |\n")
                f.write(f"| **True Pneumonia** | {cm[1][0]} | {cm[1][1]} |\n\n")
                
                # Metrics
                f.write("**Metrics:**\n")
                f.write(f"- True Positives: {metrics.get('true_positives', 'N/A')}\n")
                f.write(f"- True Negatives: {metrics.get('true_negatives', 'N/A')}\n")
                f.write(f"- False Positives: {metrics.get('false_positives', 'N/A')}\n")
                f.write(f"- False Negatives: {metrics.get('false_negatives', 'N/A')}\n\n")
            
            # Conclusions
            f.write("## Analysis\n\n")
            
            # Find best model by different metrics
            best_accuracy = max(self.results.items(), key=lambda x: x[1].get('accuracy', 0))
            best_sensitivity = max(self.results.items(), key=lambda x: x[1].get('sensitivity', 0))
            best_f1 = max(self.results.items(), key=lambda x: x[1].get('f1_score', 0))
            
            f.write(f"- **Best Accuracy:** {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})\n")
            f.write(f"- **Best Sensitivity:** {best_sensitivity[0]} ({best_sensitivity[1]['sensitivity']:.4f})\n")
            f.write(f"- **Best F1 Score:** {best_f1[0]} ({best_f1[1]['f1_score']:.4f})\n\n")
            
            f.write("### Observations\n\n")
            f.write("*(To be filled based on actual results)*\n\n")
            
            f.write("---\n")
            f.write("*Report generated by ASIC-CNN Hybrid Benchmark*\n")
        
        print(f"\n[INFO] Report saved to {report_path}")
        return str(report_path)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_benchmark_results(results: Dict, output_dir: Path):
    """
    Generate visualization of benchmark results.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARNING] matplotlib/seaborn not available for visualization")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'f1_score']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=model_name, color=colors[i % len(colors)])
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Key Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for i, v in enumerate(ax.patches):
        ax.text(v.get_x() + v.get_width() / 2, v.get_height() + 0.02,
                f'{v.get_height():.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150)
    plt.close()
    
    # 2. ROC curves (if available)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for model_name, metrics in results.items():
        roc = metrics.get('roc_curve')
        if roc:
            ax.plot(roc['fpr'], roc['tpr'],
                   label=f"{model_name} (AUC={metrics.get('auc_roc', 0):.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150)
    plt.close()
    
    # 3. Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    for ax, (model_name, metrics) in zip(axes, results.items()):
        cm = np.array(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150)
    plt.close()
    
    # 4. Inference time comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    
    model_names = list(results.keys())
    times = [results[m].get('inference_time_ms', {}).get('mean', 0) for m in model_names]
    stds = [results[m].get('inference_time_ms', {}).get('std', 0) for m in model_names]
    
    ax.bar(model_names, times, yerr=stds, capsize=5, color=colors[:len(model_names)])
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time Comparison')
    
    for i, (t, s) in enumerate(zip(times, stds)):
        ax.text(i, t + s + 0.5, f'{t:.2f}ms', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time.png', dpi=150)
    plt.close()
    
    print(f"[INFO] Visualizations saved to {output_dir}")


# =============================================================================
# TEST
# =============================================================================

def test_benchmark():
    """Test benchmark module."""
    print("\n" + "=" * 60)
    print("Testing Benchmark Module")
    print("=" * 60)
    
    # Test metrics computation
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.4, 0.7, 0.1])
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    print("\nTest metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {key}: [dict]")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_benchmark()
