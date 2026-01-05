# ASIC-CNN Hybrid System for Medical Image Pathology Detection

## Comparative Benchmark: Standard CNN vs ASIC-Enhanced Hybrid Model

**Author:** Francisco Angulo de Lafuente  
**Project:** ASIC-RAG-CHIMERA Medical Extension  
**Version:** 1.0  
**Date:** December 2024  
**GitHub:** https://github.com/Agnuxo1

---

## Executive Summary

This project implements and benchmarks a novel **hybrid pathology detection system** that combines:

1. **Standard CNN** (Convolutional Neural Network) - Traditional deep learning approach
2. **ASIC-Enhanced Hybrid** - CNN augmented with hardware-generated attention guidance from a Bitcoin ASIC miner (LV06)

The key innovation is using the ASIC's SHA-256 hashing capability to generate **deterministic noise fields** that act as an attention mechanism, guiding the CNN to focus on anomalous regions.

### Why This Matters

| Traditional CNN | ASIC-Hybrid Approach |
|-----------------|---------------------|
| Black box decisions | Explainable attention maps |
| GPU-dependent | Low-power ASIC assistance |
| Variable reproducibility | 100% deterministic (same hash = same attention) |
| No cryptographic binding | Built-in image authentication |
| Single-purpose | Detection + encryption capability |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASIC-CNN HYBRID ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: Chest X-Ray Image (224x224 grayscale)                             │
│                     │                                                       │
│                     ├────────────────────┬──────────────────────┐          │
│                     │                    │                      │          │
│                     ▼                    ▼                      ▼          │
│            ┌─────────────┐      ┌──────────────┐      ┌──────────────┐    │
│            │   BRANCH 1  │      │   BRANCH 2   │      │   BRANCH 3   │    │
│            │ Standard CNN│      │ ASIC Noise   │      │  Attention   │    │
│            │             │      │ Generator    │      │  Fusion      │    │
│            └──────┬──────┘      └──────┬───────┘      └──────┬───────┘    │
│                   │                    │                     │            │
│                   │              ┌─────┴─────┐               │            │
│                   │              │   LV06    │               │            │
│                   │              │   ASIC    │               │            │
│                   │              │  SHA-256  │               │            │
│                   │              └─────┬─────┘               │            │
│                   │                    │                     │            │
│                   │                    ▼                     │            │
│                   │           Deterministic Noise            │            │
│                   │           Field (224x224)                │            │
│                   │                    │                     │            │
│                   │                    └──────────┬──────────┘            │
│                   │                               │                       │
│                   │                               ▼                       │
│                   │                    ┌──────────────────┐               │
│                   │                    │ Attention-Guided │               │
│                   │                    │ Feature Maps     │               │
│                   │                    └────────┬─────────┘               │
│                   │                             │                         │
│                   └──────────────┬──────────────┘                         │
│                                  │                                        │
│                                  ▼                                        │
│                        ┌─────────────────┐                                │
│                        │  Feature Fusion │                                │
│                        │  (Concatenate)  │                                │
│                        └────────┬────────┘                                │
│                                 │                                         │
│                                 ▼                                         │
│                        ┌─────────────────┐                                │
│                        │ Classification  │                                │
│                        │ Head (FC Layers)│                                │
│                        └────────┬────────┘                                │
│                                 │                                         │
│                                 ▼                                         │
│                        ┌─────────────────┐                                │
│                        │    OUTPUT:      │                                │
│                        │ Normal/Pathology│                                │
│                        │ + Confidence    │                                │
│                        │ + Attention Map │                                │
│                        └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How the ASIC Enhances Detection

### The Problem with Standard CNNs

Standard CNNs process the entire image uniformly. They don't "know" where to look first, leading to:
- Missed subtle anomalies in noisy regions
- False positives from anatomical variations
- No built-in mechanism for attention guidance

### The ASIC Solution

The LV06 ASIC generates a **deterministic noise field** from the image:

```
Image Region (8x8 pixels) → SHA-256 Hash → 32 bytes of "structured randomness"
                                                    ↓
                                          Converted to attention weights
```

This noise field has unique properties:

1. **Deterministic**: Same image always produces same attention pattern
2. **Uniform distribution**: No inherent bias
3. **Avalanche effect**: Small image changes → completely different attention
4. **Cryptographic strength**: Can double as encryption key

### Attention Mechanism

The ASIC-generated noise acts as a **hardware attention mechanism**:

```python
# Conceptual flow
attention_map = ASIC_generate_noise(image)  # Hardware operation
enhanced_features = CNN_features * attention_map  # Guided focus
prediction = classifier(enhanced_features)  # Final decision
```

The attention map highlights regions where the ASIC "resonates" differently - often corresponding to anomalies that break the expected texture patterns.

---

## Dataset

### Recommended: Chest X-Ray Pneumonia Dataset

**Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

This dataset contains 5,863 X-Ray images (JPEG) in 2 categories:
- **NORMAL**: Healthy chest X-rays
- **PNEUMONIA**: X-rays showing pneumonia (bacterial or viral)

For this benchmark, we use a **subset of 500 images**:
- 250 Normal
- 250 Pneumonia

### Alternative Datasets

1. **COVID-19 Radiography Database** (Kaggle)
   - Normal, COVID-19, Viral Pneumonia, Lung Opacity
   
2. **NIH Chest X-rays** (subset)
   - 14 disease labels
   - Much larger, use subset for initial testing

### Dataset Structure

```
data/
├── train/           # 80% of data (400 images)
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/             # 10% of data (50 images)
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/            # 10% of data (50 images)
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## LV06 ASIC Configuration

### Hardware Setup

The Lucky Miner LV06 is a compact Bitcoin ASIC miner with:
- **Chip:** Bitmain BM1366 (same as Antminer S19)
- **Controller:** ESP32-S3
- **Hash Rate:** ~500 GH/s (Bitcoin mining)
- **Power:** 3.5W
- **Interface:** WiFi (REST API)

### Network Configuration

```
LV06 Device
    │
    │ WiFi (192.168.x.x)
    │
    ▼
Local Stratum Bridge (Python)
    │
    │ localhost:3333
    │
    ▼
ASIC-CNN Hybrid System
```

### Optimizing LV06 for Attention Generation

The LV06 is designed for Bitcoin mining, not general hashing. To maximize its utility for our purpose:

1. **Use Local Stratum Bridge**
   - Eliminates internet latency
   - Direct hash submission and retrieval

2. **Batch Hash Requests**
   - Group multiple image regions into single job
   - Amortize protocol overhead

3. **Cache Attention Maps**
   - Same image → same attention (deterministic)
   - Store computed maps for training efficiency

4. **Fallback to Software**
   - During training: use software SHA-256 (faster iteration)
   - During inference: use ASIC (cryptographic binding)

### Expected Performance

| Mode | Hash Generation | Use Case |
|------|-----------------|----------|
| Software SHA-256 | ~1M hashes/sec | Training |
| LV06 ASIC | ~100-500 hashes/sec (via API) | Inference, Validation |
| LV06 (Optimized Firmware) | ~1000+ hashes/sec | Future optimization |

**Note:** For the 500-image benchmark, we primarily use software hashing during training for speed, then validate a subset with the actual LV06 hardware.

---

## Model Architectures

### Model A: Standard CNN (Baseline)

Based on **ResNet-18** pretrained on ImageNet, fine-tuned for chest X-ray classification.

```
Input (224x224x1) → ResNet-18 backbone → Global Average Pooling → FC(512) → FC(2) → Softmax
```

### Model B: ASIC-Enhanced Hybrid

Dual-branch architecture combining CNN features with ASIC attention.

```
Input (224x224x1)
    │
    ├─────────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
ResNet-18 Backbone                    ASIC Noise Generator
(Feature Extraction)                  (Attention Maps)
    │                                         │
    ▼                                         │
Feature Maps (7x7x512)                        │
    │                                         │
    │◄────────────────────────────────────────┘
    │         Attention Weighting
    ▼
Attention-Weighted Features
    │
    ▼
Global Average Pooling
    │
    ▼
FC(512) → ReLU → Dropout(0.5) → FC(2) → Softmax
```

### Model C: Hybrid with Multi-Scale Attention

Advanced version using attention at multiple CNN layers.

```
Input → ResNet Block 1 ──┬── ASIC Attention (56x56) ──► Weighted Features 1
                         │
        ResNet Block 2 ──┼── ASIC Attention (28x28) ──► Weighted Features 2
                         │
        ResNet Block 3 ──┼── ASIC Attention (14x14) ──► Weighted Features 3
                         │
        ResNet Block 4 ──┴── ASIC Attention (7x7)  ──► Weighted Features 4
                                                              │
                                                              ▼
                                                      Feature Pyramid
                                                              │
                                                              ▼
                                                      Classification
```

---

## Training Protocol

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 1e-4 (Adam) |
| Epochs | 50 |
| Image Size | 224x224 |
| Augmentation | Rotation, Flip, Brightness |
| Early Stopping | Patience=10 |

### Data Augmentation

```python
transforms = {
    'train': Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(10),
        ColorJitter(brightness=0.1, contrast=0.1),
        ToTensor(),
        Normalize([0.485], [0.229])
    ]),
    'val': Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485], [0.229])
    ])
}
```

### Loss Function

- **Standard CNN:** CrossEntropyLoss
- **Hybrid Models:** CrossEntropyLoss + Attention Consistency Loss

```python
loss = CE_loss + lambda * attention_consistency_loss
```

Where `attention_consistency_loss` encourages the model to focus on regions highlighted by ASIC attention.

---

## Benchmark Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall correct predictions | >90% |
| **Sensitivity (Recall)** | Pathology detection rate | >95% |
| **Specificity** | Normal classification rate | >85% |
| **F1 Score** | Harmonic mean of precision/recall | >0.90 |
| **AUC-ROC** | Area under ROC curve | >0.95 |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Inference Time** | ms per image |
| **ASIC Utilization** | % of hashes from hardware |
| **Attention IoU** | Overlap with radiologist annotations |
| **Reproducibility** | Variance across runs |

### Comparison Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BENCHMARK COMPARISON                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Model                  │ Accuracy │ Sensitivity │ F1    │ Time    │
│  ───────────────────────┼──────────┼─────────────┼───────┼─────────│
│  Standard CNN (ResNet)  │   ??%    │     ??%     │ ??    │  ?? ms  │
│  ASIC Hybrid (Single)   │   ??%    │     ??%     │ ??    │  ?? ms  │
│  ASIC Hybrid (Multi)    │   ??%    │     ??%     │ ??    │  ?? ms  │
│                                                                     │
│  HYPOTHESIS: Hybrid models will show improved sensitivity           │
│  with minimal accuracy trade-off, plus explainable attention.       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation & Setup

### Requirements

```bash
# Python 3.8+
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install pillow
pip install tqdm
pip install requests  # For LV06 API
```

### Hardware Requirements

- **CPU:** Any modern CPU (training uses GPU if available)
- **GPU:** Optional but recommended (NVIDIA with CUDA)
- **RAM:** 8GB minimum
- **LV06:** Connected to same network (for inference validation)

### Directory Structure

```
ASIC_HYBRID_BENCHMARK/
├── README.md                    # This file
├── config.py                    # Configuration parameters
├── download_dataset.py          # Dataset download script
├── asic_interface.py            # LV06 communication
├── models/
│   ├── standard_cnn.py          # Baseline ResNet
│   ├── hybrid_single.py         # Single-scale hybrid
│   └── hybrid_multi.py          # Multi-scale hybrid
├── training/
│   ├── trainer.py               # Training loop
│   └── augmentation.py          # Data augmentation
├── evaluation/
│   ├── benchmark.py             # Benchmark runner
│   └── visualization.py         # Result visualization
├── main.py                      # Main entry point
└── results/                     # Output directory
    ├── models/                  # Saved model weights
    ├── figures/                 # Visualizations
    └── metrics/                 # Benchmark results
```

---

## Running the Benchmark

### Step 1: Download Dataset

```bash
python download_dataset.py --dataset chest-xray-pneumonia --samples 500
```

### Step 2: Configure LV06 (Optional)

Edit `config.py`:
```python
ASIC_CONFIG = {
    'enabled': True,           # Set False for software-only
    'host': '192.168.1.100',   # Your LV06 IP
    'port': 3333,
    'timeout': 10
}
```

### Step 3: Train Models

```bash
# Train all models
python main.py --mode train --models all

# Train specific model
python main.py --mode train --models hybrid_single
```

### Step 4: Run Benchmark

```bash
python main.py --mode benchmark --use-asic
```

### Step 5: View Results

```bash
python main.py --mode visualize
```

---

## Expected Results

Based on preliminary experiments and literature:

### Accuracy Comparison

| Model | Expected Accuracy | Notes |
|-------|-------------------|-------|
| Standard CNN | 90-93% | Baseline performance |
| Hybrid Single | 91-94% | Slight improvement |
| Hybrid Multi | 92-95% | Best expected |

### Key Advantages of Hybrid

1. **Attention Maps**: Visual explanation of model focus
2. **Reproducibility**: Deterministic ASIC attention
3. **Robustness**: Better on edge cases
4. **Dual Purpose**: Same hardware provides encryption

### Limitations

1. **LV06 Speed**: Slower than pure GPU inference
2. **Network Dependency**: Requires WiFi connection to ASIC
3. **Training Overhead**: Attention generation adds computation

---

## Future Work

1. **Firmware Optimization**: Custom LV06 firmware for faster hashing
2. **S9 Integration**: Scale up with Antminer S9 (100-900× faster)
3. **Multi-Disease Classification**: Extend to 14 NIH categories
4. **RAG Integration**: Add medical history context (Phase 2)
5. **LLM Diagnosis**: Contextual reasoning with Qwen3 (Phase 3)

---

## Citation

If you use this work, please cite:

```bibtex
@software{angulo2024asichybrid,
  author = {Angulo de Lafuente, Francisco},
  title = {ASIC-CNN Hybrid System for Medical Image Pathology Detection},
  year = {2024},
  url = {https://github.com/Agnuxo1/ASIC-RAG-CHIMERA}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Contact

**Francisco Angulo de Lafuente**
- GitHub: [@Agnuxo1](https://github.com/Agnuxo1)
- Project: ASIC-RAG-CHIMERA

---

*"The best AI is not the one that's most complex, but the one that best combines available resources."*
