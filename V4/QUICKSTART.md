# Quick Start Guide: ASIC-CNN Hybrid Benchmark

## 🚀 Get Running in 5 Minutes

### Step 1: Install Dependencies

```bash
cd ASIC_HYBRID_BENCHMARK
pip install -r requirements.txt
```

For GPU acceleration (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Download Real Dataset

**Option A: Kaggle (Recommended - Real X-rays)**

1. Create Kaggle account at https://www.kaggle.com
2. Go to Account → Create API Token → Download `kaggle.json`
3. Place in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows)
4. Run:
```bash
pip install kaggle
python main.py --mode dataset --kaggle
```

**Option B: Manual Download**

1. Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extract to `ASIC_HYBRID_BENCHMARK/data/`
3. Structure should be:
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Option C: Synthetic Dataset (For Testing Only)**

```bash
python main.py --mode dataset --synthetic --samples 500
```

### Step 3: Configure LV06 ASIC (Optional)

Edit `config.py`:
```python
ASIC_CONFIG = {
    'enabled': True,              # Set to True
    'host': '192.168.1.100',      # Your LV06 IP address
    'port': 4028,
    'stratum_port': 3333,
    ...
}
```

Test connection:
```bash
python main.py --mode test-asic
```

### Step 4: Train Models

```bash
# Train all models (Standard CNN + Hybrid variants)
python main.py --mode train --models all

# Or train specific model
python main.py --mode train --models standard_cnn
python main.py --mode train --models hybrid_single
python main.py --mode train --models hybrid_multi
```

### Step 5: Run Benchmark

```bash
python main.py --mode benchmark
```

### Step 6: View Results

Results are saved to:
- `results/metrics/benchmark_report.md` - Full report
- `results/metrics/benchmark_results.json` - Raw data
- `results/figures/` - Visualizations

---

## 📊 Expected Output

After running the benchmark, you'll see:

```
════════════════════════════════════════════════════════════════════════
ASIC-CNN HYBRID BENCHMARK
════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────
Evaluating: standard_cnn
Description: Standard ResNet-18 baseline
────────────────────────────────────────────────────────────────────────
  Loaded weights from results/models/standard_cnn/best_model.pth

  Results for standard_cnn:
    Accuracy:    0.9200
    Sensitivity: 0.9400
    Specificity: 0.9000
    Precision:   0.9038
    F1 Score:    0.9215
    AUC-ROC:     0.9650
    Inference:   12.34 ± 2.10 ms

────────────────────────────────────────────────────────────────────────
Evaluating: hybrid_single
Description: Hybrid with single-scale ASIC attention
────────────────────────────────────────────────────────────────────────
  ...
```

---

## 🔧 Command Reference

| Command | Description |
|---------|-------------|
| `python main.py --mode config` | Show current configuration |
| `python main.py --mode dataset --synthetic` | Create synthetic dataset |
| `python main.py --mode dataset --kaggle` | Download Kaggle dataset |
| `python main.py --mode train --models all` | Train all models |
| `python main.py --mode benchmark` | Evaluate models |
| `python main.py --mode visualize` | Generate plots |
| `python main.py --mode full` | Run complete pipeline |
| `python main.py --mode test-asic` | Test ASIC connection |

### Useful Options

```bash
# Override epochs
python main.py --mode train --models all --epochs 20

# Override batch size
python main.py --mode train --models all --batch-size 32

# Specify ASIC host
python main.py --mode train --asic-host 192.168.1.50

# Disable ASIC (software only)
python main.py --mode train --no-asic
```

---

## 🔌 LV06 ASIC Setup

### Network Configuration

1. Connect LV06 to your network via WiFi
2. Find its IP address (check your router's DHCP leases)
3. Verify connectivity: `ping 192.168.1.100`

### Stratum Bridge (For Direct Hashing)

For maximum ASIC utilization, run a local stratum bridge:

```python
# Simple stratum bridge (included in asic_interface.py)
from asic_interface import LV06Interface

asic = LV06Interface(host='192.168.1.100')
if asic.connect():
    print("Connected!")
    stats = asic.get_stats()
    print(f"Hash rate: {stats.hash_rate} GH/s")
```

### Fallback Behavior

If ASIC is unavailable, the system automatically falls back to software SHA-256. This ensures:
- Training can proceed without hardware
- Results are reproducible
- ASIC can be added later for validation

---

## 📁 Project Structure

```
ASIC_HYBRID_BENCHMARK/
├── main.py              # Entry point
├── config.py            # Configuration
├── models.py            # CNN architectures
├── dataset.py           # Data loading
├── trainer.py           # Training logic
├── benchmark.py         # Evaluation
├── asic_interface.py    # LV06 communication
├── requirements.txt     # Dependencies
├── README.md            # Documentation
├── QUICKSTART.md        # This file
├── data/                # Dataset (created)
│   ├── train/
│   ├── val/
│   └── test/
└── results/             # Output (created)
    ├── models/          # Saved weights
    ├── figures/         # Visualizations
    └── metrics/         # Benchmark results
```

---

## 🐛 Troubleshooting

### "No module named torch"
```bash
pip install torch torchvision
```

### "CUDA out of memory"
Reduce batch size in `config.py` or use:
```bash
python main.py --mode train --batch-size 8
```

### "Dataset not found"
Ensure data is in correct structure:
```bash
python main.py --mode dataset --synthetic
```

### "ASIC connection failed"
1. Check IP address in `config.py`
2. Ensure LV06 is powered and connected to network
3. Try: `ping <LV06_IP>`
4. Continue with `--no-asic` for software-only mode

### Training is slow
1. Use GPU: ensure PyTorch detects CUDA
2. Reduce epochs: `--epochs 20`
3. Use smaller batch: `--batch-size 8`

---

## 📈 Interpreting Results

### Key Metrics

| Metric | What it Measures | Target |
|--------|------------------|--------|
| **Accuracy** | Overall correctness | >90% |
| **Sensitivity** | Detecting pneumonia when present | >95% (critical!) |
| **Specificity** | Correctly identifying normal | >85% |
| **F1 Score** | Balance of precision/recall | >0.90 |
| **AUC-ROC** | Overall discriminative ability | >0.95 |

### What to Look For

1. **Hybrid vs Standard**: Does ASIC attention improve metrics?
2. **Sensitivity Priority**: In medical imaging, missing disease is worse than false alarm
3. **Inference Time**: Is hybrid model acceptable speed?
4. **Attention Maps**: Do they highlight clinically relevant regions?

---

## 🎯 Next Steps

After successful benchmark:

1. **Validate with Real ASIC**: Enable LV06 for final validation
2. **Test on Different Datasets**: Try COVID-19, tuberculosis datasets
3. **Optimize Attention**: Tune attention scales and weights
4. **Clinical Validation**: Get radiologist feedback on attention maps
5. **Integration**: Connect to CRYPTO-RAG for secure medical records

---

## 📚 References

- [Chest X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Attention Mechanisms in Medical Imaging](https://arxiv.org/abs/1804.03999)
- [LV06 Documentation](https://github.com/BitMaker-hub/NerdMiner_v2)

---

**Author:** Francisco Angulo de Lafuente  
**GitHub:** https://github.com/Agnuxo1  
**Project:** ASIC-RAG-CHIMERA Medical Extension
