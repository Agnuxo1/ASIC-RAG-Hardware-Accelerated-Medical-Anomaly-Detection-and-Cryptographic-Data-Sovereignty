# ASIC-RAG: Hardware-Accelerated Medical Anomaly Detection and Cryptographic Data Sovereignty

This repository contains the full codebase for the **ASIC-RAG-CHIMERA** framework, focusing on medical anomaly detection (Pneumonia) using hybrid ASIC-CNN architectures.

## Repository Structure

- `Firmware_ASIC_LV06/`: Drivers and firmware for the Lucky Miner LV06 hardware.
- `V4/`: The latest multi-scale hybrid model implementation, including trainer, models, and evaluation scripts.
- `ASIC-RAG-CHIMERA_...pdf`: Academic whitepaper detailing the framework.
- `ai_studio_code.html`: Interactive research report.

## Dataset Information

To keep the repository lightweight and comply with GitHub file size limits, the radiographs dataset is **not included** in this repository.

### Download Dataset
The model is trained on the **Pneumonia X-Ray Dataset**. You can download it directly from Hugging Face:

**Source:** [mmenendezg/pneumonia_x_ray](https://huggingface.co/datasets/mmenendezg/pneumonia_x_ray)

After downloading, place the dataset in the `V4/data/` directory with the following structure:
```
V4/data/
├── train/
├── val/
└── test/
```

## Hardware Requirements
This system is designed to interface with **Lucky Miner LV06 (BM1366 ASIC)** for hardware-accelerated attention maps and cryptographic verification.

## Author
Francisco Angulo de Lafuente
