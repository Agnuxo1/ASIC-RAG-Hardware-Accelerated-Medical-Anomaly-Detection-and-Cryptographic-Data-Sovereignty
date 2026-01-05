<img width="2880" height="1620" alt="1" src="https://github.com/user-attachments/assets/09919568-e101-4dbc-b01c-648c712868eb" />

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

![2](https://github.com/user-attachments/assets/1372e232-0c0e-4d5a-a29a-269eff2dbb42)
![3](https://github.com/user-attachments/assets/8a7869ea-af25-4424-bbd0-d99443bd30cd)
![4](https://github.com/user-attachments/assets/82f72aed-5c93-4a97-9a35-6659739a885d)
![5](https://github.com/user-attachments/assets/cbf1084c-bc95-4383-b5a5-e0928f1228e9)
![6](https://github.com/user-attachments/assets/f46c0adf-1b43-4479-a51f-36f2354c0472)
![7](https://github.com/user-attachments/assets/1b7392b9-06cf-418a-b83d-db8477f23c37)
![8](https://github.com/user-attachments/assets/c3af2f30-b38e-41e2-a855-401d7e29f7e7)
![9](https://github.com/user-attachments/assets/4ec37d32-42a4-434a-81cc-d9d03c481f45)
![10](https://github.com/user-attachments/assets/115e4377-8552-446a-b3c5-e6cd648962f4)
![11](https://github.com/user-attachments/assets/ba3092c8-017d-4002-b34e-89d871592914)
![12](https://github.com/user-attachments/assets/e2d49a87-ce10-4661-ad0d-be810fc0ae9a)


