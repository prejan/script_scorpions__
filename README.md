# IEEE GRSS Hackathon - Data Driven AI in Remote Sensing
## Team: Script_scorpions
### SRM Institute of Science and Technology

# Burn Scar Detection from Satellite Imagery

This project implements a deep learning solution for analyzing remote sensing data using PyTorch. The system is designed to detect and segment burn scars in satellite images using a transformer-based architecture.



## Table of Contents
- [Initial Setup](#initial-setup)
- [Model Architecture](#model-architecture)
- [Training Evolution](#training-evolution)
- [Performance and Metrics](#performance-and-metrics)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Monitoring and Logging](#monitoring-and-logging)
- [Usage Instructions](#usage-instructions)
- [Contributing](#contributing)
- [Summary](#summary)
- [Acknowledgments](#acknowledgments)

## Initial Setup

### Prerequisites
- AWS account with required permissions
- Python environment with necessary packages
- Git

### Environment Setup
1. Log in to AWS using team credentials.
2. Set up a private JupyterLab workspace.
3. Clone the project repository:
   ```bash
   git clone https://github.com/NASA-IMPACT/rsds-hackathon-24.git
   Install system dependencies:
bash
Copy code
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
Install Python requirements:
bash
Copy code
pip install -r requirements.txt
Model Architecture
The system utilizes a transformer-based segmentation model, consisting of:

Patch Embedding: Converts input images into embedded patches.
Multi-Head Attention: Captures spatial relationships in the data.
Transformer Blocks: For deep feature extraction.
Segmentation Head: For pixel-wise classification and burn scar detection.
Training Evolution
Stage 1
Mixed-precision training (16-bit floating point).
Early stopping patience: 5 epochs.
Max epochs: 100, Batch size: 16.
AdamW optimizer (lr: 0.001, weight_decay: 0.01).
OneCycleLR scheduler, combined Dice and Cross-Entropy loss.
Checkpointing enabled.
Stage 2
Improved logging and version tracking.
Early stopping patience: 10 epochs.
Batch size: 8, Gradient clipping: 1.0.
UperNetDecoder, added augmentations (ColorJitter, larger RandomCrop).
Dice loss, Adam optimizer with reduced weight decay.
Stage 3
Optimized mixed-precision training.
Early stopping patience: 3 epochs, Batch size: 16.
Enhanced augmentations (vertical and 90-degree rotations).
Optimized checkpointing, Dice and Cross-Entropy loss with class weighting.
AdamW optimizer.
Final Configuration
Batch size: 4, Max epochs: 100.
Simple augmentations (RandomCrop 224x224, HorizontalFlip).
FCNDecoder with 256 decoder channels.
Dice loss function, Adam optimizer (lr: 1.3e-5), ReduceLROnPlateau scheduler.
Performance and Metrics
Final Test Metrics (Version 11)
Multiclass Accuracy: 96.71%
Multiclass F1 Score: 96.71%
Multiclass Jaccard Index: 84.72%
Multiclass Jaccard Index (Micro): 93.64%
Test Loss: 0.1276
Class-Specific Metrics
Accuracy:
Class 0: 98.79%
Class 1: 80.11%
Jaccard Index:
Class 0: 96.39%
Class 1: 73.06%
Key Features
Custom BurnScarDataset for efficient data handling.
Real-time metrics tracking.
Advanced data augmentation pipeline.
Detailed logging and visualization of training progress.
Multi-metric evaluation (IoU, F1-Score, Accuracy).
Dependencies
PyTorch
rasterio
pandas
numpy
matplotlib
tqdm
torchmetrics
einops
Monitoring and Logging
Detailed logging (both file and console outputs).
Real-time metric tracking using Weights & Biases and TensorBoard.
Training curve visualization every 5 epochs.
Model checkpointing based on validation IoU.
Usage Instructions
Set up the environment as described.
Open training_terratorch.ipynb in JupyterLab.
Follow the notebook instructions for training.
Monitor progress using Weights & Biases and TensorBoard.
Contributing
Fork the repository.
Create a feature branch.
Submit a pull request.
Summary
This project successfully detects and segments burn scars with a multiclass accuracy of 96% and a Jaccard Index of 84%. The model evolved through various training stages, balancing complexity with performance, and achieving strong results with an FCNDecoder and basic augmentations.


