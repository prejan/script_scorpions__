# IEEE GRSS Hackathon - Data Driven AI in Remote Sensing
## Team: Solar Hawks – Skibidi
### SRM Institute of Science and Technology

## Project Overview
This project implements a deep learning solution for remote sensing data analysis using PyTorch. The system is designed to detect and segment burn scars from satellite imagery using a transformer-based architecture.

## WanDB Link
[https://wandb.ai/srm-srm/rsds-hackathon-24/reports/GRSS-HACKTHON-SKIBIDI--Vmlldzo5ODYwMjUy?accessToken=bd3brxg2l1gbq4g9bd048d5egfwmulm03cwapxxqlboqit6qr9kob0vs1o3blkuv](https://wandb.ai/srm-srm/rsds-hackathon-24/reports/GRSS-HACKTHON-SKIBIDI--Vmlldzo5ODYwMjUy?accessToken=bd3brxg2l1gbq4g9bd048d5egfwmulm03cwapxxqlboqit6qr9kob0vs1o3blkuv)
## Initial Setup
### Prerequisites
- AWS Account with appropriate permissions
- Python environment with required packages
- Git

### Environment Setup
1. Access AWS Login Portal using assigned team credentials
2. Create a new private JupyterLab space
3. Clone the repository:
```bash
git clone https://github.com/NASA-IMPACT/rsds-hackathon-24.git
```
4. Install system dependencies:
```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
```
5. Install Python requirements:
```bash
pip install -r requirements.txt
```

## Model Architecture
The project implements a Transformer-based segmentation model with the following key components:
- **Patch Embedding**: Converts input images into embedded patches
- **Multi-Head Attention**: Processes spatial relationships in the data
- **Transformer Blocks**: Deep learning layers for feature extraction
- **Segmentation Head**: Final layer for pixel-wise classification

## Training Evolution and Configuration
### Stage 1
- Mixed-precision training (16-bit floating point)
- Increased early stopping patience (5 epochs)
- Max epochs: 100
- Batch size: 16
- AdamW optimizer (lr: 0.001, weight_decay: 0.01)
- OneCycleLR scheduler (max_lr: 0.001)
- Dice & Cross-Entropy combined loss
- Enabled checkpointing

### Stage 2
- Enhanced logging with version tracking
- Early stopping patience: 10 epochs
- Batch size: 8
- Gradient clipping (1.0)
- UperNetDecoder implementation
- Additional augmentations (ColorJitter, increased RandomCrop size)
- Switched to pure Dice loss
- Adam optimizer with lower weight decay

### Stage 3
- Mixed-precision training optimization
- Early stopping patience: 3
- Batch size: 16
- Enhanced augmentations (vertical and 90-degree rotations)
- Optimized checkpointing
- Dice & Cross-Entropy with class weighting
- AdamW optimizer

### Final Configuration
- Batch size: 4
- Max epochs: 100
- Simple augmentations (RandomCrop 224x224, HorizontalFlip)
- FCNDecoder with 256 decoder channels
- Dice loss function
- Adam optimizer (lr: 1.3e-5)
- ReduceLROnPlateau scheduler

## Performance Metrics and Results
### Final Test Metrics (Version 11)
- **Multiclass Accuracy**: 0.9671
- **Multiclass F1 Score**: 0.9671
- **Multiclass Jaccard Index**: 0.8472
- **Multiclass Jaccard Index (Micro)**: 0.9364
- **Test Loss**: 0.1276

### Class-Specific Metrics
- **Accuracy**:
  - Class 0: 0.9879
  - Class 1: 0.8011
- **Jaccard Index**:
  - Class 0: 0.9639
  - Class 1: 0.7306

## Key Features
- Custom BurnScarDataset implementation
- Real-time metrics tracking
- Advanced data augmentation pipeline
- Comprehensive logging system
- Training curve visualization
- Multi-metric evaluation (IoU, F1-Score, Accuracy)

## Dependencies
- PyTorch
- rasterio
- pandas
- numpy
- matplotlib
- tqdm
- torchmetrics
- einops

## Monitoring and Logging
- Comprehensive logging system with both file and console output
- Real-time metric tracking through Weights & Biases and TensorBoard
- Training curve visualization every 5 epochs
- Model checkpointing based on validation IoU

## Usage Instructions
1. Prepare the environment as described in setup
2. Open `training_terratorch.ipynb` in JupyterLab
3. Follow the notebook instructions for training
4. Monitor progress using Weights & Biases and TensorBoard

## Contributing
For contributions, please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Summary
The project achieved exceptional results with a Multiclass Accuracy of 96.71% and a Jaccard Index of 84.72% on the validation dataset. The training evolution shows systematic improvements through multiple stages, ultimately arriving at a simplified yet highly effective configuration. The final approach successfully balanced model complexity with performance, using basic augmentations and an FCNDecoder architecture.

## Acknowledgments
- SRM Institute of Science and Technology
- IEEE GRSS
- NASA-IMPACT
