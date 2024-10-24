import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import pandas as pd
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
import glob
from torchmetrics import Accuracy, F1Score, JaccardIndex
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
torch.set_num_threads(multiprocessing.cpu_count())

class AsyncDataLoader:
    def __init__(self, dataloader, num_workers=4, queue_size=3):
        self.dataloader = dataloader
        self.queue = Queue(maxsize=queue_size)
        self.num_workers = num_workers
        self.stop_event = threading.Event()
        self.workers = []
        
    def start(self):
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_fn)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
    def _worker_fn(self):
        try:
            while not self.stop_event.is_set():
                for batch in self.dataloader:
                    if self.stop_event.is_set():
                        break
                    self.queue.put(batch)
                if not self.stop_event.is_set():
                    self.queue.put(None)  # End of epoch marker
                    break
        except Exception as e:
            print(f"Worker error: {e}")
            
    def __iter__(self):
        self.stop_event.clear()
        self.start()
        active_workers = self.num_workers
        
        while active_workers > 0:
            batch = self.queue.get()
            if batch is None:
                active_workers -= 1
                continue
            yield batch
            
    def stop(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
        self.workers = []

class ParallelDataProcessor:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
    def process_batch(self, batch_data, device):
        images, masks = batch_data
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        return images, masks

# [Previous MetricsCalculator class remains the same]
class BurnScarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Find all files in the directory
        merged_files = glob.glob(os.path.join(root_dir, "*_merged.tif"))
        if not merged_files:
            raise ValueError(f"No merged files found in {root_dir}")
            
        logger.info(f"Found {len(merged_files)} merged files in {root_dir}")
        
        merged_mask_pairs = []
        for merged_file in merged_files:
            base_name = merged_file.replace("_merged.tif", "")
            mask_file = f"{base_name}.mask.tif"
            
            if os.path.exists(mask_file):
                # Verify file readability
                try:
                    with rasterio.open(merged_file) as src:
                        merged_shape = src.shape
                    with rasterio.open(mask_file) as src:
                        mask_shape = src.shape
                        
                    if merged_shape == mask_shape:
                        merged_mask_pairs.append({
                            'merged': merged_file,
                            'mask': mask_file
                        })
                    else:
                        logger.warning(f"Shape mismatch for pair {base_name}: {merged_shape} vs {mask_shape}")
                except Exception as e:
                    logger.warning(f"Could not read pair {base_name}: {str(e)}")
                    continue
            else:
                logger.warning(f"No matching mask file for {merged_file}")
        
        if not merged_mask_pairs:
            raise ValueError(f"No valid image pairs found in {root_dir}")
            
        self.pairs = merged_mask_pairs
        logger.info(f"Successfully loaded {len(self.pairs)} valid image pairs")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        try:
            # Read and process image
            with rasterio.open(pair['merged']) as src:
                image = src.read().astype(np.float32)
                # Normalize image
                image = (image - image.mean()) / (image.std() + 1e-6)
                
            with rasterio.open(pair['mask']) as src:
                mask = src.read(1).astype(np.int64)
                
            if self.transform:
                image = self.transform(image)
                
            return torch.from_numpy(image), torch.from_numpy(mask)
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class TransformerSegmentation(nn.Module):
    def __init__(self, in_channels=6, num_classes=2, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (512 // patch_size) ** 2, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm and head
        x = self.norm(x)
        x = self.head(x)
        
        # Reshape to image dimensions
        x = x.reshape(B, H // 16, W // 16, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x
    
def plot_training_curves(history, output_dir, epoch):
    """
    Plot and save training curves
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'training_curves_epoch_{epoch}.png'))
    plt.close()

class MetricsCalculator:
    def __init__(self, num_classes=2, device='mps'):
        self.accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
        self.f1_score = MulticlassF1Score(num_classes=num_classes).to(device)
        self.jaccard = MulticlassJaccardIndex(num_classes=num_classes).to(device)
        self.jaccard_micro = MulticlassJaccardIndex(num_classes=num_classes, average='micro').to(device)
        self.class_accuracy = nn.ModuleList([
            MulticlassAccuracy(num_classes=num_classes, average='none').to(device)
            for _ in range(num_classes)
        ])
        self.class_jaccard = nn.ModuleList([
            MulticlassJaccardIndex(num_classes=num_classes, average='none').to(device)
            for _ in range(num_classes)
        ])
        
    def update(self, preds, target):
        self.accuracy.update(preds, target)
        self.f1_score.update(preds, target)
        self.jaccard.update(preds, target)
        self.jaccard_micro.update(preds, target)
        for i in range(len(self.class_accuracy)):
            self.class_accuracy[i].update(preds, target)
            self.class_jaccard[i].update(preds, target)
            
    def compute(self):
        return {
            'Multiclass_Accuracy': self.accuracy.compute(),
            'Multiclass_F1_Score': self.f1_score.compute(),
            'Multiclass_Jaccard_Index': self.jaccard.compute(),
            'Multiclass_Jaccard_Index_Micro': self.jaccard_micro.compute(),
            'multiclassaccuracy_0': self.class_accuracy[0].compute()[0],
            'multiclassaccuracy_1': self.class_accuracy[0].compute()[1],
            'multiclassjaccardindex_0': self.class_jaccard[0].compute()[0],
            'multiclassjaccardindex_1': self.class_jaccard[0].compute()[1]
        }
        
    def reset(self):
        self.accuracy.reset()
        self.f1_score.reset()
        self.jaccard.reset()
        self.jaccard_micro.reset()
        for metric in self.class_accuracy:
            metric.reset()
        for metric in self.class_jaccard:
            metric.reset()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, output_dir):
    """
    Training function with multithreaded optimization
    """
    best_val_iou = 0
    history = {'train_loss': [], 'train_metrics': [], 'val_loss': [], 'val_metrics': []}
    
    # Initialize parallel components
    data_processor = ParallelDataProcessor(num_workers=4)
    metrics_calc = MetricsCalculator(num_classes=3, device=device)
    scaler = GradScaler('mps')  # For mixed precision training
    
    # Create async data loaders
    async_train_loader = AsyncDataLoader(train_loader, num_workers=4)
    async_val_loader = AsyncDataLoader(val_loader, num_workers=4)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        metrics_calc.reset()
        batch_futures = []
        
        train_iterator = tqdm(async_train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in train_iterator:
            # Process batch in parallel
            future = data_processor.executor.submit(data_processor.process_batch, batch, device)
            batch_futures.append(future)
            
            # Process completed batches
            while len(batch_futures) >= data_processor.num_workers:
                future = batch_futures.pop(0)
                images, masks = future.result()
                
                # Mixed precision training
                with autocast():  # MPS doesn't support autocast yet
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                metrics_calc.update(predictions, masks)
        
        # Process remaining batches
        for future in batch_futures:
            images, masks = future.result()
            with autocast(device_type='cpu'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            metrics_calc.update(predictions, masks)
        
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = metrics_calc.compute()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        metrics_calc.reset()
        batch_futures = []
        
        with torch.no_grad():
            for batch in async_val_loader:
                future = data_processor.executor.submit(data_processor.process_batch, batch, device)
                batch_futures.append(future)
                
                while len(batch_futures) >= data_processor.num_workers:
                    future = batch_futures.pop(0)
                    images, masks = future.result()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    predictions = outputs.argmax(dim=1)
                    metrics_calc.update(predictions, masks)
            
            # Process remaining batches
            for future in batch_futures:
                images, masks = future.result()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                metrics_calc.update(predictions, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = metrics_calc.compute()
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics and update history
        logger.info(f"Epoch {epoch+1}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info("Train Metrics:")
        for k, v in train_metrics.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info("Val Metrics:")
        for k, v in val_metrics.items():
            logger.info(f"{k}: {v:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['train_metrics'].append(train_metrics)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Save best model
        if val_metrics['Multiclass_Jaccard_Index'] > best_val_iou:
            best_val_iou = val_metrics['Multiclass_Jaccard_Index']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history,
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Plot training curves
        if (epoch + 1) % 5 == 0:
            plot_training_curves(history, output_dir, epoch + 1)
    
    # Clean up
    async_train_loader.stop()
    async_val_loader.stop()
    data_processor.executor.shutdown()
    
    return model, history

def main():
    # Configuration
    config = {
        'batch_size': 8,
        'num_epochs': 5,
        'learning_rate': 1e-2,
        'weight_decay': 0.01,
        'patch_size': 16,
        'embed_dim': 768,
        'num_heads': 12,
        'depth': 12,
        'num_workers': 4,  # Adjust based on your CPU cores
        'prefetch_factor': 2,
        'pin_memory': True
    }
    
    # Create output directory
    output_dir = f"training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    global logger
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device('mps')
    logger.info(f"Using device: {device}")
    
    try:
        # Create datasets with optimized loading
        train_dataset = BurnScarDataset(
            '/Users/admin63/Python-Programs/Geospatial-Image-Stitcher-and-cloud-remover/custom-transformer-inferences/hls_burn_scars/training'
        )
        val_dataset = BurnScarDataset(
            '/Users/admin63/Python-Programs/Geospatial-Image-Stitcher-and-cloud-remover/custom-transformer-inferences/hls_burn_scars/validation'
        )
        
        # Create optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=True
        )
        
        # Initialize model with optimized memory handling
        model = TransformerSegmentation(
            in_channels=6,
            num_classes=2,
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads']
        ).to(device)
        
        # Setup training with optimization flags
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            eps=1e-8,
            betas=(0.9, 0.999),
            maximize=False,
            foreach=True  # Enables more efficient parameter updates
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs']
        )
        
        # Train model with optimizations
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['num_epochs'],
            device=device,
            output_dir=output_dir
        )
        
        # Save final model and metrics
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'config': config
        }, os.path.join(output_dir, 'final_model.pth'))
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()