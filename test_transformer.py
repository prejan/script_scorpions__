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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# class BurnScarDataset(Dataset):
#     def __init__(self, root_dir, index_file, transform=None):
#         """
#         Args:
#             root_dir (str): Directory with all the images
#             index_file (str): Path to the index CSV file
#             transform (callable, optional): Optional transform to be applied on a sample
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.index_df = pd.read_csv(index_file)
#         logger.info(f"Loaded dataset from {root_dir} with {len(self.index_df)} samples")
        
#     def __len__(self):
#         return len(self.index_df)
    
#     def __getitem__(self, idx):
#         try:
#             if torch.is_tensor(idx):
#                 idx = idx.tolist()
                
#             # Get image paths
#             merged_img_name = self.index_df.iloc[idx]['merged_filename']
#             mask_img_name = self.index_df.iloc[idx]['mask_filename']
            
#             merged_path = os.path.join(self.root_dir, merged_img_name)
#             mask_path = os.path.join(self.root_dir, mask_img_name)
            
#             # Read and process image
#             with rasterio.open(merged_path) as src:
#                 image = src.read().astype(np.float32)
#                 # Normalize image
#                 image = (image - image.mean()) / (image.std() + 1e-6)
                
#             with rasterio.open(mask_path) as src:
#                 mask = src.read(1).astype(np.int64)
            
#             if self.transform:
#                 image = self.transform(image)
                
#             return torch.from_numpy(image), torch.from_numpy(mask)
            
#         except Exception as e:
#             logger.error(f"Error loading sample {idx}: {str(e)}")
#             raise
################################################################
# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# import os
# import rasterio
# import logging
# import glob

# logger = logging.getLogger(__name__)

# class BurnScarDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         """
#         Args:
#             root_dir (str): Directory with all the images
#             transform (callable, optional): Optional transform to be applied on a sample
#         """
#         self.root_dir = root_dir
#         self.transform = transform
        
#         # Find all files in the directory
#         merged_files = glob.glob(os.path.join(root_dir, "*_merged.tif"))
#         merged_mask_pairs = []
        
#         for merged_file in merged_files:
#             # Get the base name without _merged.tif
#             base_name = merged_file.replace("_merged.tif", "")
#             mask_file = f"{base_name}.mask.tif"
            
#             # Check if corresponding mask file exists
#             if os.path.exists(mask_file):
#                 merged_mask_pairs.append({
#                     'merged': os.path.basename(merged_file),
#                     'mask': os.path.basename(mask_file)
#                 })
        
#         self.index_df = pd.DataFrame(merged_mask_pairs)
#         logger.info(f"Found {len(self.index_df)} merged/mask pairs")
        
#         # Verify all samples are loadable and create a valid samples mask
#         self.valid_samples = self._verify_samples()
#         self.valid_indices = np.where(self.valid_samples)[0]
        
#         if len(self.valid_indices) == 0:
#             logger.warning(f"No valid image pairs found in {root_dir}")
#             logger.warning(f"Directory contents: {os.listdir(root_dir)}")
            
#         logger.info(f"Loaded dataset from {root_dir} with {len(self.valid_indices)} valid samples "
#                    f"out of {len(self.index_df)} total pairs")
    
#     def _verify_samples(self):
#         """Verify which samples can be loaded successfully."""
#         valid_samples = np.ones(len(self.index_df), dtype=bool)
        
#         for idx in range(len(self.index_df)):
#             try:
#                 # Get filenames for the pair
#                 merged_img_name = self.index_df.iloc[idx]['merged']
#                 mask_img_name = self.index_df.iloc[idx]['mask']
                
#                 # Construct full paths
#                 merged_path = os.path.join(self.root_dir, merged_img_name)
#                 mask_path = os.path.join(self.root_dir, mask_img_name)
                
#                 # Check if files exist
#                 if not os.path.exists(merged_path):
#                     raise FileNotFoundError(f"Merged image not found: {merged_path}")
#                 if not os.path.exists(mask_path):
#                     raise FileNotFoundError(f"Mask image not found: {mask_path}")
                
#                 # Try opening the files to verify they are valid raster files
#                 with rasterio.open(merged_path) as src:
#                     _ = src.read()
#                 with rasterio.open(mask_path) as src:
#                     _ = src.read(1)
                    
#             except Exception as e:
#                 sample_info = f"Sample {idx}"
#                 try:
#                     sample_info += f" (merged: {merged_img_name}, mask: {mask_img_name})"
#                 except NameError:
#                     pass
#                 logger.warning(f"{sample_info} will be skipped: {str(e)}")
#                 valid_samples[idx] = False
                
#         return valid_samples

#     def __len__(self):
#         return len(self.valid_indices)
    
#     def __getitem__(self, idx):
#         # Map the requested index to the valid samples index
#         original_idx = self.valid_indices[idx]
        
#         # Get image paths
#         merged_img_name = self.index_df.iloc[original_idx]['merged']
#         mask_img_name = self.index_df.iloc[idx]['mask']
        
#         merged_path = os.path.join(self.root_dir, merged_img_name)
#         mask_path = os.path.join(self.root_dir, mask_img_name)
        
#         # Read and process image
#         with rasterio.open(merged_path) as src:
#             image = src.read().astype(np.float32)
#             # Normalize image
#             image = (image - image.mean()) / (image.std() + 1e-6)
            
#         with rasterio.open(mask_path) as src:
#             mask = src.read(1).astype(np.int64)
            
#         if self.transform:
#             image = self.transform(image)
            
#         return torch.from_numpy(image), torch.from_numpy(mask)
    
# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
#         super().__init__()
#         self.patch_size = patch_size
#         self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         x = self.projection(x)
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = self.norm(x)
#         return x

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.qkv = nn.Linear(embed_dim, embed_dim * 3)
#         self.proj = nn.Linear(embed_dim, embed_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.dropout(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         return x

# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * mlp_ratio, embed_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

# class TransformerSegmentation(nn.Module):
#     def __init__(self, in_channels=3, num_classes=2, patch_size=16, embed_dim=768,
#                  depth=12, num_heads=12, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
#         self.pos_embed = nn.Parameter(torch.zeros(1, (512 // patch_size) ** 2, embed_dim))
#         self.blocks = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(embed_dim // 2, num_classes)
#         )
        
#         # Initialize position embeddings
#         nn.init.normal_(self.pos_embed, std=0.02)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # Patch embedding
#         x = self.patch_embed(x)
        
#         # Add position embedding
#         x = x + self.pos_embed
        
#         # Apply transformer blocks
#         for block in self.blocks:
#             x = block(x)
        
#         # Final norm and head
#         x = self.norm(x)
#         x = self.head(x)
        
#         # Reshape to image dimensions
#         x = x.reshape(B, H // 16, W // 16, -1)
#         x = x.permute(0, 3, 1, 2)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
#         return x

# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
#                 num_epochs, device, output_dir):
#     """
#     Training function with validation
#     """
#     best_val_iou = 0
#     history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
    
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         train_loss = 0
#         train_iou = 0
#         batch_count = 0
        
#         for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
#             try:
#                 images = images.to(device)
#                 masks = masks.to(device)
                
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
                
#                 loss.backward()
#                 optimizer.step()
                
#                 # Calculate IoU
#                 predictions = outputs.argmax(dim=1)
#                 intersection = torch.logical_and(masks, predictions)
#                 union = torch.logical_or(masks, predictions)
#                 iou = (intersection.sum() / (union.sum() + 1e-6)).item()
                
#                 train_loss += loss.item()
#                 train_iou += iou
#                 batch_count += 1
                
#             except Exception as e:
#                 logger.error(f"Error in training batch: {str(e)}")
#                 continue
        
#         avg_train_loss = train_loss / batch_count
#         avg_train_iou = train_iou / batch_count
        
#         # Validation phase
#         model.eval()
#         val_loss = 0
#         val_iou = 0
#         batch_count = 0
        
#         with torch.no_grad():
#             for images, masks in tqdm(val_loader, desc='Validation'):
#                 try:
#                     images = images.to(device)
#                     masks = masks.to(device)
                    
#                     outputs = model(images)
#                     loss = criterion(outputs, masks)
                    
#                     predictions = outputs.argmax(dim=1)
#                     intersection = torch.logical_and(masks, predictions)
#                     union = torch.logical_or(masks, predictions)
#                     iou = (intersection.sum() / (union.sum() + 1e-6)).item()
                    
#                     val_loss += loss.item()
#                     val_iou += iou
#                     batch_count += 1
                    
#                 except Exception as e:
#                     logger.error(f"Error in validation batch: {str(e)}")
#                     continue
        
#         avg_val_loss = val_loss / batch_count
#         avg_val_iou = val_iou / batch_count
        
#         # Update learning rate
#         scheduler.step()
        
#         # Log metrics
#         logger.info(
#             f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, "
#             f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}"
#         )
        
#         # Update history
#         history['train_loss'].append(avg_train_loss)
#         history['train_iou'].append(avg_train_iou)
#         history['val_loss'].append(avg_val_loss)
#         history['val_iou'].append(avg_val_iou)
        
#         # Save best model
#         if avg_val_iou > best_val_iou:
#             best_val_iou = avg_val_iou
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_iou': avg_val_iou,
#             }, os.path.join(output_dir, 'best_model.pth'))
#             logger.info(f"Saved best model with IoU: {avg_val_iou:.4f}")
        
#         # Plot training curves
#         if (epoch + 1) % 5 == 0:
#             plot_training_curves(history, output_dir, epoch + 1)
    
#     return model, history

# def plot_training_curves(history, output_dir, epoch):
#     """
#     Plot and save training curves
#     """
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'], label='Train')
#     plt.plot(history['val_loss'], label='Validation')
#     plt.title('Loss')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history['train_iou'], label='Train')
#     plt.plot(history['val_iou'], label='Validation')
#     plt.title('IoU')
#     plt.legend()
    
#     plt.savefig(os.path.join(output_dir, f'training_curves_epoch_{epoch}.png'))
#     plt.close()

# def main():
#     # Configuration
#     config = {
#         'batch_size': 8,
#         'num_epochs': 100,
#         'learning_rate': 1e-4,
#         'weight_decay': 0.01,
#         'patch_size': 16,
#         'embed_dim': 768,
#         'num_heads': 12,
#         'depth': 12,
#     }
    
#     # Create output directory
#     output_dir = f"training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Set device
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")
    
#     # Create datasets
#     train_dataset = BurnScarDataset('/Users/admin63/Python-Programs/Geospatial-Image-Stitcher-and-cloud-remover/custom-transformer-inferences/hls_burn_scars/training')
#     val_dataset = BurnScarDataset('/Users/admin63/Python-Programs/Geospatial-Image-Stitcher-and-cloud-remover/custom-transformer-inferences/hls_burn_scars/validation')
    
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
#     # Initialize model
#     model = TransformerSegmentation(
#         in_channels=3,
#         num_classes=2,
#         patch_size=config['patch_size'],
#         embed_dim=config['embed_dim'],
#         depth=config['depth'],
#         num_heads=config['num_heads']
#     ).to(device)
    
#     # Setup training
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=config['learning_rate'],
#         weight_decay=config['weight_decay']
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=config['num_epochs']
#     )
    
#     # Train model
#     try:
#         model, history = train_model(
#             model=model,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             criterion=criterion,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             num_epochs=config['num_epochs'],
#             device=device,
#             output_dir=output_dir
#         )
        
#         # Save final model and training history
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'history': history,
#             'config': config
#         }, os.path.join(output_dir, 'final_model.pth'))
        
#         logger.info("Training completed successfully!")
        
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()
####################################
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
import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, output_dir):
    """
    Training function with validation and improved error handling
    """
    best_val_iou = 0
    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
    
    # Verify data loaders
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty")
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty")
        
    logger.info(f"Training with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metrics = {
            'loss': 0.0,
            'iou': 0.0,
            'batch_count': 0,
            'sample_count': 0
        }
        
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks) in enumerate(train_iterator):
            try:
                if images.size(0) == 0:
                    logger.warning(f"Skipping empty batch {batch_idx}")
                    continue
                    
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                # Calculate IoU
                predictions = outputs.argmax(dim=1)
                intersection = torch.logical_and(masks, predictions).float().sum()
                union = torch.logical_or(masks, predictions).float().sum()
                iou = (intersection / (union + 1e-6)).item()
                
                train_metrics['loss'] += loss.item() * images.size(0)
                train_metrics['iou'] += iou * images.size(0)
                train_metrics['batch_count'] += 1
                train_metrics['sample_count'] += images.size(0)
                
                # Update progress bar
                train_iterator.set_postfix({
                    'loss': train_metrics['loss'] / max(train_metrics['sample_count'], 1),
                    'iou': train_metrics['iou'] / max(train_metrics['sample_count'], 1)
                })
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        if train_metrics['batch_count'] == 0:
            raise RuntimeError("No training batches were processed successfully")
            
        avg_train_loss = train_metrics['loss'] / train_metrics['sample_count']
        avg_train_iou = train_metrics['iou'] / train_metrics['sample_count']
        
        # Validation phase
        model.eval()
        val_metrics = {
            'loss': 0.0,
            'iou': 0.0,
            'batch_count': 0,
            'sample_count': 0
        }
        
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc='Validation')
            for batch_idx, (images, masks) in enumerate(val_iterator):
                try:
                    if images.size(0) == 0:
                        continue
                        
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    predictions = outputs.argmax(dim=1)
                    intersection = torch.logical_and(masks, predictions).float().sum()
                    union = torch.logical_or(masks, predictions).float().sum()
                    iou = (intersection / (union + 1e-6)).item()
                    
                    val_metrics['loss'] += loss.item() * images.size(0)
                    val_metrics['iou'] += iou * images.size(0)
                    val_metrics['batch_count'] += 1
                    val_metrics['sample_count'] += images.size(0)
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        if val_metrics['batch_count'] == 0:
            raise RuntimeError("No validation batches were processed successfully")
            
        avg_val_loss = val_metrics['loss'] / val_metrics['sample_count']
        avg_val_iou = val_metrics['iou'] / val_metrics['sample_count']
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1} - "
            f"Train Loss: {avg_train_loss:.4f} ({train_metrics['sample_count']} samples), "
            f"Train IoU: {avg_train_iou:.4f}, "
            f"Val Loss: {avg_val_loss:.4f} ({val_metrics['sample_count']} samples), "
            f"Val IoU: {avg_val_iou:.4f}"
        )
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_iou'].append(avg_train_iou)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        
        # Save best model
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': avg_val_iou,
                'history': history,
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"Saved best model with IoU: {avg_val_iou:.4f}")
        
        # Plot training curves
        if (epoch + 1) % 5 == 0:
            plot_training_curves(history, output_dir, epoch + 1)
    
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
    }
    
    # Create output directory with timestamp
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
    
    # Log configuration
    logger.info(f"Starting training with config: {config}")
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Create datasets
        train_dataset = BurnScarDataset(
            '/Users/admin63/Python-Programs/Geospatial-Image-Stitcher-and-cloud-remover/custom-transformer-inferences/hls_burn_scars/training'
        )
        val_dataset = BurnScarDataset(
            '/Users/admin63/Python-Programs/Geospatial-Image-Stitcher-and-cloud-remover/custom-transformer-inferences/hls_burn_scars/validation'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0  # Increase if needed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0  # Increase if needed
        )
        
        # Initialize model
        model = TransformerSegmentation(
            in_channels=6,
            num_classes=2,
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads']
        ).to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs']
        )
        
        # Train model
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
        
        # Save final model
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
    main()