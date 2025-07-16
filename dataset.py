# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=(256, 256)):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        # Get paths - adjust based on your actual folder structure
        self.images_dir = os.path.join(root_dir, split, 'image')
        self.masks_dir = os.path.join(root_dir, split, 'mask')
        
        # Create list of (image_path, mask_path) tuples
        self.image_mask_pairs = []

    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('L')  # Grayscale image
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        
        # Resize images
        image = image.resize(self.img_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.img_size, Image.Resampling.NEAREST)
        
        # Convert to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Convert mask to binary: 0 for background, 1 for any tumor
        if mask_np.max() > 0:  # If mask is not completely black
            # Any non-zero pixel becomes tumor (class 1)
            mask_np = (mask_np > 0).astype(np.int64)
        else:
            # All pixels are background (class 0)
            mask_np = mask_np.astype(np.int64)
        
        # Ensure mask only contains 0 and 1
        mask_np = np.clip(mask_np, 0, 1)
        
        # Convert to tensors
        image = torch.from_numpy(image_np).float() / 255.0
        image = image.unsqueeze(0)  # Add channel dimension (1, H, W)
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        mask = torch.from_numpy(mask_np).long()  # Ensure mask is Long tensor
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask


def get_data_loaders(root_dir, batch_size=8, num_workers=4):
    """Create data loaders for train and validation sets"""
    # Create datasets
    train_dataset = SegmentationDataset(
        root_dir=root_dir,
        split='train',
        img_size=(256, 256)
    )
    
    val_dataset = SegmentationDataset(
        root_dir=root_dir,
        split='val',
        img_size=(256, 256)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
