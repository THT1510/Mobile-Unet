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
        self._load_data()

    def _load_data(self):
        """Load image and mask file pairs from the dataset directories"""
        if not os.path.exists(self.images_dir):
            print(f"Warning: Images directory not found: {self.images_dir}")
            return
        
        if not os.path.exists(self.masks_dir):
            print(f"Warning: Masks directory not found: {self.masks_dir}")
            return
        
        # Check if the directory structure includes class subdirectories
        if os.path.isdir(os.path.join(self.images_dir, '0')):
            # Structure: dataset_split/train/image/0/, dataset_split/train/image/1/, etc.
            self._load_data_with_classes()
        else:
            # Structure: dataset_split/train/image/, dataset_split/train/mask/ (flat structure)
            self._load_data_flat()
    
    def _load_data_with_classes(self):
        """Load data when organized in class subdirectories"""
        # Binary classification: 0 = no tumor, 1 = tumor
        class_dirs = ['0', '1']
        
        for class_name in class_dirs:
            img_class_dir = os.path.join(self.images_dir, class_name)
            mask_class_dir = os.path.join(self.masks_dir, class_name)
            
            if os.path.exists(img_class_dir) and os.path.exists(mask_class_dir):
                img_files = sorted([f for f in os.listdir(img_class_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                for img_file in img_files:
                    img_path = os.path.join(img_class_dir, img_file)
                    
                    # Create mask filename by adding '_m' before the extension
                    name_without_ext = os.path.splitext(img_file)[0]
                    ext = os.path.splitext(img_file)[1]
                    mask_file = name_without_ext + '_m' + ext
                    mask_path = os.path.join(mask_class_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.image_mask_pairs.append((img_path, mask_path))
                    else:
                        # Try original filename as fallback
                        fallback_mask_path = os.path.join(mask_class_dir, img_file)
                        if os.path.exists(fallback_mask_path):
                            self.image_mask_pairs.append((img_path, fallback_mask_path))
    
    def _load_data_flat(self):
        """Load data from flat directory structure"""
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            return
        
        img_files = sorted([f for f in os.listdir(self.images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_file in img_files:
            img_path = os.path.join(self.images_dir, img_file)
            
            # Create mask filename by adding '_m' before the extension
            name_without_ext = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            mask_file = name_without_ext + '_m' + ext
            mask_path = os.path.join(self.masks_dir, mask_file)
            
            if os.path.exists(mask_path):
                self.image_mask_pairs.append((img_path, mask_path))
            else:
                # Try original filename as fallback
                fallback_mask_path = os.path.join(self.masks_dir, img_file)
                if os.path.exists(fallback_mask_path):
                    self.image_mask_pairs.append((img_path, fallback_mask_path))

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
    
    # Print dataset info
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty! Please check the data directory: {root_dir}")
    
    if len(val_dataset) == 0:
        print(f"Warning: Validation dataset is empty! Using train dataset for validation.")
        val_dataset = train_dataset
    
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
