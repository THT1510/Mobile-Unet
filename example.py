import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import create_segmentation_metrics
from src.model import MobileNetUNet
from dataset import get_data_loaders
from trainer import ModelTrainer
from pathlib import Path
import wandb
from wandb_config import init_wandb, log_model_summary, finish_wandb

def main():
    # ✅ W&B setup using config file
    run, config = init_wandb()
    
    # Configuration
    CONFIG = {
        'data_dir': r"d:\FPT BT\DSP391\train_model\Mobile-unet\dataset_split",
        'batch_size': config.batch_size,
        'num_workers': 4,  # Reduced from 5 to 4 to avoid warning
        'learning_rate': config.lr,
        'num_epochs': config.epochs,
        'patience': config.patience,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seg_weight': config.seg_weight,  # Only segmentation weight needed
        'save_dir': 'checkpoint_new'  
    }

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    save_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')

    # 1. Initialize model
    model = MobileNetUNet(                         
        img_ch=1,
        seg_ch=2,  # Binary segmentation: background + tumor
        num_classes=None  # No classification needed
    ).to(CONFIG['device'])

    # 2. Create data loaders
    train_loader, val_loader = get_data_loaders(
        root_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    # Check the number of samples in the data loaders
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    
    # ✅ Log model summary to W&B
    log_model_summary(model, input_size=(1, 1, 256, 256))

    # 3. Setup metrics
    metrics = create_segmentation_metrics()

    # 4. Initialize optimizer and loss functions
    optimizer = Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )
    
    # Only segmentation loss needed
    criterion = nn.CrossEntropyLoss()
    
    # 5. Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=10,
        factor=0.5,
        verbose=False,  # Changed from True to False to avoid deprecation warning
        min_lr=0.0001
    )

    # 6. Initialize trainer (simplified for segmentation only)
    trainer = ModelTrainer(
        model=model,
        dataloaders={
            'train': train_loader,
            'val': val_loader
        },
        criterion_seg=criterion,  # Use criterion directly
        criterion_cls=None,       # No classification loss
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=CONFIG['device'],
        patience=CONFIG['patience'],
        task_weights={
            'seg': CONFIG['seg_weight']
        },
        wandb_logger=wandb  # Add wandb logger
    )

    # 7. Train the model
    model = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_path=save_path
    )
    
    # Save final model
    final_save_path = os.path.join(CONFIG['save_dir'], 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': CONFIG
    }, final_save_path)

    # ✅ Save model to W&B
    wandb.save(save_path)
    wandb.save(final_save_path)
    
    print("Training completed successfully!")
    finish_wandb()

if __name__ == "__main__":
    main()