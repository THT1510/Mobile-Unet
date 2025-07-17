# W&B Configuration for Brain Tumor Segmentation
import wandb

def init_wandb(project_name="Brain tumor segmentation", entity="baohtse183146-fpt-university"):
    """
    Initialize Weights & Biases tracking
    """
    
    # Optional: Set custom wandb directory FIRST
    import os
    os.makedirs("./logs", exist_ok=True)
    os.environ["WANDB_DIR"] = "./logs"
    
    # Check if already logged in, if not then login
    try:
        wandb.login(key="c6d810ee5bdf998ffc951ff2f6cd758d2919ad8c")
    except Exception as e:
        print(f"Login warning (can be ignored): {e}")
    
    # Finish any existing runs to avoid conflicts
    try:
        wandb.finish()
    except:
        pass
    
    # Define hyperparameters
    hyperparams = {
        'epochs': 150,
        'batch_size': 16,
        'lr': 0.0025,
        'img_size': 256,
        'model': 'MobileNetUNet',
        'backbone': 'mobilenet_v2',
        'num_classes': 2,
        'patience': 100,
        'seg_weight': 1.0,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'loss_function': 'CrossEntropyLoss',
        'weight_decay': 0.0001
    }
    
    # Initialize W&B run
    run = wandb.init(
        project=project_name,
        entity=entity,
        config=hyperparams,
        name="MobileUNet",
        tags=["brain-tumor", "segmentation", "mobile-unet", "medical-imaging"]
    )
    
    return run, wandb.config

def log_model_summary(model, input_size=(1, 1, 256, 256)):
    """
    Log model architecture summary to W&B
    """
    import torch
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log model info
    wandb.log({
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "model_input_size": input_size
    })
    
    # Create dummy input for model visualization
    dummy_input = torch.randn(input_size)
    
    try:
        # Log model graph (optional)
        wandb.watch(model, log="all", log_freq=10)
        print("Model watching enabled in W&B")
    except Exception as e:
        print(f"Could not enable model watching: {e}")

def finish_wandb():
    """
    Finish W&B run
    """
    wandb.finish()
