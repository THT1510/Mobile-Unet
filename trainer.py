import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
class ModelTrainer:
    def __init__(
        self, 
        model, 
        dataloaders, 
        criterion_seg,
        criterion_cls,
        optimizer, 
        scheduler=None,
        metrics=None,
        device=None,
        patience=5,
        task_weights={'seg': 1.0, 'cls': 1.0},
        wandb_logger=None   
    ):
        # Model and device setup
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Training configuration
        self.dataloaders = dataloaders
        self.criterion_seg = criterion_seg
        self.criterion_cls = criterion_cls
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}
        
        # Set default task weights if not provided
        if task_weights is None:
            self.task_weights = {'seg': 1.0, 'cls': 1.0}
        else:
            self.task_weights = task_weights

        # Early stopping parameters
        self.patience = patience
        self.best_score = float('inf')
        self.counter = 0
        
        # W&B logger
        self.wandb_logger = wandb_logger

        # Training history tracking
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_seg_dice': [],
            'val_seg_dice': [],
            'train_cls_accuracy': [],
            'val_cls_accuracy': []
        }

    def _run_epoch(self, phase):
        is_train = phase == 'train'
        self.model.train() if is_train else self.model.eval()
        
        running_loss = {'total': 0.0, 'seg': 0.0, 'cls': 0.0}
        running_metrics = {k: 0.0 for k in self.metrics.keys()}
        
        # Create progress bar for batches
        total_batches = len(self.dataloaders[phase])
        pbar = tqdm(
            enumerate(self.dataloaders[phase]),
            total=total_batches,
            desc=f'{total_batches}/{total_batches}',
            bar_format='{desc} {bar} {postfix}',
            leave=True
        )
        
        with torch.set_grad_enabled(is_train):
            for batch_idx, batch_data in pbar:
                # Handle both segmentation-only and multi-task cases
                if len(batch_data) == 2:  # Segmentation only
                    inputs, masks = batch_data
                    labels = None
                else:  # Multi-task (segmentation + classification)
                    inputs, masks, labels = batch_data
                
                # Move data to device
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                # Forward pass
                model_outputs = self.model(inputs)
                
                # Handle different output formats
                if isinstance(model_outputs, tuple):
                    seg_outputs, cls_outputs = model_outputs
                else:
                    seg_outputs = model_outputs
                    cls_outputs = None
                
                # Calculate losses
                loss_seg = self.criterion_seg(seg_outputs, masks)
                
                if cls_outputs is not None and labels is not None and self.criterion_cls is not None:
                    loss_cls = self.criterion_cls(cls_outputs, labels)
                    loss_total = (
                        self.task_weights['seg'] * loss_seg + 
                        self.task_weights['cls'] * loss_cls
                    )
                else:
                    loss_cls = torch.tensor(0.0, device=self.device)
                    loss_total = self.task_weights['seg'] * loss_seg
                
                # Backward pass and optimization
                if is_train:
                    self.optimizer.zero_grad()
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Update running losses
                running_loss['total'] += loss_total.item()
                running_loss['seg'] += loss_seg.item()
                running_loss['cls'] += loss_cls.item()
                
                # Calculate metrics
                for name, metric_fn in self.metrics.items():
                    if name.startswith('seg_'):
                        value = metric_fn(seg_outputs, masks).item()
                        running_metrics[name] += value
                    elif cls_outputs is not None and labels is not None:
                        value = metric_fn(cls_outputs, labels).item()
                        running_metrics[name] += value
                
                # Calculate averages - only include metrics that have been updated
                avg_metrics = {
                    'loss': running_loss['total'] / (batch_idx + 1),
                    'seg_loss': running_loss['seg'] / (batch_idx + 1),
                    'cls_loss': running_loss['cls'] / (batch_idx + 1),
                }
                
                # Only add metrics that have been calculated (non-zero values)
                for k, v in running_metrics.items():
                    if v > 0 or k.startswith('seg_'):  # Always include segmentation metrics
                        avg_metrics[k] = v / (batch_idx + 1)
                
                # Update progress bar
                batch_time = pbar.format_dict["elapsed"] / (batch_idx + 1)
                current_batch = batch_idx + 1
                
                # Format with more decimal places
                postfix_items = []
                postfix_items.append(f"{batch_time*1000:.0f}ms/step")
                
                # Add segmentation metrics if available
                if 'seg_dice' in avg_metrics:
                    postfix_items.append(f"dice_coefficient: {avg_metrics['seg_dice']:.6f}")
                if 'seg_iou' in avg_metrics:
                    postfix_items.append(f"iou: {avg_metrics['seg_iou']:.6f}")
                
                # Add classification metrics if available
                if 'cls_accuracy' in avg_metrics:
                    postfix_items.append(f"accuracy: {avg_metrics['cls_accuracy']:.6f}")
                
                # Always show loss
                postfix_items.append(f"loss: {avg_metrics['loss']:.6f}")
                
                postfix = " - ".join(postfix_items)
                
                pbar.set_description(f"{current_batch}/{total_batches}")
                pbar.set_postfix_str(postfix)
        
        # Calculate final epoch metrics
        num_batches = len(self.dataloaders[phase])
        epoch_metrics = {
            'loss_total': running_loss['total'] / num_batches,
            'loss_seg': running_loss['seg'] / num_batches,
            'loss_cls': running_loss['cls'] / num_batches,
            **{k: v / num_batches for k, v in running_metrics.items()}
        }
        
        # Print validation metrics with more decimal places
        if phase == 'val':
            # Build validation metrics string dynamically
            val_metrics_str = f"val_loss: {epoch_metrics['loss_total']:.4f} - "
            val_metrics_str += f"val_dice: {epoch_metrics.get('seg_dice', 0):.4f} - "
            val_metrics_str += f"val_iou: {epoch_metrics.get('seg_iou', 0):.4f}"
            
            # Only add accuracy if it exists in metrics
            if 'cls_accuracy' in epoch_metrics:
                val_metrics_str = f"val_accuracy: {epoch_metrics['cls_accuracy']:.4f} - " + val_metrics_str
            
            print(f"- {val_metrics_str}")
        
        return epoch_metrics

    def train(self, num_epochs, save_path='best_model.pt'):
        """Training loop with modified progress display"""
        # Create directory for save_path if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:  # If save_path contains a directory path
            os.makedirs(save_dir, exist_ok=True)
        
        # Create results directory
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize best_score for dice (higher is better)
        self.best_score = float('-inf')

        try:
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                
                # Training and validation
                train_metrics = self._run_epoch('train')
                val_metrics = self._run_epoch('val')
                
                # ✅ Log metrics to W&B
                if self.wandb_logger:
                    self.wandb_logger.log({
                        "epoch": epoch + 1,
                        "train_loss": train_metrics['loss_total'],
                        "train_dice": train_metrics.get('seg_dice', 0),
                        "train_iou": train_metrics.get('seg_iou', 0),
                        "val_loss": val_metrics['loss_total'],
                        "val_dice": val_metrics.get('seg_dice', 0),
                        "val_iou": val_metrics.get('seg_iou', 0),
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                
                # Update training history
                self._update_history(train_metrics, val_metrics)
                
                # Early stopping check using dice coefficient
                if self._check_early_stopping(val_metrics['seg_dice'], save_path):
                    print("\nEarly stopping triggered!")
                    break
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_metrics['seg_dice'])
            
            # Generate and save training progress plot
            self._plot_training_progress()
            # REMOVE THESE TWO LINES:
            # plt.savefig(os.path.join(results_dir, 'training_progress.png'))
            # plt.close()  # Close the plot to free memory
            
            # Load best model with error handling
            if os.path.exists(save_path):
                self.model.load_state_dict(
                    torch.load(save_path, weights_only=True)
                )
                print(f"\nLoaded best model from {save_path}")
            else:
                print(f"\nWarning: No best model found at {save_path}, using current model state")
                
            return self.model
        
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            import traceback
            traceback.print_exc()  # Add this for better error reporting
            if os.path.exists(save_path):
                print(f"Loading last saved model from {save_path}")
                self.model.load_state_dict(
                    torch.load(save_path, weights_only=True)
                )
            return self.model



    def _update_history(self, train_metrics, val_metrics):
        """Update training history with explicit error handling"""
        try:
            # Make sure keys exist
            if 'loss_total' not in train_metrics or 'loss_total' not in val_metrics:
                print("Warning: Missing 'loss_total' in metrics")
                
            # Update with explicit keys and default values
            self.train_history['train_loss'].append(train_metrics.get('loss_total', 0))
            self.train_history['val_loss'].append(val_metrics.get('loss_total', 0))
            self.train_history['train_seg_dice'].append(train_metrics.get('seg_dice', 0))
            self.train_history['val_seg_dice'].append(val_metrics.get('seg_dice', 0))
            self.train_history['train_cls_accuracy'].append(train_metrics.get('cls_accuracy', 0))
            self.train_history['val_cls_accuracy'].append(val_metrics.get('cls_accuracy', 0))
            
            # Confirmation
            print(f"Updated history - train_loss: {train_metrics.get('loss_total', 0):.4f}, "
                f"val_loss: {val_metrics.get('loss_total', 0):.4f}")
        
        except Exception as e:
            print(f"Error updating history: {str(e)}")

    def _plot_training_progress(self):
        """Plot training progress with better error handling"""
        try:
            # Check if history is empty
            if not self.train_history['train_loss']:
                print("Warning: Training history is empty. Nothing to plot.")
                return
                
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Loss subplot with explicit data
            plt.subplot(2, 2, 1)
            epochs = range(1, len(self.train_history['train_loss'])+1)
            plt.plot(epochs, self.train_history['train_loss'], 'bo-', label='Train Loss')
            plt.plot(epochs, self.train_history['val_loss'], 'ro-', label='Validation Loss')
            plt.title('Training and Validation Loss', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True)
            plt.legend(fontsize=12)
            
            # Dice Coefficient subplot
            plt.subplot(2, 2, 2)
            plt.plot(epochs, self.train_history['train_seg_dice'], 'bo-', label='Train Dice')
            plt.plot(epochs, self.train_history['val_seg_dice'], 'ro-', label='Validation Dice')
            plt.title('Segmentation Dice Coefficient', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Dice Coefficient', fontsize=12)
            plt.grid(True)
            plt.legend(fontsize=12)
            
            # Classification Accuracy subplot
            plt.subplot(2, 2, 3)
            plt.plot(epochs, self.train_history['train_cls_accuracy'], 'bo-', label='Train Accuracy')
            plt.plot(epochs, self.train_history['val_cls_accuracy'], 'ro-', label='Validation Accuracy')
            plt.title('Classification Accuracy', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.grid(True)
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            
            # Ensure the results directory exists
            os.makedirs('results', exist_ok=True)
            
            # Save with high DPI
            save_path = os.path.join('results', 'training_progress.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nTraining plot saved to: {save_path}")
            
            # Close the plot to free memory
            plt.close()
        
        except Exception as e:
            print(f"Error generating plot: {str(e)}")
            import traceback
            traceback.print_exc()


    # def _check_early_stopping(self, current_score, save_path):
    #     """
    #     Check for early stopping and save best model
    #     """
    #     if current_score < self.best_score:
    #         self.best_score = current_score
    #         self.counter = 0
    #         torch.save(self.model.state_dict(), save_path)
    #         return False
        
    #     self.counter += 1
    #     return self.counter >= self.patience

    def _check_early_stopping(self, current_dice, save_path):
        """
        Check for early stopping and save best model based on validation dice coefficient
        Args:
            val_metrics: Dictionary containing validation metrics
            save_path: Path to save the model
        """
        # If dice coefficient improved (increased)
        if current_dice > self.best_score:
            improvement = current_dice - self.best_score
            self.best_score = current_dice
            self.counter = 0
            
            # Save model
            torch.save(self.model.state_dict(), save_path)
            print(f"\nEpoch: val_dice_coefficient improved from {self.best_score-improvement:.4f} to {self.best_score:.4f}, saving model to {save_path}")
            
            # Log best model to W&B
            if self.wandb_logger:
                self.wandb_logger.log({"best_val_dice": self.best_score})
                
            return False
        
        # If no improvement
        self.counter += 1
        return self.counter >= self.patience