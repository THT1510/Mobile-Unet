import torch
import torch.nn as nn

class SegmentationMetrics:
    def __init__(self, smooth=1.0, threshold=0.5):
        self.smooth = smooth
        self.threshold = threshold

    def dice_coef_metric(self, pred, label):
        # pred shape: [B, C, H, W], label shape: [B, H, W]
        pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
        label_one_hot = torch.zeros_like(pred)  # Create one-hot encoded labels
        label_one_hot.scatter_(1, label.unsqueeze(1), 1)  # Convert to one-hot encoding
        
        intersection = (pred * label_one_hot).sum(dim=(2, 3))  # Calculate intersection
        denominator  = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3))  # Calculate denominator
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)  # Calculate Dice coefficient
        
        return dice.mean()  # Average over batch and classes

    def iou(self, pred, label):
        pred = torch.softmax(pred, dim=1)
        label_one_hot = torch.zeros_like(pred)
        label_one_hot.scatter_(1, label.unsqueeze(1), 1)
        
        intersection = (pred * label_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return iou.mean()


class ClassificationMetrics:
    def accuracy(self, pred, label):
        pred_cls = torch.argmax(pred, dim=1)
        accuracy = (pred_cls == label).float().mean()
        return accuracy * 100

    def f1_score_cls(self, pred, label):
        pred_cls = torch.argmax(pred, dim=1)
        f1_scores = []
        
        for cls_idx in range(pred.size(1)):
            true_pos = ((pred_cls == cls_idx) & (label == cls_idx)).sum().float()
            false_pos = ((pred_cls == cls_idx) & (label != cls_idx)).sum().float()
            false_neg = ((pred_cls != cls_idx) & (label == cls_idx)).sum().float()
            
            precision = true_pos / (true_pos + false_pos + 1e-6)
            recall = true_pos / (true_pos + false_neg + 1e-6)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            f1_scores.append(f1)
        
        return torch.stack(f1_scores).mean()


class DualTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        super(DualTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_criterion = self.dice_loss         # Use Dice Loss for segmentation
        self.cls_criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification

    def dice_loss(self, pred, target):
        pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
        target_one_hot = torch.zeros_like(pred)  # Create one-hot encoded labels
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)  # Convert to one-hot encoding
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))  # Calculate intersection
        denominator = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # Calculate denominator
        dice = (2. * intersection + 1e-6) / (denominator + 1e-6)  # Calculate Dice coefficient
        
        return 1 - dice.mean()  # Return Dice Loss (1 - Dice Coefficient)

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.dice_loss(seg_pred, seg_target)  # Calculate Dice Loss for segmentation
        cls_loss = self.cls_criterion(cls_pred, cls_target)  # Calculate CrossEntropyLoss for classification
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss  # Combine losses


def create_metrics():
    seg_metrics = SegmentationMetrics()
    cls_metrics = ClassificationMetrics()

    metrics = {
        'seg_dice': seg_metrics.dice_coef_metric,
        'seg_iou': seg_metrics.iou,
        'cls_accuracy': cls_metrics.accuracy,
    }
    
    return metrics
