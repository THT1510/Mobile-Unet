import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.model import MobileNetUNet
import torchvision.transforms as transforms
from skimage import measure
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter
from skimage.measure import approximate_polygon
from scipy.interpolate import splprep, splev
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import shutil

def load_model(model_path, device):
    """Load model from checkpoint"""
    print(f"Using device: {device}")
    model = MobileNetUNet(img_ch=1, seg_ch=4, num_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model

def load_and_preprocess_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize((256, 256))
    return img, np.array(img)

def get_model_prediction(model, img, device):
    """Get model prediction for an image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        seg_output, cls_output = model(img_tensor)
        seg_pred = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
        cls_pred = torch.argmax(cls_output, dim=1).item()
        cls_probs = torch.softmax(cls_output, dim=1).squeeze().cpu().numpy()
    
    return seg_pred, cls_pred, cls_probs

def process_masks_and_predict(model_path, mask_folder, output_base_dir, device='cuda'):
    """Process all masks and predict corresponding images"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    class_names = ['No_Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    predicted_dirs = {}
    
    # Create output directories for each class
    for class_name in class_names:
        class_dir = os.path.join(output_base_dir, 'predicted', class_name)
        os.makedirs(class_dir, exist_ok=True)
        predicted_dirs[class_name] = class_dir
    
    predictions = {}
    
    # Process each mask file
    for mask_file in os.listdir(mask_folder):
        if mask_file.endswith(('.png', '.jpg', '.jpeg')):
            mask_path = os.path.join(mask_folder, mask_file)
            # Assuming image has same name but different extension
            image_file = os.path.splitext(mask_file)[0] + '.jpg'
            image_path = os.path.join(mask_folder, image_file)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for mask {mask_file}")
                continue
                
            # Load and predict
            img, img_np = load_and_preprocess_image(image_path)
            seg_pred, cls_pred, cls_probs = get_model_prediction(model, img, device)
            
            # Save prediction
            class_name = class_names[cls_pred]
            output_path = os.path.join(predicted_dirs[class_name], image_file)
            Image.fromarray(img_np).save(output_path)
            
            predictions[image_file] = {
                'cls_pred': cls_pred,
                'seg_pred': seg_pred,
                'mask_path': mask_path,
                'image_path': image_path
            }
            print(f"Processed: {image_file} -> {class_name}")
    
    return predictions, predicted_dirs

def dice_coef_metric(pred_mask, gt_mask, cls_pred, smooth=1.0):
    """Calculate Dice coefficient"""
    pred_mask = pred_mask.astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)
    
    if cls_pred == 0:
        if np.all(gt_mask == 0):
            return 1.0
        pred_bg = (pred_mask == 0)
        gt_bg = (gt_mask == 0)
        intersection = np.sum(pred_bg & gt_bg)
        denominator = np.sum(pred_bg) + np.sum(gt_bg)
        return (2. * intersection + smooth) / (denominator + smooth)
    
    pred_c = (pred_mask == cls_pred)
    gt_c = (gt_mask == cls_pred)
    intersection = np.sum(pred_c & gt_c)
    denominator = np.sum(pred_c) + np.sum(gt_c)
    
    if denominator > 0:
        return (2. * intersection + smooth) / (denominator + smooth)
    return 0.0

def compare_predictions(predictions, mask_folder, output_base_dir):
    """Compare predictions with ground truth"""
    class_names = ['No_Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    class_counts = {name: {'correct': 0, 'total': 0, 'dice_sum': 0.0} for name in class_names}
    
    for image_file, pred_data in predictions.items():
        # Load ground truth mask
        gt_mask = Image.open(pred_data['mask_path']).convert('L').resize((256, 256))
        gt_mask_np = np.array(gt_mask)
        if gt_mask_np.max() > 3:
            gt_mask_np = (gt_mask_np / 255 * 3).astype(np.uint8)
            
        # Determine ground truth class (assuming highest class value indicates tumor type)
        gt_class = 0 if np.all(gt_mask_np == 0) else np.max(gt_mask_np)
        
        # Update statistics
        pred_class = pred_data['cls_pred']
        class_counts[class_names[gt_class]]['total'] += 1
        if gt_class == pred_class:
            class_counts[class_names[gt_class]]['correct'] += 1
        
        # Calculate Dice score
        dice_score = dice_coef_metric(pred_data['seg_pred'], gt_mask_np, pred_class)
        class_counts[class_names[gt_class]]['dice_sum'] += dice_score
    
    # Print results
    print("\nClassification Results:")
    print("-" * 50)
    total_correct = 0
    total_images = 0
    
    for class_name, stats in class_counts.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        avg_dice = stats['dice_sum'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{class_name}:")
        print(f"  Accuracy: {stats['correct']}/{stats['total']} ({accuracy:.2%})")
        print(f"  Average Dice: {avg_dice:.4f}")
        total_correct += stats['correct']
        total_images += stats['total']
    
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print("-" * 50)
    print(f"Overall Accuracy: {total_correct}/{total_images} ({overall_accuracy:.2%})")
    
    # Save results to file
    with open(os.path.join(output_base_dir, 'prediction_results.txt'), 'w') as f:
        f.write("Classification Results:\n")
        f.write("-" * 50 + "\n")
        for class_name, stats in class_counts.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            avg_dice = stats['dice_sum'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"{class_name}:\n")
            f.write(f"  Accuracy: {stats['correct']}/{stats['total']} ({accuracy:.2%})\n")
            f.write(f"  Average Dice: {avg_dice:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall Accuracy: {total_correct}/{total_images} ({overall_accuracy:.2%})\n")

if __name__ == "__main__":
    # Paths
    model_path = "checkpoint_new/best_model.pt"
    mask_folder = "test/masks"
    output_base_dir = r"D:\FPT BT\DBM\final project\train_model\results_predict"
    
    # Process masks and predict
    predictions, predicted_dirs = process_masks_and_predict(model_path, mask_folder, output_base_dir)
    
    # Compare predictions with ground truth
    compare_predictions(predictions, mask_folder, output_base_dir)