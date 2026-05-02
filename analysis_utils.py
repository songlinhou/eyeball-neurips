import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import cv2


def visualize_attention_maps(model, video, device, save_path=None):
    """
    Visualize temporal and spatial attention maps
    Only works with ImprovedResNet3D that has attention modules
    """
    model.eval()
    
    # Register hooks to capture attention weights
    attention_maps = {}
    
    def get_activation(name):
        def hook(model, input, output):
            attention_maps[name] = output.detach()
        return hook
    
    # Register hooks
    if hasattr(model, 'temporal_attention'):
        model.temporal_attention.register_forward_hook(get_activation('temporal'))
    if hasattr(model, 'cbam'):
        model.cbam.register_forward_hook(get_activation('cbam'))
    
    # Forward pass
    with torch.no_grad():
        video = video.unsqueeze(0).to(device)
        output = model(video)
    
    # Visualize temporal attention
    if 'temporal' in attention_maps:
        temporal_att = attention_maps['temporal']
        # Average over spatial dimensions: (B, C, T, H, W) -> (T,)
        temporal_weights = temporal_att.mean(dim=[0, 1, 3, 4]).cpu().numpy()
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(temporal_weights)
        plt.xlabel('Frame Index')
        plt.ylabel('Attention Weight')
        plt.title('Temporal Attention Weights')
        plt.grid(True)
        
        # Visualize spatial attention
        if 'cbam' in attention_maps:
            spatial_att = attention_maps['cbam']
            # Average over time and channels: (B, C, T, H, W) -> (H, W)
            spatial_weights = spatial_att.mean(dim=[0, 1, 2]).cpu().numpy()
            
            plt.subplot(1, 2, 2)
            plt.imshow(spatial_weights, cmap='hot')
            plt.colorbar()
            plt.title('Spatial Attention Map')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    return attention_maps


def plot_confusion_matrix(y_true, y_pred, class_names=['Intact', 'Detached'], 
                         save_path=None):
    """
    Plot confusion matrix with percentages
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Percentages)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_roc_curve(y_true, y_probs, save_path=None):
    """
    Plot ROC curve and calculate AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return roc_auc, fpr, tpr, thresholds


def analyze_misclassifications(model, dataloader, device, num_examples=5):
    """
    Analyze and visualize misclassified examples
    """
    model.eval()
    
    misclassified = []
    
    with torch.no_grad():
        for videos, labels, masks in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Find misclassified samples
            incorrect = predicted != labels
            
            for i in range(len(labels)):
                if incorrect[i]:
                    misclassified.append({
                        'video': videos[i].cpu(),
                        'true_label': labels[i].item(),
                        'pred_label': predicted[i].item(),
                        'confidence': probs[i, predicted[i]].item()
                    })
                    
                    if len(misclassified) >= num_examples:
                        break
            
            if len(misclassified) >= num_examples:
                break
    
    # Visualize misclassified examples
    class_names = ['Intact', 'Detached']
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    
    for idx, example in enumerate(misclassified[:num_examples]):
        video = example['video']  # (C, T, H, W)
        true_label = example['true_label']
        pred_label = example['pred_label']
        confidence = example['confidence']
        
        # Show 4 frames from the video
        num_frames = video.shape[1]
        frame_indices = [0, num_frames//3, 2*num_frames//3, num_frames-1]
        
        for j, frame_idx in enumerate(frame_indices):
            frame = video[:, frame_idx, :, :].permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0, 1)
            
            if num_examples == 1:
                ax = axes[j]
            else:
                ax = axes[idx, j]
            
            ax.imshow(frame)
            ax.axis('off')
            
            if j == 0:
                ax.set_title(f'True: {class_names[true_label]}\n'
                           f'Pred: {class_names[pred_label]} ({confidence:.2%})',
                           fontsize=10)
            else:
                ax.set_title(f'Frame {frame_idx}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return misclassified


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over epochs
    
    history should be a dict with keys: 
    'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    if 'train_f1' in history and 'val_f1' in history:
        axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Training')
        axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    if 'train_auc' in history and 'val_auc' in history:
        axes[1, 1].plot(epochs, history['train_auc'], 'b-', label='Training')
        axes[1, 1].plot(epochs, history['val_auc'], 'r-', label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Training and Validation AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def get_class_activation_map(model, video, target_layer, device):
    """
    Generate Class Activation Map (CAM) for video
    Helps visualize which spatial regions are important
    """
    model.eval()
    
    # Forward pass
    video = video.unsqueeze(0).to(device)
    
    # Get features from target layer
    features = None
    
    def hook_fn(module, input, output):
        nonlocal features
        features = output
    
    # Register hook
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        output = model(video)
    
    handle.remove()
    
    # Get the predicted class
    pred_class = output.argmax(dim=1).item()
    
    # Get weights from the final FC layer
    weights = model.classifier[-1].weight[pred_class]
    
    # Generate CAM
    cam = torch.zeros(features.shape[2:], dtype=torch.float32)
    
    for i, w in enumerate(weights):
        cam += w * features[0, i]
    
    # Normalize
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    
    return cam.cpu().numpy(), pred_class


def compare_model_predictions(models, video, device, class_names=['Intact', 'Detached']):
    """
    Compare predictions from multiple models on the same video
    """
    results = []
    
    video_input = video.unsqueeze(0).to(device)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(video_input)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            
            results.append({
                'Model': name,
                'Prediction': class_names[pred_class],
                'Confidence': f'{confidence:.2%}',
                'Prob_Intact': f'{probs[0, 0].item():.2%}',
                'Prob_Detached': f'{probs[0, 1].item():.2%}'
            })
    
    import pandas as pd
    df = pd.DataFrame(results)
    print("\nModel Predictions Comparison:")
    print(df.to_string(index=False))
    
    return df


def calculate_per_class_metrics(y_true, y_pred, class_names=['Intact', 'Detached']):
    """
    Calculate detailed metrics for each class
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True)
    
    import pandas as pd
    df = pd.DataFrame(report).transpose()
    
    print("\nPer-Class Metrics:")
    print(df.to_string())
    
    return df


if __name__ == '__main__':
    print("Analysis utilities loaded successfully!")
    print("\nAvailable functions:")
    print("- visualize_attention_maps()")
    print("- plot_confusion_matrix()")
    print("- plot_roc_curve()")
    print("- analyze_misclassifications()")
    print("- plot_training_history()")
    print("- get_class_activation_map()")
    print("- compare_model_predictions()")
    print("- calculate_per_class_metrics()")
