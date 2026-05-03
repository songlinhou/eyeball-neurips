"""
Multi-class video models adapted from binary classification models.

Each model outputs two classification heads:
- diagnostic_class: Primary diagnosis (non_rd, rd)
- subtype: Subtype classification (macula_detached, macula_intact, normal, pvd)

Supported models:
1. ResNet3D - Baseline 3D ResNet
2. I3D - Inflated 3D ConvNet
3. SlowFast - Dual-pathway architecture
4. X3D - Efficient video network
5. MViT - Multiscale Vision Transformer
6. VideoMAE - Masked autoencoder
7. TimeSformer - Space-time attention
8. C3D - Classic 3D CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import sys
import os

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../model'))


class MultiClassI3D(nn.Module):
    """I3D with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4, 
                 pretrained=True, dropout=0.5):
        super(MultiClassI3D, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2)
        )
        
        # Classification heads
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        return {
            'diagnostic': self.diagnostic_classifier(shared),
            'subtype': self.subtype_classifier(shared)
        }
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiClassResNet3D(nn.Module):
    """ResNet3D with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=True, dropout=0.5):
        super(MultiClassResNet3D, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.shared_features = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2)
        )
        
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        return {
            'diagnostic': self.diagnostic_classifier(shared),
            'subtype': self.subtype_classifier(shared)
        }
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiClassMViT(nn.Module):
    """MViT with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=True, dropout=0.5):
        super(MultiClassMViT, self).__init__()
        
        if pretrained:
            self.backbone = models.video.mvit_v2_s(weights=models.video.MViT_V2_S_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.mvit_v2_s(weights=None)
        
        self.feature_dim = 768
        self.backbone.head = nn.Identity()
        self.expected_frames = 16
        
        self.shared_features = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )
        
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # Temporal sampling for MViT
        if T != self.expected_frames:
            indices = torch.linspace(0, T - 1, self.expected_frames).long()
            x = x[:, :, indices, :, :]
        
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        return {
            'diagnostic': self.diagnostic_classifier(shared),
            'subtype': self.subtype_classifier(shared)
        }
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiClassSlowFast(nn.Module):
    """SlowFast with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=True, dropout=0.5):
        super(MultiClassSlowFast, self).__init__()
        
        if pretrained:
            self.slow_pathway = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
            self.fast_pathway = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.slow_pathway = models.video.r3d_18(weights=None)
            self.fast_pathway = models.video.r3d_18(weights=None)
        
        self.slow_pathway.fc = nn.Identity()
        self.fast_pathway.fc = nn.Identity()
        
        self.feature_dim = 512
        
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout)
        )
        
        self.shared_features = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2)
        )
        
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        slow_indices = torch.linspace(0, T - 1, T // 4).long()
        fast_indices = torch.arange(0, T, 2)
        
        slow_input = x[:, :, slow_indices, :, :]
        fast_input = x[:, :, fast_indices, :, :]
        
        slow_features = self.slow_pathway(slow_input)
        fast_features = self.fast_pathway(fast_input)
        
        fused = torch.cat([slow_features, fast_features], dim=1)
        fused = self.fusion(fused)
        shared = self.shared_features(fused)
        
        return {
            'diagnostic': self.diagnostic_classifier(shared),
            'subtype': self.subtype_classifier(shared)
        }
    
    def freeze_backbone(self):
        for param in self.slow_pathway.parameters():
            param.requires_grad = False
        for param in self.fast_pathway.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.slow_pathway.parameters():
            param.requires_grad = True
        for param in self.fast_pathway.parameters():
            param.requires_grad = True


class MultiClassX3D(nn.Module):
    """X3D with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=True, dropout=0.5):
        super(MultiClassX3D, self).__init__()
        
        if pretrained:
            self.backbone = models.video.s3d(weights=models.video.S3D_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.s3d(weights=None)
        
        self.feature_dim = 1024
        self.backbone.classifier = nn.Identity()
        
        self.shared_features = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2)
        )
        
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        shared = self.shared_features(features)
        
        return {
            'diagnostic': self.diagnostic_classifier(shared),
            'subtype': self.subtype_classifier(shared)
        }
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiClassVideoMAE(nn.Module):
    """VideoMAE with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=True, dropout=0.5):
        super(MultiClassVideoMAE, self).__init__()
        
        if pretrained:
            self.backbone = models.video.mvit_v2_s(weights=models.video.MViT_V2_S_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.mvit_v2_s(weights=None)
        
        self.feature_dim = 768
        self.backbone.head = nn.Identity()
        self.expected_frames = 16
        
        self.shared_features = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )
        
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        if T != self.expected_frames:
            indices = torch.linspace(0, T - 1, self.expected_frames).long()
            x = x[:, :, indices, :, :]
        
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        return {
            'diagnostic': self.diagnostic_classifier(shared),
            'subtype': self.subtype_classifier(shared)
        }
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiClassTimeSformer(nn.Module):
    """TimeSformer with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=True, dropout=0.5, num_frames=32):
        super(MultiClassTimeSformer, self).__init__()
        
        try:
            if pretrained:
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            else:
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
            self.feature_dim = self.backbone.num_features
        except:
            self.backbone = models.video.mvit_v1_b(weights=models.video.MViT_V1_B_Weights.KINETICS400_V1 if pretrained else None)
            self.feature_dim = 768
            self.backbone.head = nn.Identity()
        
        self.num_frames = num_frames
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, self.feature_dim))
        
        self.shared_features = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )
        
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        frame_features = []
        for t in range(T):
            frame = x[:, :, t, :, :]
            if hasattr(self.backbone, 'forward_features'):
                feat = self.backbone.forward_features(frame)
                if len(feat.shape) == 3:
                    feat = feat[:, 0]
            else:
                feat = self.backbone(frame)
            frame_features.append(feat)
        
        features = torch.stack(frame_features, dim=1)
        
        if features.shape[1] == self.num_frames:
            features = features + self.temporal_embed
        
        pooled = features.mean(dim=1)
        shared = self.shared_features(pooled)
        
        return {
            'diagnostic': self.diagnostic_classifier(shared),
            'subtype': self.subtype_classifier(shared)
        }
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiClassC3D(nn.Module):
    """C3D with multi-class outputs"""
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=False, dropout=0.5):
        super(MultiClassC3D, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 2048)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
        self.diagnostic_classifier = nn.Linear(2048, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(2048, num_subtype_classes)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        
        return {
            'diagnostic': self.diagnostic_classifier(x),
            'subtype': self.subtype_classifier(x)
        }
    
    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not name.startswith('diagnostic') and not name.startswith('subtype'):
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True


class MultiClassExplainableModel(nn.Module):
    """
    Multi-class version of ExplainableResNet3D with attention.
    Imports from the model directory.
    """
    def __init__(self, num_diagnostic_classes=2, num_subtype_classes=4,
                 pretrained=True, dropout=0.5):
        super(MultiClassExplainableModel, self).__init__()
        
        try:
            from multiclass_model import MultiClassExplainableResNet3D
            self.model = MultiClassExplainableResNet3D(
                num_diagnostic_classes=num_diagnostic_classes,
                num_subtype_classes=num_subtype_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_attention=True
            )
        except ImportError:
            raise ImportError(
                "Could not import MultiClassExplainableResNet3D. "
                "Make sure multiclass_model.py is in the model directory."
            )
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        self.model.freeze_backbone()
    
    def unfreeze_backbone(self):
        self.model.unfreeze_backbone()


def get_multiclass_model(model_name, num_diagnostic_classes=2, num_subtype_classes=4,
                         pretrained=True, dropout=0.5, **kwargs):
    """
    Factory function to get multi-class video classification models.
    
    Args:
        model_name: Name of the model
        num_diagnostic_classes: Number of diagnostic classes (default: 2)
        num_subtype_classes: Number of subtype classes (default: 4)
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        **kwargs: Additional model-specific arguments (img_size, num_frames, etc.)
    
    Returns:
        PyTorch model with multi-class outputs
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet3d':
        return MultiClassResNet3D(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    elif model_name == 'i3d':
        return MultiClassI3D(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    elif model_name == 'slowfast':
        return MultiClassSlowFast(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    elif model_name == 'x3d':
        return MultiClassX3D(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    elif model_name == 'mvit':
        return MultiClassMViT(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    elif model_name == 'videomae':
        return MultiClassVideoMAE(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    elif model_name == 'timesformer':
        return MultiClassTimeSformer(num_diagnostic_classes, num_subtype_classes, pretrained, dropout,
                                     num_frames=kwargs.get('num_frames', 32))
    elif model_name == 'c3d':
        return MultiClassC3D(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    elif model_name == 'explainable':
        return MultiClassExplainableModel(num_diagnostic_classes, num_subtype_classes, pretrained, dropout)
    else:
        raise ValueError(f"Unknown multi-class model: {model_name}. "
                        f"Supported: resnet3d, i3d, slowfast, x3d, mvit, videomae, timesformer, c3d, explainable")
