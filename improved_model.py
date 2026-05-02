import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), 
                             padding=(0, kernel_size//2, kernel_size//2))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM3D, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ImprovedResNet3D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5, use_attention=True):
        super(ImprovedResNet3D, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.use_attention = use_attention
        
        # Remove the final FC layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add attention modules
        if use_attention:
            self.temporal_attention = TemporalAttention(512)
            self.cbam = CBAM3D(512)
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        
        # Extract features through backbone layers
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply attention
        if self.use_attention:
            x = self.temporal_attention(x)
            x = self.cbam(x)
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        x = x.flatten(1)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def unfreeze_layer(self, layer_name):
        layer = getattr(self.backbone, layer_name)
        for param in layer.parameters():
            param.requires_grad = True


class MultiScaleResNet3D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(MultiScaleResNet3D, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.backbone.fc = nn.Identity()
        
        # Multi-scale feature extraction
        self.layer2_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.layer3_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.layer4_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Feature dimensions from different layers
        self.layer2_dim = 128
        self.layer3_dim = 256
        self.layer4_dim = 512
        
        total_features = self.layer2_dim + self.layer3_dim + self.layer4_dim
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        
        x2 = self.backbone.layer2(x)
        feat2 = self.layer2_pool(x2).flatten(1)
        
        x3 = self.backbone.layer3(x2)
        feat3 = self.layer3_pool(x3).flatten(1)
        
        x4 = self.backbone.layer4(x3)
        feat4 = self.layer4_pool(x4).flatten(1)
        
        # Concatenate multi-scale features
        features = torch.cat([feat2, feat3, feat4], dim=1)
        
        # Classification
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNet3DWithAuxiliary(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(ResNet3DWithAuxiliary, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Main classifier
        self.main_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary classifier for intermediate features
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, return_aux=False):
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        
        # Auxiliary output from layer3
        aux_out = None
        if return_aux:
            aux_out = self.aux_classifier(x)
        
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.flatten(1)
        
        # Main output
        main_out = self.main_classifier(x)
        
        if return_aux:
            return main_out, aux_out
        return main_out
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)
        
        loss = -log_preds.sum(dim=1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        
        return (1 - self.smoothing) * nll + self.smoothing * loss / n_classes
