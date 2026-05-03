import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class OpticalFlowExtractor(nn.Module):
    """Extract optical flow features from consecutive frames"""
    def __init__(self, in_channels=3):
        super(OpticalFlowExtractor, self).__init__()
        
        # Lightweight flow estimation network
        self.flow_conv1 = nn.Conv3d(in_channels, 32, kernel_size=(2, 3, 3), 
                                    stride=(1, 1, 1), padding=(0, 1, 1))
        self.flow_conv2 = nn.Conv3d(32, 64, kernel_size=(2, 3, 3), 
                                    stride=(1, 1, 1), padding=(0, 1, 1))
        self.flow_conv3 = nn.Conv3d(64, 32, kernel_size=(1, 3, 3), 
                                    stride=(1, 1, 1), padding=(0, 1, 1))
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        flow = F.relu(self.flow_conv1(x))
        flow = F.relu(self.flow_conv2(flow))
        flow = self.flow_conv3(flow)
        return flow


class FrameImportanceModule(nn.Module):
    """Learn which frames are most important for classification"""
    def __init__(self, feature_dim=512):
        super(FrameImportanceModule, self).__init__()
        
        self.attention_conv = nn.Conv3d(feature_dim, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)  # Softmax over temporal dimension
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        # Compute frame importance scores
        importance = self.attention_conv(x)  # (B, 1, T, H, W)
        
        # Average over spatial dimensions
        importance = torch.mean(importance, dim=[3, 4], keepdim=True)  # (B, 1, T, 1, 1)
        
        # Normalize across frames
        importance_scores = self.softmax(importance)  # (B, 1, T, 1, 1)
        
        # Apply attention
        weighted_features = x * importance_scores
        
        return weighted_features, importance_scores.squeeze()


class SpatialExplainabilityModule(nn.Module):
    """Generate spatial attention maps showing important regions"""
    def __init__(self, in_channels=512):
        super(SpatialExplainabilityModule, self).__init__()
        
        # Generate attention maps
        self.attention_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        attention_map = self.attention_conv(x)  # (B, 1, T, H, W)
        weighted_features = x * attention_map
        
        return weighted_features, attention_map


class ExplainableResNet3D(nn.Module):
    """
    ResNet3D with explainability features:
    - Frame importance scores
    - Spatial attention maps
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(ExplainableResNet3D, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Explainability modules
        self.frame_importance = FrameImportanceModule(512)
        self.spatial_explainability = SpatialExplainabilityModule(512)
        
        # Enhanced classifier
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
        
    def forward(self, x, return_attention=False):
        # x: (B, C, T, H, W)
        
        # Extract features
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply explainability modules
        x, frame_importance = self.frame_importance(x)
        x, spatial_attention = self.spatial_explainability(x)
        
        # Global pooling
        x = self.backbone.avgpool(x)
        x = x.flatten(1)
        
        # Classification
        output = self.classifier(x)
        
        if return_attention:
            return output, {
                'frame_importance': frame_importance,
                'spatial_attention': spatial_attention
            }
        
        return output
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class OpticalFlowResNet3D(nn.Module):
    """
    ResNet3D with optical flow features for motion analysis
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(OpticalFlowResNet3D, self).__init__()
        
        # RGB stream
        if pretrained:
            self.rgb_backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.rgb_backbone = models.video.r3d_18(weights=None)
        
        self.rgb_backbone.fc = nn.Identity()
        
        # Optical flow extraction
        self.flow_extractor = OpticalFlowExtractor(in_channels=3)
        
        # Flow stream (smaller network)
        self.flow_stream = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        
        # RGB stream
        rgb_features = self.rgb_backbone.stem(x)
        rgb_features = self.rgb_backbone.layer1(rgb_features)
        rgb_features = self.rgb_backbone.layer2(rgb_features)
        rgb_features = self.rgb_backbone.layer3(rgb_features)
        rgb_features = self.rgb_backbone.layer4(rgb_features)
        rgb_features = self.rgb_backbone.avgpool(rgb_features)
        rgb_features = rgb_features.flatten(1)
        
        # Flow stream
        flow_features = self.flow_extractor(x)
        flow_features = self.flow_stream(flow_features)
        flow_features = flow_features.flatten(1)
        
        # Fusion
        combined = torch.cat([rgb_features, flow_features], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.classifier(fused)
        
        return output
    
    def freeze_backbone(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True


class ExplainableOpticalFlowResNet3D(nn.Module):
    """
    Combined model with both optical flow and explainability features
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(ExplainableOpticalFlowResNet3D, self).__init__()
        
        # RGB stream
        if pretrained:
            self.rgb_backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.rgb_backbone = models.video.r3d_18(weights=None)
        
        self.rgb_backbone.fc = nn.Identity()
        
        # Optical flow extraction
        self.flow_extractor = OpticalFlowExtractor(in_channels=3)
        
        # Flow stream
        self.flow_stream = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Explainability modules
        self.frame_importance = FrameImportanceModule(512)
        self.spatial_explainability = SpatialExplainabilityModule(512)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        # x: (B, C, T, H, W)
        
        # RGB stream with explainability
        rgb_features = self.rgb_backbone.stem(x)
        rgb_features = self.rgb_backbone.layer1(rgb_features)
        rgb_features = self.rgb_backbone.layer2(rgb_features)
        rgb_features = self.rgb_backbone.layer3(rgb_features)
        rgb_features = self.rgb_backbone.layer4(rgb_features)
        
        # Apply explainability
        rgb_features, frame_importance = self.frame_importance(rgb_features)
        rgb_features, spatial_attention = self.spatial_explainability(rgb_features)
        
        rgb_features = self.rgb_backbone.avgpool(rgb_features)
        rgb_features = rgb_features.flatten(1)
        
        # Flow stream
        flow_features = self.flow_extractor(x)
        flow_features = self.flow_stream(flow_features)
        flow_features = flow_features.flatten(1)
        
        # Fusion
        combined = torch.cat([rgb_features, flow_features], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.classifier(fused)
        
        if return_attention:
            return output, {
                'frame_importance': frame_importance,
                'spatial_attention': spatial_attention
            }
        
        return output
    
    def freeze_backbone(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        # Keep flow extractor trainable
            
    def unfreeze_backbone(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True


class TemporalShiftModule(nn.Module):
    """Temporal Shift Module for efficient temporal modeling"""
    def __init__(self, n_div=8):
        super(TemporalShiftModule, self).__init__()
        self.n_div = n_div
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        fold = C // self.n_div
        
        out = torch.zeros_like(x)
        out[:, :fold, 1:, :, :] = x[:, :fold, :-1, :, :]  # Shift left
        out[:, fold:2*fold, :-1, :, :] = x[:, fold:2*fold, 1:, :, :]  # Shift right
        out[:, 2*fold:, :, :, :] = x[:, 2*fold:, :, :, :]  # No shift
        
        return out


class TSMResNet3D(nn.Module):
    """
    ResNet3D with Temporal Shift Module and explainability
    More memory efficient than full 3D convolutions
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(TSMResNet3D, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add TSM modules
        self.tsm1 = TemporalShiftModule(n_div=8)
        self.tsm2 = TemporalShiftModule(n_div=8)
        
        # Explainability
        self.frame_importance = FrameImportanceModule(512)
        self.spatial_explainability = SpatialExplainabilityModule(512)
        
        # Classifier
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
        
    def forward(self, x, return_attention=False):
        # x: (B, C, T, H, W)
        
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        
        x = self.tsm1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        
        x = self.tsm2(x)
        x = self.backbone.layer4(x)
        
        # Explainability
        x, frame_importance = self.frame_importance(x)
        x, spatial_attention = self.spatial_explainability(x)
        
        x = self.backbone.avgpool(x)
        x = x.flatten(1)
        
        output = self.classifier(x)
        
        if return_attention:
            return output, {
                'frame_importance': frame_importance,
                'spatial_attention': spatial_attention
            }
        
        return output
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
