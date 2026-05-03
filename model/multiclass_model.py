"""
Multi-class ExplainableOpticalFlowResNet3D for hierarchical diagnosis
Outputs: diagnostic_class, subtype, anatomical_subclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class TemporalAttention(nn.Module):
    """Temporal attention module from ImprovedResNet3D"""
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
    """Spatial attention module from ImprovedResNet3D"""
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
    """Convolutional Block Attention Module for 3D from ImprovedResNet3D"""
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


class FrameImportanceModule(nn.Module):
    """Learn which frames are most important for classification"""
    def __init__(self, feature_dim=512):
        super(FrameImportanceModule, self).__init__()
        
        self.attention_conv = nn.Conv3d(feature_dim, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        importance = self.attention_conv(x)  # (B, 1, T, H, W)
        importance = torch.mean(importance, dim=[3, 4], keepdim=True)  # (B, 1, T, 1, 1)
        importance_scores = self.softmax(importance)  # (B, 1, T, 1, 1)
        weighted_features = x * importance_scores
        
        # Squeeze only the singleton dimensions (1, 1) but keep batch and temporal dims
        return weighted_features, importance_scores.squeeze(-1).squeeze(-1).squeeze(1)  # (B, T)


class SpatialExplainabilityModule(nn.Module):
    """Generate spatial attention maps showing important regions"""
    def __init__(self, in_channels=512):
        super(SpatialExplainabilityModule, self).__init__()
        
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


class OpticalFlowExtractor(nn.Module):
    """Extract optical flow features from consecutive frames"""
    def __init__(self, in_channels=3):
        super(OpticalFlowExtractor, self).__init__()
        
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


class MultiClassExplainableResNet3D(nn.Module):
    """
    Multi-class hierarchical diagnosis model
    
    Outputs two classification heads:
    - diagnostic_class: Primary diagnosis (non_rd, rd)
    - subtype: Subtype classification (normal, macula_intact, macula_detached, pvd)
    """
    def __init__(self, 
                 num_diagnostic_classes=2,   # non_rd, rd
                 num_subtype_classes=4,      # normal, macula_intact, macula_detached, pvd
                 pretrained=True, 
                 dropout=0.3,
                 use_attention=True):
        super(MultiClassExplainableResNet3D, self).__init__()
        
        self.num_diagnostic_classes = num_diagnostic_classes
        self.num_subtype_classes = num_subtype_classes
        self.use_attention = use_attention
        
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
        
        # Attention modules (from ImprovedResNet3D)
        if use_attention:
            self.temporal_attention = TemporalAttention(512)
            self.cbam = CBAM3D(512)
        
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
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2)
        )
        
        # Classification heads
        self.diagnostic_classifier = nn.Linear(512, num_diagnostic_classes)
        self.subtype_classifier = nn.Linear(512, num_subtype_classes)
        
    def forward(self, x, return_attention=False):
        """
        Forward pass with multi-class outputs
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            return_attention: If True, return attention maps
            
        Returns:
            outputs: Dict with 'diagnostic', 'subtype' logits
            attention_dict (optional): Dict with frame_importance and spatial_attention
        """
        # x: (B, C, T, H, W)
        
        # RGB stream with explainability
        rgb_features = self.rgb_backbone.stem(x)
        rgb_features = self.rgb_backbone.layer1(rgb_features)
        rgb_features = self.rgb_backbone.layer2(rgb_features)
        rgb_features = self.rgb_backbone.layer3(rgb_features)
        rgb_features = self.rgb_backbone.layer4(rgb_features)
        
        # Apply attention (from ImprovedResNet3D)
        if self.use_attention:
            rgb_features = self.temporal_attention(rgb_features)
            rgb_features = self.cbam(rgb_features)
        
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
        
        # Multi-class outputs
        diagnostic_output = self.diagnostic_classifier(fused)
        subtype_output = self.subtype_classifier(fused)
        
        outputs = {
            'diagnostic': diagnostic_output,
            'subtype': subtype_output
        }
        
        if return_attention:
            return outputs, {
                'frame_importance': frame_importance,
                'spatial_attention': spatial_attention
            }
        
        return outputs
    
    def freeze_backbone(self):
        """Freeze RGB backbone parameters for phase 1 training"""
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze RGB backbone parameters for phase 2 fine-tuning"""
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization"""
        with torch.no_grad():
            _, attention_dict = self.forward(x, return_attention=True)
        return attention_dict
    
    def extract_important_frames(self, x, top_k=5):
        """
        Extract top-k most important frames based on frame importance scores
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            top_k: Number of top frames to extract
            
        Returns:
            important_frames: Tensor of shape (B, top_k, C, H, W)
            frame_indices: Indices of important frames (B, top_k)
            importance_scores: Importance scores (B, top_k)
            spatial_attention: Spatial attention maps for important frames (B, top_k, 1, H, W)
        """
        with torch.no_grad():
            # Get attention maps
            _, attention_dict = self.forward(x, return_attention=True)
            frame_importance = attention_dict['frame_importance']  # (B, T)
            spatial_attention = attention_dict['spatial_attention']  # (B, 1, T, H, W)
            
            # Get top-k frame indices
            top_k = min(top_k, frame_importance.shape[1])
            importance_scores, frame_indices = torch.topk(frame_importance, top_k, dim=1)
            
            # Extract important frames
            B, C, T, H, W = x.shape
            important_frames = []
            important_attention = []
            
            for b in range(B):
                batch_frames = []
                batch_attention = []
                for k in range(top_k):
                    frame_idx = frame_indices[b, k]
                    batch_frames.append(x[b, :, frame_idx, :, :])
                    batch_attention.append(spatial_attention[b, :, frame_idx, :, :])
                
                important_frames.append(torch.stack(batch_frames))
                important_attention.append(torch.stack(batch_attention))
            
            important_frames = torch.stack(important_frames)  # (B, top_k, C, H, W)
            important_attention = torch.stack(important_attention)  # (B, top_k, 1, H, W)
            
        return important_frames, frame_indices, importance_scores, important_attention


def create_multiclass_model(num_diagnostic_classes=2,   # non_rd, rd
                            num_subtype_classes=4,      # normal, macula_intact, macula_detached, pvd
                            pretrained=True,
                            dropout=0.3,
                            use_attention=True):
    """
    Factory function to create multi-class model
    
    Args:
        num_diagnostic_classes: Number of primary diagnostic classes
        num_subtype_classes: Number of subtype classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        use_attention: Whether to use attention modules (TemporalAttention + CBAM3D)
        
    Returns:
        MultiClassExplainableResNet3D model
    """
    return MultiClassExplainableResNet3D(
        num_diagnostic_classes=num_diagnostic_classes,
        num_subtype_classes=num_subtype_classes,
        pretrained=pretrained,
        dropout=dropout,
        use_attention=use_attention
    )
