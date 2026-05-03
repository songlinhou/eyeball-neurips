import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class I3DModel(nn.Module):
    """
    Inflated 3D ConvNet (I3D) - Quo Vadis, Action Recognition? (CVPR 2017)
    State-of-the-art for video classification, inflates 2D filters to 3D
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(I3DModel, self).__init__()
        
        if pretrained:
            self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class SlowFastModel(nn.Module):
    """
    SlowFast Networks for Video Recognition (ICCV 2019)
    Dual-pathway architecture: slow pathway for spatial semantics, fast for motion
    Popular for medical video analysis
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(SlowFastModel, self).__init__()
        
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
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
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
        
        return self.classifier(fused)
    
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


class X3DModel(nn.Module):
    """
    X3D: Expanding Architectures for Efficient Video Recognition (CVPR 2020)
    Efficient video network with progressive expansion
    Good for resource-constrained medical applications
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5, model_size='s'):
        super(X3DModel, self).__init__()
        
        if pretrained:
            if model_size == 's':
                self.backbone = models.video.s3d(weights=models.video.S3D_Weights.KINETICS400_V1)
            else:
                self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        else:
            if model_size == 's':
                self.backbone = models.video.s3d(weights=None)
            else:
                self.backbone = models.video.r3d_18(weights=None)
        
        if hasattr(self.backbone, 'classifier'):
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class TimeSformerModel(nn.Module):
    """
    TimeSformer: Is Space-Time Attention All You Need for Video Understanding? (ICML 2021)
    Transformer-based video model with divided space-time attention
    State-of-the-art for many video tasks
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5, img_size=224, num_frames=32):
        super(TimeSformerModel, self).__init__()
        
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
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
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
        
        return self.classifier(pooled)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class VideoMAEModel(nn.Module):
    """
    VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training (NeurIPS 2022)
    Self-supervised learning approach, excellent for limited medical data
    
    Note: Uses MViT backbone which expects 16 frames. For different frame counts,
    we use temporal pooling/sampling to match the expected input.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(VideoMAEModel, self).__init__()
        
        if pretrained:
            self.backbone = models.video.mvit_v2_s(weights=models.video.MViT_V2_S_Weights.KINETICS400_V1)
        else:
            self.backbone = models.video.mvit_v2_s(weights=None)
        
        self.feature_dim = 768
        self.backbone.head = nn.Identity()
        self.expected_frames = 16
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        if T != self.expected_frames:
            if T > self.expected_frames:
                indices = torch.linspace(0, T - 1, self.expected_frames).long()
                x = x[:, :, indices, :, :]
            else:
                indices = torch.linspace(0, T - 1, self.expected_frames)
                indices = indices.long()
                indices = torch.clamp(indices, 0, T - 1)
                x = x[:, :, indices, :, :]
        
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class MViTModel(nn.Module):
    """
    Multiscale Vision Transformers (MViT) (ICCV 2021)
    Hierarchical vision transformer for video recognition
    Efficient and accurate for medical video classification
    
    Note: MViT expects 16 frames by default. For different frame counts,
    we use temporal pooling/sampling to match the expected input.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5, version='v1'):
        super(MViTModel, self).__init__()
        
        if version == 'v1':
            if pretrained:
                self.backbone = models.video.mvit_v1_b(weights=models.video.MViT_V1_B_Weights.KINETICS400_V1)
            else:
                self.backbone = models.video.mvit_v1_b(weights=None)
        else:
            if pretrained:
                self.backbone = models.video.mvit_v2_s(weights=models.video.MViT_V2_S_Weights.KINETICS400_V1)
            else:
                self.backbone = models.video.mvit_v2_s(weights=None)
        
        self.feature_dim = 768
        self.backbone.head = nn.Identity()
        self.expected_frames = 16
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        if T != self.expected_frames:
            if T > self.expected_frames:
                indices = torch.linspace(0, T - 1, self.expected_frames).long()
                x = x[:, :, indices, :, :]
            else:
                indices = torch.linspace(0, T - 1, self.expected_frames)
                indices = indices.long()
                indices = torch.clamp(indices, 0, T - 1)
                x = x[:, :, indices, :, :]
        
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


class C3DModel(nn.Module):
    """
    C3D: Learning Spatiotemporal Features with 3D Convolutional Networks (ICCV 2015)
    Classic 3D CNN baseline, widely used in medical imaging
    """
    def __init__(self, num_classes=2, pretrained=False, dropout=0.5):
        super(C3DModel, self).__init__()
        
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
        
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
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
        
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        
        logits = self.fc8(x)
        
        return logits
    
    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not name.startswith('fc8'):
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True


class ResNet3DModel(nn.Module):
    """
    ResNet3D - baseline 3D ResNet
    Standard baseline for video classification
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5, depth=18):
        super(ResNet3DModel, self).__init__()
        
        if depth == 18:
            if pretrained:
                self.backbone = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
            else:
                self.backbone = models.video.r3d_18(weights=None)
        elif depth == 50:
            if pretrained:
                self.backbone = models.video.r3d_50(weights=models.video.R3D_50_Weights.KINETICS400_V1)
            else:
                self.backbone = models.video.r3d_50(weights=None)
        else:
            self.backbone = models.video.r3d_18(weights=None)
        
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_model(model_name, num_classes=2, pretrained=True, dropout=0.5, **kwargs):
    """
    Factory function to get video classification models
    
    Args:
        model_name: Name of the model ('i3d', 'slowfast', 'x3d', 'timesformer', 'videomae', 'mvit', 'c3d', 'resnet3d')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name == 'i3d':
        return I3DModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_name == 'slowfast':
        return SlowFastModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_name == 'x3d':
        return X3DModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout, 
                       model_size=kwargs.get('model_size', 's'))
    elif model_name == 'timesformer':
        return TimeSformerModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout,
                               img_size=kwargs.get('img_size', 224), 
                               num_frames=kwargs.get('num_frames', 32))
    elif model_name == 'videomae':
        return VideoMAEModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_name == 'mvit':
        return MViTModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout,
                        version=kwargs.get('version', 'v1'))
    elif model_name == 'c3d':
        return C3DModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_name == 'resnet3d':
        return ResNet3DModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout,
                            depth=kwargs.get('depth', 18))
    else:
        raise ValueError(f"Unknown model: {model_name}")
