"""
Model Architectures for ASIC-CNN Hybrid Benchmark

This module contains:
1. StandardCNN - Baseline ResNet-18 for chest X-ray classification
2. HybridSingleScale - CNN with single-scale ASIC attention
3. HybridMultiScale - CNN with multi-scale ASIC attention pyramid

Author: Francisco Angulo de Lafuente
GitHub: https://github.com/Agnuxo1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# STANDARD CNN (BASELINE)
# =============================================================================

class StandardCNN(nn.Module):
    """
    Standard ResNet-18 based classifier for chest X-ray images.
    
    This serves as the baseline for comparison with hybrid models.
    
    Architecture:
        Input (224x224x1) → ResNet-18 → Global Avg Pool → FC → Softmax
    """
    
    def __init__(self, backbone_name: str = 'resnet50', num_classes: int = 2, 
                 pretrained: bool = True, dropout: float = 0.5):
        """
        Initialize standard CNN.
        
        Args:
            backbone_name: Backbone architecture (resnet18, resnet50)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability for classifier
        """
        super(StandardCNN, self).__init__()
        
        # Load backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Modify first conv layer for grayscale input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize with mean of RGB channels
        if pretrained:
            self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Identity()
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, H, W)
            
        Returns:
            Dictionary with 'logits' and 'features'
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'probabilities': F.softmax(logits, dim=1)
        }
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract intermediate feature maps.
        
        Useful for visualization and hybrid models.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps at different scales
        """
        feature_maps = []
        
        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # ResNet blocks
        x = self.backbone.layer1(x)
        feature_maps.append(x)  # 56x56
        
        x = self.backbone.layer2(x)
        feature_maps.append(x)  # 28x28
        
        x = self.backbone.layer3(x)
        feature_maps.append(x)  # 14x14
        
        x = self.backbone.layer4(x)
        feature_maps.append(x)  # 7x7
        
        return feature_maps


# =============================================================================
# ATTENTION MODULE
# =============================================================================

class ASICAttentionModule(nn.Module):
    """
    Module that applies ASIC-generated attention to feature maps.
    
    The attention map is generated externally by the ASIC and passed
    during forward. This module learns how to best integrate the
    attention with CNN features.
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        """
        Initialize attention module.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for attention processing
        """
        super(ASICAttentionModule, self).__init__()
        
        # Learn to refine ASIC attention
        self.attention_processor = nn.Sequential(
            nn.Conv2d(1, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Learnable blend weight
        self.blend_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, features: torch.Tensor, 
                asic_attention: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to features.
        
        Args:
            features: CNN feature map (B, C, H, W)
            asic_attention: ASIC attention map (B, 1, H, W)
            
        Returns:
            Attention-weighted features (B, C, H, W)
        """
        # Resize attention if needed
        if asic_attention.shape[-2:] != features.shape[-2:]:
            asic_attention = F.interpolate(
                asic_attention,
                size=features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Process attention
        processed_attention = self.attention_processor(asic_attention)
        
        # Apply attention with learnable blend
        attended = features * (1 + self.blend_weight * processed_attention)
        
        return attended


# =============================================================================
# HYBRID SINGLE-SCALE MODEL
# =============================================================================

class HybridSingleScale(nn.Module):
    """
    Hybrid CNN with single-scale ASIC attention.
    
    The ASIC attention is injected at the final feature map,
    guiding the classifier to focus on relevant regions.
    
    Architecture:
        Input (224x224x1) → ResNet-18 → ASIC Attention → Classifier
    """
    
    def __init__(self, backbone_name: str = 'resnet50', num_classes: int = 2, 
                 pretrained: bool = True, dropout: float = 0.5):
        """
        Initialize hybrid model.
        
        Args:
            backbone_name: Backbone architecture
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability
        """
        super(HybridSingleScale, self).__init__()
        
        # Load backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Modify for grayscale
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Attention module (applied to final feature map)
        self.attention_module = ASICAttentionModule(self.feature_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor,
                asic_attention: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, H, W)
            asic_attention: ASIC attention map (B, 1, H, W) or None
            
        Returns:
            Dictionary with outputs
        """
        # Extract features through backbone (up to layer4)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply ASIC attention if provided
        attention_map = None
        if asic_attention is not None:
            attention_map = asic_attention
            x = self.attention_module(x, asic_attention)
        
        # Global average pooling
        features = self.backbone.avgpool(x)
        features = torch.flatten(features, 1)
        
        # Classify
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'probabilities': F.softmax(logits, dim=1),
            'attention_map': attention_map
        }


# =============================================================================
# HYBRID MULTI-SCALE MODEL
# =============================================================================

class HybridMultiScale(nn.Module):
    """
    Hybrid CNN with multi-scale ASIC attention pyramid.
    
    The ASIC attention is injected at multiple scales,
    providing hierarchical guidance throughout the network.
    
    Architecture:
        Input → ResNet Layer 1 + Attention → Layer 2 + Attention → ... → Classifier
    """
    
    def __init__(self, backbone_name: str = 'resnet50', num_classes: int = 2, 
                 pretrained: bool = True, dropout: float = 0.5):
        """
        Initialize multi-scale hybrid model.
        
        Args:
            backbone_name: Backbone architecture
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability
        """
        super(HybridMultiScale, self).__init__()
        
        # Load backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.channels = [64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.channels = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Modify for grayscale
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Remove original fc
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Attention modules at different scales
        self.attention_modules = nn.ModuleDict({
            'layer1': ASICAttentionModule(self.channels[0]),
            'layer2': ASICAttentionModule(self.channels[1]),
            'layer3': ASICAttentionModule(self.channels[2]),
            'layer4': ASICAttentionModule(self.channels[3]),
        })
        
        # Feature fusion
        total_channels = sum(self.channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor,
                attention_pyramid: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, H, W)
            attention_pyramid: List of attention maps at different scales
                              [56x56, 28x28, 14x14, 7x7] or None
            
        Returns:
            Dictionary with outputs
        """
        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # ResNet blocks with attention
        feature_maps = []
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        for i, layer_name in enumerate(layer_names):
            layer = getattr(self.backbone, layer_name)
            x = layer(x)
            
            # Apply attention if provided
            if attention_pyramid is not None and i < len(attention_pyramid):
                attn = attention_pyramid[i]
                if attn.dim() == 3:
                    attn = attn.unsqueeze(1)  # Add channel dim
                x = self.attention_modules[layer_name](x, attn)
            
            feature_maps.append(x)
        
        # Multi-scale feature fusion
        # Resize all to smallest (7x7) and concatenate
        target_size = feature_maps[-1].shape[-2:]
        
        resized_features = []
        for fm in feature_maps:
            if fm.shape[-2:] != target_size:
                fm = F.adaptive_avg_pool2d(fm, target_size)
            resized_features.append(fm)
        
        fused = torch.cat(resized_features, dim=1)
        fused = self.fusion(fused)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(fused, 1)
        features = torch.flatten(features, 1)
        
        # Classify
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'probabilities': F.softmax(logits, dim=1),
            'attention_pyramid': attention_pyramid,
            'feature_maps': feature_maps
        }


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_name: str, backbone_name: str = 'resnet50', 
                 num_classes: int = 2, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Create model by name.
    
    Args:
        model_name: One of 'standard_cnn', 'hybrid_single', 'hybrid_multi'
        backbone_name: Backbone architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for model
        
    Returns:
        Model instance
    """
    models_dict = {
        'standard_cnn': StandardCNN,
        'hybrid_single': HybridSingleScale,
        'hybrid_multi': HybridMultiScale
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")
    
    return models_dict[model_name](
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


# =============================================================================
# MODEL INFO
# =============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def print_model_summary(model: nn.Module, model_name: str):
    """Print model summary."""
    params = count_parameters(model)
    
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Total parameters:     {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Non-trainable:        {params['non_trainable']:,}")
    print(f"{'=' * 60}\n")


# =============================================================================
# TEST
# =============================================================================

def test_models():
    """Test all model architectures."""
    print("\n" + "=" * 60)
    print("Testing Model Architectures")
    print("=" * 60)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 1, 224, 224)
    
    # Test standard CNN
    print("\n1. Standard CNN")
    model1 = StandardCNN(num_classes=2)
    out1 = model1(x)
    print(f"   Input:  {x.shape}")
    print(f"   Logits: {out1['logits'].shape}")
    print(f"   Features: {out1['features'].shape}")
    print_model_summary(model1, "StandardCNN")
    
    # Test hybrid single-scale
    print("\n2. Hybrid Single-Scale")
    model2 = HybridSingleScale(num_classes=2)
    attention = torch.rand(batch_size, 1, 7, 7)  # Final feature map size
    out2 = model2(x, attention)
    print(f"   Input:     {x.shape}")
    print(f"   Attention: {attention.shape}")
    print(f"   Logits:    {out2['logits'].shape}")
    print_model_summary(model2, "HybridSingleScale")
    
    # Test hybrid multi-scale
    print("\n3. Hybrid Multi-Scale")
    model3 = HybridMultiScale(num_classes=2)
    attention_pyramid = [
        torch.rand(batch_size, 1, 56, 56),  # layer1
        torch.rand(batch_size, 1, 28, 28),  # layer2
        torch.rand(batch_size, 1, 14, 14),  # layer3
        torch.rand(batch_size, 1, 7, 7),    # layer4
    ]
    out3 = model3(x, attention_pyramid)
    print(f"   Input:  {x.shape}")
    print(f"   Pyramid: {[a.shape for a in attention_pyramid]}")
    print(f"   Logits: {out3['logits'].shape}")
    print_model_summary(model3, "HybridMultiScale")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_models()
