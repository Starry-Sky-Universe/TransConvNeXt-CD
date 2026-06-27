"""
TransConvNeXt-CD: Transformer-CNN Hybrid Network for Change Detection

Core model architecture:
- Encoder: ConvNeXt-Tiny (timm backbone)
- Interaction: SpatialDifferenceModule + CrossAttentionBlock
- Decoder: Multi-level with SE attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CrossAttentionBlock(nn.Module):
    """Cross-attention between two feature streams (T1 and T2)."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)

        q1 = self.norm_q(x1_flat)
        k2 = self.norm_k(x2_flat)
        attn_out1, _ = self.attn(q1, k2, k2)
        x1_out = x1_flat + attn_out1
        x1_out = x1_out + self.ffn(x1_out)

        q2 = self.norm_q(x2_flat)
        k1 = self.norm_k(x1_flat)
        attn_out2, _ = self.attn(q2, k1, k1)
        x2_out = x2_flat + attn_out2
        x2_out = x2_out + self.ffn(x2_out)

        return (
            x1_out.transpose(1, 2).reshape(B, C, H, W),
            x2_out.transpose(1, 2).reshape(B, C, H, W),
        )


class SpatialDifferenceModule(nn.Module):
    """Spatial difference module for early-stage features."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
        )

    def forward(self, t1, t2):
        return self.conv(torch.cat([t1, t2], dim=1))


class DecoderBlock(nn.Module):
    """Decoder block with SE attention and skip connections."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 8, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x * self.se(x)


class TransConvNeXt_CD(nn.Module):
    """
    Transformer-CNN Hybrid Network for Change Detection.

    Uses ConvNeXt-Tiny as encoder, Cross-Attention for high-level
    feature interaction, and a multi-level decoder with deep supervision.
    """

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True
        )

        # Interaction modules
        self.inter_1 = SpatialDifferenceModule(96)
        self.inter_2 = SpatialDifferenceModule(192)
        self.inter_3 = SpatialDifferenceModule(384)
        self.transformer_inter = CrossAttentionBlock(dim=768, num_heads=12)
        self.inter_4_fuse = nn.Conv2d(768 * 2, 768, 1)

        # Decoder
        self.dec4 = DecoderBlock(768, 384, 384)
        self.dec3 = DecoderBlock(384, 192, 192)
        self.dec2 = DecoderBlock(192, 96, 96)
        self.dec1 = DecoderBlock(96, 0, 64)

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

        # Deep supervision heads
        self.head_d4 = nn.Conv2d(384, 1, 1)
        self.head_d3 = nn.Conv2d(192, 1, 1)

    def forward(self, t1, t2):
        f1 = self.encoder(t1)
        f2 = self.encoder(t2)

        diff_1 = self.inter_1(f1[0], f2[0])
        diff_2 = self.inter_2(f1[1], f2[1])
        diff_3 = self.inter_3(f1[2], f2[2])

        t1_out, t2_out = self.transformer_inter(f1[3], f2[3])
        diff_4 = self.inter_4_fuse(torch.cat([t1_out, t2_out], dim=1))

        d4 = self.dec4(diff_4, diff_3)
        d3 = self.dec3(d4, diff_2)
        d2 = self.dec2(d3, diff_1)
        d1 = self.dec1(d2)
        out = self.final_conv(d1)

        if self.training:
            return out, self.head_d4(d4), self.head_d3(d3)
        return out