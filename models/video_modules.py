import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.monitor import DataFlowMonitor

import torchvision.models as models


class LandmarkEncoder(nn.Module):
    def __init__(self, config, monitor=None):
        super().__init__()
        self.monitor = monitor
        self.point_encoder = nn.Sequential(
            nn.Linear(2, config.landmark_point_hidden),  # 直接映射到64维
            nn.ReLU(),
            nn.LayerNorm(config.landmark_point_hidden)
        )
        self.spatial_attn = nn.MultiheadAttention(
            config.landmark_point_hidden, config.landmark_num_heads, dropout=config.landmark_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.landmark_point_hidden)
        self.output_proj = nn.Linear(config.landmark_point_hidden * 68, config.landmark_hidden_dim)

    def forward(self, x):
        B, T, N, _ = x.shape
        x_reshaped = x.view(B * T, N, 2)
        point_feats = self.point_encoder(x_reshaped)  # [B*T, 68, 64]

        # 空间注意力
        attn_out, _ = self.spatial_attn(point_feats, point_feats, point_feats)
        point_feats = self.norm(point_feats + attn_out)  # 残差连接+归一化

        # 输出投影
        features = self.output_proj(point_feats.reshape(B * T, -1))  # [B*T, 256]
        features = features.view(B, T, -1)  # [B, T, 256]

        # 监控
        if self.monitor:
            self.monitor.log_data(data=features, location="LandmarkEncoder", data_type="output",
                                  processing_step="output")
        return features


class LipEncoder(nn.Module):
    def __init__(self, config, monitor=None):
        super().__init__()
        self.monitor = monitor
        # 使用ResNet-18
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 去掉池化和FC层
        for param in self.feature_extractor[:6].parameters():
            param.requires_grad = False
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, config.video_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.video_channels),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, 3, 96, 96]

        # ResNet提取特征
        features = self.feature_extractor(x)  # [B*T, 512, 3, 3]
        features = self.pool(features)  # [B*T, 512, 1, 1]
        features = features.view(B, T, 512)  # [B, T, 512]

        # 时序处理
        features = features.transpose(1, 2)  # [B, 512, T]
        features = self.temporal_conv(features)  # [B, 256, T]
        features = features.transpose(1, 2)  # [B, T, 256]

        # 监控
        if self.monitor:
            self.monitor.log_data(data=features, location="LipEncoder", data_type="output", processing_step="output")
        return features


class TemporalProcessor(nn.Module):
    def __init__(self, config, monitor=None):
        super().__init__()
        self.monitor = monitor
        self.dim = config.video_channels
        self.temporal_attn = nn.MultiheadAttention(
            self.dim, config.temporal_num_heads, dropout=config.temporal_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        identity = x
        x, _ = self.temporal_attn(self.norm(x), self.norm(x), self.norm(x))
        out = self.norm(identity + x)  # 残差连接

        if self.monitor:
            self.monitor.log_data(data=out, location="TemporalProcessor", data_type="output", processing_step="output")
        return out


class VideoEncoder(nn.Module):
    def __init__(self, config, monitor=None):
        super().__init__()
        self.monitor = monitor
        self.landmark_encoder = LandmarkEncoder(config, monitor)
        self.lip_encoder = LipEncoder(config, monitor)
        self.temporal_processor = TemporalProcessor(config, monitor)

        self.fusion = nn.Sequential(
            nn.Linear(config.landmark_hidden_dim + config.video_channels, config.video_channels),
            nn.LayerNorm(config.video_channels),
            nn.ReLU()
        )

    def forward(self, batch):
        landmarks = batch['landmarks']  # [B, T, 68, 2]
        lip_images = batch['lip_images']  # [B, T, 3, 96, 96]
        confidence = batch.get('confidence', None)  # [B, T]

        if self.monitor:
            self.monitor.log_data(data=landmarks, location="VideoEncoder", data_type="landmarks",
                                  processing_step="input")
            self.monitor.log_data(data=lip_images, location="VideoEncoder", data_type="lip_images",
                                  processing_step="input")

        landmark_features = self.landmark_encoder(landmarks)  # [B, T, 256]
        lip_features = self.lip_encoder(lip_images)  # [B, T, 256]

        fused = self.fusion(torch.cat([landmark_features, lip_features], dim=-1))  # [B, T, 256]
        out = self.temporal_processor(fused)  # [B, T, 256]

        if confidence is not None:
            out = out * confidence.unsqueeze(-1)

        if torch.isnan(out).any():
            print("Warning: NaN values in VideoEncoder output")
            out = torch.nan_to_num(out, 0.0)
        if torch.all(out == 0):
            print("Warning: VideoEncoder output is all zeros")

        if self.monitor:
            self.monitor.log_data(data=out, location="VideoEncoder", data_type="output_features",
                                  processing_step="output")
        return out
