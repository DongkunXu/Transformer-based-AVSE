import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.video_modules import VideoEncoder
from models.audio_modules import AudioProcessor
import torch.nn.functional as F
from utils.losses import AVSELoss
from typing import Optional
from utils.monitor import DataFlowMonitor


class CrossModalFusion(nn.Module):
    """优化的跨模态融合模块"""

    def __init__(self, config, monitor: Optional[DataFlowMonitor] = None):
        super().__init__()
        self.monitor = monitor
        self.dim = config.fusion_dim

        # 简化特征投影
        self.video_proj = nn.Sequential(
            nn.Linear(config.video_channels, self.dim),
            nn.LayerNorm(self.dim)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_channels, self.dim),
            nn.LayerNorm(self.dim)
        )

        # 优化的跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            self.dim,
            config.fusion_heads,
            dropout=config.fusion_dropout,
            batch_first=True
        )

        # 简化的特征融合FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(self.dim * 2, self.dim)
        )

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)

        # 输出投影
        self.output_proj = nn.Linear(self.dim, config.audio_channels)

    def adjust_time_resolution(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Improved time resolution adjustment with better interpolation
        Args:
            x: [B, T, C] Input features
            target_length: Target sequence length
        Returns:
            [B, target_length, C] Adjusted features
        """
        B, T, C = x.shape
        if T == target_length:
            return x

        # 使用三次插值以获得更平滑的结果
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.interpolate(
            x,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        return x.transpose(1, 2)  # [B, target_length, C]

    def forward(self, video_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        """Improved fusion process with better alignment
        Args:
            video_feat: [B, T_video, video_channels] Video features
            audio_feat: [B, T_audio, audio_channels] Audio features
        Returns:
            [B, T_audio, audio_channels] Fused features
        """
        # Monitor inputs
        if self.monitor:
            self.monitor.log_data(
                data=video_feat,
                location="CrossModalFusion",
                data_type="video_features",
                processing_step="input"
            )
            self.monitor.log_data(
                data=audio_feat,
                location="CrossModalFusion",
                data_type="audio_features",
                processing_step="input"
            )

        # 调整音频特征的时序长度到视频帧率
        audio_v = self.adjust_time_resolution(audio_feat, video_feat.size(1))

        # 特征投影
        v = self.video_proj(video_feat)  # [B, T_video, dim]
        a_v = self.audio_proj(audio_v)  # [B, T_video, dim]

        # Monitor projected features
        if self.monitor:
            self.monitor.log_data(
                data=v,
                location="CrossModalFusion",
                data_type="projected_video",
                processing_step="projection"
            )
            self.monitor.log_data(
                data=a_v,
                location="CrossModalFusion",
                data_type="projected_audio",
                processing_step="projection"
            )

        # 跨模态注意力：视频引导音频
        attn_out = self.cross_attn(
            query=self.norm1(a_v),
            key=v,
            value=v
        )[0]
        a_v = a_v + attn_out

        # FFN with residual
        a_v = a_v + self.ffn(self.norm2(a_v))

        # 调整回音频序列长度
        out = self.adjust_time_resolution(a_v, audio_feat.size(1))

        # 投影回音频维度
        out = self.output_proj(out)

        # Monitor output
        if self.monitor:
            self.monitor.log_data(
                data=out,
                location="CrossModalFusion",
                data_type="fused_features",
                processing_step="output"
            )

        return out



class AVSEModel(pl.LightningModule):
    def __init__(self, config, monitor: Optional[DataFlowMonitor] = None):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.monitor = monitor

        # 初始化组件
        self.video_encoder = VideoEncoder(config.model, monitor)
        self.audio_processor = AudioProcessor(config.model, monitor)
        self.fusion = CrossModalFusion(config.model, monitor)
        self.criterion = AVSELoss(config.loss, sample_rate=16000, monitor=monitor)

    def forward(self, batch):
        """Improved forward pass with comprehensive monitoring"""
        # 1. Video encoding
        visual_features = self.video_encoder(batch)

        if self.monitor:
            self.monitor.log_data(
                data=visual_features,
                location="AVSEModel",
                data_type="visual_features",
                processing_step="video_encoding"
            )

        # Check if visual features are valid
        if torch.isnan(visual_features).any():
            print("Warning: NaN values in visual features")
            visual_features = torch.nan_to_num(visual_features, 0.0)

        # 2. Audio encoding
        audio_features = self.audio_processor(batch['mixed_audio'], encoder_only=True)

        if self.monitor:
            self.monitor.log_data(
                data=audio_features,
                location="AVSEModel",
                data_type="audio_features",
                processing_step="audio_encoding"
            )

        # Check if audio features are valid
        if torch.isnan(audio_features).any():
            print("Warning: NaN values in audio features")
            audio_features = torch.nan_to_num(audio_features, 0.0)

        # 3. Feature fusion
        fused_features = self.fusion(visual_features, audio_features)

        if self.monitor:
            self.monitor.log_data(
                data=fused_features,
                location="AVSEModel",
                data_type="fused_features",
                processing_step="feature_fusion"
            )

        # 4. Audio reconstruction
        enhanced_audio = self.audio_processor.decode(fused_features)

        if self.monitor:
            self.monitor.log_data(
                data=enhanced_audio,
                location="AVSEModel",
                data_type="enhanced_audio",
                processing_step="audio_decoding"
            )

        return enhanced_audio

    def training_step(self, batch, batch_idx):
        """Improved training step with better monitoring"""
        # Forward pass
        enhanced_audio = self(batch)

        # Calculate losses
        losses = self.criterion(enhanced_audio, batch['target_audio'])

        # Log individual losses
        for name, value in losses.items():
            self.log(f'train_{name}', value,
                     prog_bar=True if name == 'loss' else False)

        # Monitor gradients periodically
        if self.trainer.global_step % 50 == 0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    module = name.split('.')[0]
                    grad_norm = param.grad.norm().item()
                    self.log(f'grad_norm/{name}', grad_norm)

                    # Summary for progress bar
                    self.log(f'grad_{module}', grad_norm, prog_bar=True)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        """Improved validation step with comprehensive audio monitoring"""
        enhanced_audio = self(batch)

        # Check audio validity
        for audio_name, audio_data in [
            ('enhanced', enhanced_audio),
            ('mixed', batch['mixed_audio']),
            ('target', batch['target_audio'])
        ]:
            if torch.isnan(audio_data).any():
                print(f"Warning: NaN values in {audio_name} audio")
            if torch.all(audio_data == 0):
                print(f"Warning: {audio_name} audio is all zeros")

        # Calculate losses
        losses = self.criterion(enhanced_audio, batch['target_audio'])

        # Log validation metrics
        for name, value in losses.items():
            self.log(f'val_{name}', value,
                     prog_bar=True if name == 'loss' else False)

        # Save audio samples periodically
        if self.current_epoch % 1 == 0 and batch_idx < 1:
            for i in range(min(3, enhanced_audio.size(0))):
                # Log enhanced audio
                self.logger.experiment.add_audio(
                    f'epoch_{self.current_epoch}/enhanced_{i}',
                    enhanced_audio[i].cpu(),
                    self.current_epoch,
                    sample_rate=16000
                )

                # Log mixed input audio
                self.logger.experiment.add_audio(
                    f'epoch_{self.current_epoch}/mixed_{i}',
                    batch['mixed_audio'][i].cpu(),
                    self.current_epoch,
                    sample_rate=16000
                )

                # Log target audio
                self.logger.experiment.add_audio(
                    f'epoch_{self.current_epoch}/target_{i}',
                    batch['target_audio'][i].cpu(),
                    self.current_epoch,
                    sample_rate=16000
                )

        return losses['loss']

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            eps=1e-7
        )

        # 计算总步数
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = steps_per_epoch * self.config.training.num_epochs
        warmup_steps = self.config.training.warmup_steps

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            div_factor=10.0,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }