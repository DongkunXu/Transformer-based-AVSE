import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.monitor import DataFlowMonitor


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConformerBlock(nn.Module):
    """优化的Conformer块"""

    def __init__(self, config):
        super().__init__()
        dim = config.audio_channels
        num_heads = config.conformer_num_heads
        kernel_size = config.conformer_kernel_size
        dropout = config.conformer_dropout

        # 为每个子模块添加输入归一化
        self.ff_norm = nn.LayerNorm(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.conv_norm = nn.LayerNorm(dim)
        # 添加最终的输出归一化
        self.final_norm = nn.LayerNorm(dim)

        # Feed Forward模块
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * config.conformer_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * config.conformer_expansion, dim),
            nn.Dropout(dropout)
        )

        # 多头自注意力
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # 优化的卷积模块 - 调整了归一化层的位置
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.BatchNorm1d(dim * 2),  # 移到GLU之前
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # 1. Feed Forward模块
        x = x + self.ff(self.ff_norm(x))

        # 2. 多头自注意力
        x = x + self.attn(
            self.attn_norm(x), self.attn_norm(x), self.attn_norm(x),
            key_padding_mask=mask
        )[0]

        # 3. 卷积模块
        conv_x = self.conv_norm(x).transpose(1, 2)
        x = x + self.conv(conv_x).transpose(1, 2)

        # 4. 最终归一化
        return self.final_norm(x)


class AudioProcessor(nn.Module):
    """优化的音频处理器"""

    def __init__(self, config, monitor: Optional[DataFlowMonitor] = None):
        super().__init__()
        self.config = config
        self.monitor = monitor

        # STFT 参数
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.register_buffer('window', torch.hamming_window(config.win_length)) # 使用更适合保留高频的窗函数，例如Hamming窗（比Hann窗边缘衰减更少）

        # 特征提取
        self.freq_bins = self.n_fft // 2 + 1
        self.input_proj = nn.Sequential(
            nn.Conv2d(2, config.audio_channels, 1),
            nn.BatchNorm2d(config.audio_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        # 简化的特征投影,保留更多信息
        self.feature_proj = nn.Sequential(
            nn.Linear(self.freq_bins * config.audio_channels, config.audio_channels * 2),
            nn.LayerNorm(config.audio_channels * 2),
            nn.ReLU(),
            nn.Linear(config.audio_channels * 2, config.audio_channels),
            nn.LayerNorm(config.audio_channels),
            nn.Dropout(0.1)
        )

        self.pos_enc = PositionalEncoding(config.audio_channels)
        self.encoder_layers = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.audio_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.audio_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(config.audio_channels, config.audio_channels * 2),
            nn.LayerNorm(config.audio_channels * 2),
            nn.ReLU(),
            nn.Linear(config.audio_channels * 2, self.freq_bins * 2),
            nn.Tanh()
        )

    def stft(self, x):
        """优化的STFT计算"""

        if self.monitor:
            self.monitor.log_data(
                data=x,
                location="AudioProcessor",
                data_type="audio",
                processing_step="pre_stft"
            )

        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            normalized=True,
            center=True
        )

        # 转换为实部和虚部
        spec_real_imag = torch.stack([spec.real, spec.imag], dim=1)

        # 单次动态范围调整
        max_abs = torch.max(torch.abs(spec_real_imag))
        if max_abs > 1e-6:  # 防止除零
            spec_real_imag = spec_real_imag / max_abs

        if self.monitor:
            self.monitor.log_data(
                data=spec_real_imag,
                location="AudioProcessor",
                data_type="spectrogram",
                processing_step="post_stft"
            )

        return spec_real_imag

    def istft(self, spec):
        """优化的ISTFT计算"""
        # 处理异常值
        if torch.isnan(spec).any():
            spec = torch.where(torch.isnan(spec), torch.zeros_like(spec), spec)

        # 单次幅值限制
        max_abs = torch.max(torch.abs(spec))
        if max_abs > 1.0:
            spec = spec / max_abs

        # 创建复数频谱
        complex_spec = torch.complex(
            spec[:, 0].float(),
            spec[:, 1].float()
        )

        # ISTFT转换
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            return_complex=False
        )

        # 单次输出归一化
        waveform = waveform.unsqueeze(1)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-6:
            waveform = waveform * (0.98 / max_val)

        return waveform

    def encode(self, x):
        """编码过程"""
        x_freq = self.stft(x)
        x = self.input_proj(x_freq)
        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2).reshape(B, T, -1)
        x = self.feature_proj(x)
        x = self.pos_enc(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return x

    def decode(self, x):
        """解码过程"""
        for layer in self.decoder_layers:
            x = layer(x)

        B, T = x.shape[:2]
        spec = self.output_proj(x)
        spec = spec.view(B, T, 2, self.freq_bins).permute(0, 2, 3, 1)

        return self.istft(spec)

    def forward(self, x, encoder_only=False):
        encoded = self.encode(x)

        if encoder_only:
            return encoded

        return self.decode(encoded)