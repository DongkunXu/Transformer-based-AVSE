import torch
import torch.nn as nn
import torch.nn.functional as F


class STOILoss(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128, win_length=512, n_bands=15, freq_low=150,
                 freq_high=8000):
        """
        Args:
            sample_rate (int): 采样率，默认为 16000 Hz
            n_fft (int): STFT 的 FFT 大小
            hop_length (int): STFT 的步长
            win_length (int): 窗长
            n_bands (int): 频带数，默认 15 个 1/3 倍频程频带
            freq_low (float): 最低频率，默认 150 Hz
            freq_high (float): 最高频率，默认 8000 Hz
        """
        super(STOILoss, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_bands = n_bands
        self.eps = 1e-8

        # 注册汉宁窗到 GPU
        self.register_buffer('window', torch.hann_window(win_length))

        # 计算 1/3 倍频程频带的边界
        freq_bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        center_freqs = torch.logspace(torch.log10(torch.tensor(freq_low, dtype=torch.float32)),
                                      torch.log10(torch.tensor(freq_high, dtype=torch.float32)),
                                      steps=n_bands + 2)[:-1]
        band_edges = torch.sqrt(center_freqs[:-1] * center_freqs[1:])
        self.register_buffer('band_edges', band_edges)
        self.register_buffer('freq_bins', freq_bins)

        # 创建频带滤波器矩阵 (n_bins, n_bands)
        filter_bank = torch.zeros((n_fft // 2 + 1, n_bands))
        for i in range(n_bands):
            low = band_edges[i] if i == 0 else band_edges[i - 1]
            high = band_edges[i]
            filter_bank[:, i] = ((freq_bins >= low) * (freq_bins <= high)).float()
        self.register_buffer('filter_bank', filter_bank)

    def stft(self, x):
        """计算 STFT，输入为 (batch, time)，输出为 (batch, freq, time)"""
        stft = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            normalized=True
        )
        return torch.abs(stft) ** 2  # 返回功率谱

    def apply_filter_bank(self, power_spec):
        """将功率谱映射到频带，输入 (batch, freq, time)，输出 (batch, n_bands, time)"""
        return torch.einsum('bft,fn->bnt', power_spec, self.filter_bank)

    def compute_envelope(self, band_spec, frame_length=30):
        """计算短时包络，输入 (batch, n_bands, time)，frame_length 为帧数"""
        # 创建多通道卷积核，每个通道一个
        kernel = torch.ones(self.n_bands, 1, frame_length, device=band_spec.device) / frame_length
        # 使用 groups 参数确保每个通道独立计算
        envelope = F.conv1d(band_spec, kernel, padding=frame_length // 2, groups=self.n_bands)
        return envelope[..., :band_spec.size(-1)]  # 裁剪到原始长度

    def stoi_score(self, pred_env, target_env):
        """计算 STOI 分数，输入为包络 (batch, n_bands, time)"""
        # 归一化
        pred_mean = pred_env.mean(dim=-1, keepdim=True)
        target_mean = target_env.mean(dim=-1, keepdim=True)
        pred_std = torch.sqrt(torch.var(pred_env, dim=-1, keepdim=True) + self.eps)
        target_std = torch.sqrt(torch.var(target_env, dim=-1, keepdim=True) + self.eps)

        pred_norm = (pred_env - pred_mean) / pred_std
        target_norm = (target_env - target_mean) / target_std

        # 计算相关系数
        corr = torch.sum(pred_norm * target_norm, dim=-1) / (pred_norm.shape[-1] + self.eps)
        return corr.mean(dim=-1)  # 平均所有频带的相关系数

    def forward(self, pred, target):
        """
        计算 STOI 损失
        Args:
            pred (torch.Tensor): 预测信号，(batch, 1, time)
            target (torch.Tensor): 目标信号，(batch, 1, time)
        Returns:
            torch.Tensor: STOI 损失，负值越小越好
        """
        # 计算 STFT 功率谱
        pred_power = self.stft(pred)
        target_power = self.stft(target)

        # 应用滤波器组
        pred_band = self.apply_filter_bank(pred_power)
        target_band = self.apply_filter_bank(target_power)

        # 计算短时包络
        pred_env = self.compute_envelope(pred_band)
        target_env = self.compute_envelope(target_band)

        # 计算 STOI 分数并转为损失
        stoi = self.stoi_score(pred_env, target_env)
        loss = -stoi.mean()  # 负值作为损失

        return loss
