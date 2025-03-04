import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config.config import LossConfig
from utils.STOI import STOILoss
from utils.monitor import DataFlowMonitor
from pystoi import stoi  # 需要安装 pystoi
from pesq import pesq as pesq_func
from utils.volume_perception_loss import volume_perception_loss

def cal_si_snr(source: torch.Tensor, estimate_source: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """计算SI-SNR损失，增加全零和小值检测"""
    if source.dim() == 3:
        source = source.squeeze(1)
        estimate_source = estimate_source.squeeze(1)

    source_energy = torch.sum(source ** 2, dim=-1)
    estimate_energy = torch.sum(estimate_source ** 2, dim=-1)
    if torch.any(source_energy < eps):
        print(f"Warning: Source signal has very small or zero energy: {source_energy}")
    if torch.any(estimate_energy < eps):
        print(f"Warning: Estimate signal has very small or zero energy: {estimate_energy}")

    source = source - torch.mean(source, dim=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

    source_norm = torch.norm(source, dim=-1, keepdim=True)
    estimate_norm = torch.norm(estimate_source, dim=-1, keepdim=True)
    source = source / (source_norm + eps)
    estimate_source = estimate_source / (estimate_norm + eps)

    dot_product = torch.sum(estimate_source * source, dim=-1, keepdim=True)
    s_target = dot_product * source
    e_noise = estimate_source - s_target

    target_power = torch.sum(s_target ** 2, dim=-1)
    noise_power = torch.sum(e_noise ** 2, dim=-1)
    if torch.any(noise_power < eps):
        print(f"Warning: Noise power is very small: {noise_power}")
    si_snr = 10 * torch.log10((target_power + eps) / (noise_power + eps))

    return -torch.mean(si_snr)


def noise_energy_loss(pred: torch.Tensor, target: torch.Tensor, energy_threshold: float = 0.01,
                      eps: float = 1e-6, monitor: Optional[DataFlowMonitor] = None) -> torch.Tensor:
    """计算预测音频在低能量区域的噪声能量"""
    target_energy = torch.mean(target ** 2, dim=-1, keepdim=True)
    silence_mask = (target_energy < energy_threshold).float()

    mask_ratio = torch.mean(silence_mask)
    if mask_ratio < 0.05:  # 静音片段少于5%，动态降低权重
        silence_weight = mask_ratio / 0.05
        if monitor:
            monitor.log_data(data=torch.tensor(mask_ratio), location="NoiseEnergyLoss", data_type="silence_mask_ratio",
                             processing_step="calculation", additional_info={"warning": "Low silence ratio"})
    else:
        silence_weight = 1.0

    pred_noise_energy = torch.mean((pred * silence_mask) ** 2, dim=-1)
    noise_loss = torch.mean(torch.clamp(pred_noise_energy, min=eps, max=1.0)) * silence_weight

    if monitor:
        monitor.log_data(data=noise_loss, location="NoiseEnergyLoss", data_type="noise_loss", processing_step="output")

    return noise_loss


def high_freq_loss(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024, hop_length: int = 256,
                   sample_rate: int = 16000, monitor: Optional[DataFlowMonitor] = None) -> torch.Tensor:
    """改进的高频损失，覆盖 2000 Hz 到 8000 Hz，分段加权"""
    pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=hop_length, return_complex=True)
    target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=hop_length, return_complex=True)

    freq_bins = n_fft // 2 + 1  # 513 bins for n_fft=1024
    freq_resolution = sample_rate / 2 / (freq_bins - 1)  # ≈ 15.59 Hz/bin

    # 定义频率范围
    cutoff_2000_bin = int(2000 / freq_resolution)  # ≈ 128
    cutoff_4000_bin = int(4000 / freq_resolution)  # ≈ 256
    cutoff_8000_bin = int(8000 / freq_resolution)  # ≈ 512

    # 计算 STFT 幅值
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)

    # 分段计算损失
    mid_freq_loss = F.l1_loss(pred_mag[:, cutoff_2000_bin:cutoff_4000_bin],
                              target_mag[:, cutoff_2000_bin:cutoff_4000_bin])  # 2000~4000 Hz
    high_freq_loss = F.l1_loss(pred_mag[:, cutoff_4000_bin:cutoff_8000_bin],
                               target_mag[:, cutoff_4000_bin:cutoff_8000_bin])  # 4000~8000 Hz

    # 加权组合（高频部分赋予更高权重）
    total_hf_loss = 1.0 * mid_freq_loss + 1.0 * high_freq_loss  # 4000~8000 Hz 权重更高

    if monitor:
        monitor.log_data(data=total_hf_loss, location="HighFreqLoss", data_type="hf_loss",
                         processing_step="output", additional_info={"mid_freq_loss": mid_freq_loss.item(),
                                                                    "high_freq_loss": high_freq_loss.item()})

    return total_hf_loss

def pesq_loss(pred: torch.Tensor, target: torch.Tensor, sample_rate: int = 16000,
              monitor: Optional['DataFlowMonitor'] = None) -> torch.Tensor:
    """计算 PESQ 感知损失，使用 pesq 包的实现"""
    # 将 Tensor 转换为 NumPy 数组，确保是 1D 信号
    pred_np = pred.detach().cpu().numpy().squeeze()
    target_np = target.detach().cpu().numpy().squeeze()

    pesq_scores = []

    # 确保 pred_np 和 target_np 是一维数组或批量一维数组
    if pred_np.ndim == 1:  # 单样本情况
        pred_np = [pred_np]
        target_np = [target_np]
    elif pred_np.ndim != 2:  # 检查维度是否正确
        raise ValueError("Input tensors must be 1D or 2D (batch_size, signal_length)")

    for p, t in zip(pred_np, target_np):
        try:
            # pesq-0.0.4 的接口：pesq(fs, ref, deg, mode='wb' 或 'nb')
            # ref 是参考信号，deg 是增强信号，fs 是采样率
            score = pesq_func(sample_rate, t, p, 'wb')  # 'wb' 表示宽带，适用于 16kHz
            pesq_scores.append(max(score, -0.5))  # 限制下界，与原代码保持一致
        except Exception as e:
            print(f"Warning: PESQ calculation failed: {e}")
            pesq_scores.append(-0.5)  # 默认最低值

    pesq_score = torch.tensor(pesq_scores, dtype=torch.float32).mean().to(pred.device)
    pesq_loss = -pesq_score  # 转换为损失，负值越小越好

    if monitor:
        monitor.log_data(data=pesq_loss, location="PesqLoss", data_type="pesq_loss",
                         processing_step="output", additional_info={"raw_score": pesq_score.item()})

    return pesq_loss


class SimpleSTFTLoss(nn.Module):
    """改进的 STFT 损失，增强高频感知"""

    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512,
                 sample_rate: int = 16000, monitor: Optional[DataFlowMonitor] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.monitor = monitor
        self.register_buffer('window', torch.hann_window(win_length))

        # 频带加权
        freq_bins = n_fft // 2 + 1
        freq_resolution = sample_rate / 2 / (freq_bins - 1)
        weights = torch.ones(freq_bins, dtype=torch.float32)
        weights[int(1000 / freq_resolution):int(3000 / freq_resolution)] *= 1.5  # 1000~3000 Hz
        weights[int(3000 / freq_resolution):] *= 2.0  # 3000 Hz 以上
        self.register_buffer('freq_weights', weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.monitor:
            self.monitor.log_data(data=x, location="STFTLoss", data_type="pred_audio", processing_step="input")
            self.monitor.log_data(data=y, location="STFTLoss", data_type="target_audio", processing_step="input")

        if torch.isnan(x).any() or torch.isnan(y).any():
            print("Warning: NaN values detected in STFT loss input")

        x_stft = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window, return_complex=True, normalized=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window, return_complex=True, normalized=True)

        if self.monitor:
            self.monitor.log_data(data=x_stft, location="STFTLoss", data_type="pred_stft",
                                  processing_step="stft_calculation")
            self.monitor.log_data(data=y_stft, location="STFTLoss", data_type="target_stft",
                                  processing_step="stft_calculation")

        eps = 1e-6
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # 加权 Magnitude Loss，去除对数变换
        mag_diff = torch.abs(x_mag - y_mag)  # [batch, freq_bins, time]
        weighted_mag_loss = torch.mean(mag_diff * self.freq_weights[:, None], dim=[1, 2])

        # Phase Loss，提高权重并移除 mag_mask
        x_phase = torch.angle(x_stft)
        y_phase = torch.angle(y_stft)
        phase_diff = torch.abs(x_phase - y_phase)  # 相位差在 [-π, π]
        phase_loss = torch.mean(phase_diff * self.freq_weights[:, None], dim=[1, 2])  # 加权

        mag_loss = torch.mean(weighted_mag_loss)
        phase_loss = torch.mean(phase_loss)

        if self.monitor:
            self.monitor.log_data(data=mag_loss, location="STFTLoss", data_type="magnitude_loss",
                                  processing_step="loss_calculation")
            self.monitor.log_data(data=phase_loss, location="STFTLoss", data_type="phase_loss",
                                  processing_step="loss_calculation")

        return {'mag_loss': mag_loss, 'phase_loss': phase_loss}


class ImprovedAVSELoss(nn.Module):
    """改进的AVSE损失函数，调用独立函数"""

    def __init__(self, config: LossConfig, sample_rate: int = 16000, monitor: Optional[DataFlowMonitor] = None):
        super().__init__()
        self.config = config
        self.sample_rate = sample_rate
        self.monitor = monitor
        self.stft_loss = SimpleSTFTLoss(n_fft=config.fft_sizes[0], hop_length=config.hop_sizes[0],
                                        win_length=config.win_lengths[0], sample_rate=sample_rate, monitor=monitor)
        # === 修改 1：添加 STOILoss 初始化 ===
        self.stoi_loss = STOILoss(sample_rate=sample_rate, n_fft=512, hop_length=128)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_details: bool = True) -> Dict[str, torch.Tensor]:
        if self.monitor:
            # 记录输入音频的 RMS，便于调试
            pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=-1)).mean()
            target_rms = torch.sqrt(torch.mean(target ** 2, dim=-1)).mean()
            self.monitor.log_data(data=pred, location="AVSELoss", data_type="pred_audio", processing_step="input",
                                  additional_info={"rms": pred_rms.item()})
            self.monitor.log_data(data=target, location="AVSELoss", data_type="target_audio", processing_step="input",
                                  additional_info={"rms": target_rms.item()})

        si_snr_loss = cal_si_snr(pred, target)
        if self.monitor:
            self.monitor.log_data(data=torch.tensor(si_snr_loss), location="AVSELoss", data_type="si_snr_loss",
                                  processing_step="calculation")

        stft_losses = self.stft_loss(pred, target)
        if self.monitor:
            for loss_name, loss_value in stft_losses.items():
                self.monitor.log_data(data=loss_value, location="AVSELoss", data_type=f"stft_{loss_name}",
                                      processing_step="calculation")

        noise_loss = noise_energy_loss(pred, target, energy_threshold=0.01, monitor=self.monitor)
        stoi_loss = self.stoi_loss(pred, target)  # 计算 STOI 损失
        hf_loss = high_freq_loss(pred, target, n_fft=1024, hop_length=256, sample_rate=self.sample_rate, monitor=self.monitor)
        volume_loss = volume_perception_loss(pred, target, monitor=self.monitor)
        # pesq_loss_value = pesq_loss(pred, target, sample_rate=self.sample_rate, monitor=self.monitor)

        total_loss = (
            self.config.si_snr_weight * si_snr_loss +
            self.config.mag_loss_weight * stft_losses['mag_loss'] +
            self.config.phase_loss_weight * stft_losses['phase_loss'] +
            self.config.noise_loss_weight * noise_loss +  # 噪声抑制权重
            self.config.perc_loss_weight * stoi_loss +  # 感知损失权重 stoi
            self.config.highF_loss_weight * hf_loss +  #高频损失
            self.config.volume_loss_weight * volume_loss  # 新增音量损失
            # self.config.pesq_loss_weight * pesq_loss_value  # 新增 PESQ 权重
        )

        if self.monitor:
            self.monitor.log_data(data=torch.tensor(total_loss), location="AVSELoss", data_type="total_loss",
                                  processing_step="output")

        if return_details:
            return {
                'loss': total_loss,
                'si_snr_loss': si_snr_loss,
                'mag_loss': stft_losses['mag_loss'],
                'phase_loss': stft_losses['phase_loss'],
                'stoi_loss': stoi_loss,
                'volume_loss': volume_loss,
            }
        return {'loss': total_loss}


AVSELoss = ImprovedAVSELoss