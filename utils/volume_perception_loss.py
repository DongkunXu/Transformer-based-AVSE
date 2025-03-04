import torch
from typing import Optional
from utils.monitor import DataFlowMonitor


def volume_perception_loss(pred: torch.Tensor, target: torch.Tensor,
                           frame_length: int = 2048, hop_length: int = 512,
                           monitor: Optional[DataFlowMonitor] = None) -> torch.Tensor:
    """计算音频音量感知损失，比较预测音频和目标音频的音量轮廓

    Args:
        pred: 预测音频 [B, 1, T]
        target: 目标音频 [B, 1, T]
        frame_length: 帧长度，用于计算短时能量
        hop_length: 帧移，用于计算短时能量
        monitor: 可选的数据监控器

    Returns:
        归一化到 [0, 1] 范围的音量差异损失，值越小表示音量越相似
    """
    # 确保输入是正确的形状
    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
    if target.dim() == 2:
        target = target.unsqueeze(1)

    batch_size = pred.shape[0]

    # 1. 计算整体RMS能量比例
    pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=-1))
    target_rms = torch.sqrt(torch.mean(target ** 2, dim=-1))

    # 避免除零
    eps = 1e-8
    rms_ratio = torch.abs(pred_rms / (target_rms + eps) - 1.0)

    # 2. 计算短时音量包络
    # 创建窗口
    window = torch.hann_window(frame_length, device=pred.device)

    # 计算短时能量函数
    def compute_energy_envelope(audio):
        # 重塑为2D张量 [batch_size, time]
        audio_flat = audio.reshape(batch_size, -1)

        # 初始化输出张量
        num_frames = (audio_flat.shape[1] - frame_length) // hop_length + 1
        energy = torch.zeros(batch_size, num_frames, device=audio.device)

        # 计算每一帧的能量
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio_flat[:, start:end] * window
            energy[:, i] = torch.mean(frame ** 2, dim=1)

        # 平滑处理
        smoothed_energy = torch.nn.functional.avg_pool1d(
            energy.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)

        # 归一化到 [0, 1]
        max_energy = torch.max(smoothed_energy, dim=1, keepdim=True)[0]
        min_energy = torch.min(smoothed_energy, dim=1, keepdim=True)[0]
        range_energy = max_energy - min_energy + eps
        normalized_energy = (smoothed_energy - min_energy) / range_energy

        return normalized_energy

    # 计算两个音频的能量包络
    pred_env = compute_energy_envelope(pred)
    target_env = compute_energy_envelope(target)

    # 3. 计算包络相似性损失
    env_loss = torch.mean(torch.abs(pred_env - target_env), dim=1)

    # 4. 计算峰值比例损失
    pred_peak = torch.max(torch.abs(pred), dim=2)[0]
    target_peak = torch.max(torch.abs(target), dim=2)[0]
    peak_ratio = torch.abs(pred_peak / (target_peak + eps) - 1.0)

    # 5. 组合损失并归一化到 [0, 1]
    combined_loss = (0.4 * rms_ratio + 0.4 * env_loss + 0.2 * peak_ratio).mean()
    normalized_loss = torch.clamp(combined_loss, 0.0, 1.0)

    if monitor:
        monitor.log_data(
            data=normalized_loss,
            location="VolumeLoss",
            data_type="volume_loss",
            processing_step="output",
            additional_info={
                "rms_ratio": rms_ratio.mean().item(),
                "env_loss": env_loss.mean().item(),
                "peak_ratio": peak_ratio.mean().item()
            }
        )

    return normalized_loss