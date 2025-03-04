import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader
from tabulate import tabulate
from datetime import datetime
from alive_progress import alive_bar
import argparse

from config.config import Config
from data.data_module import AVSEDataset, custom_collate
from models.avse_model import AVSEModel
from pesq import pesq
from pystoi import stoi


class ModelEvaluator:
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str):
        """初始化评估器"""
        self.config = Config.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model = AVSEModel.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
            monitor=None
        )
        self.model.to(self.device)
        self.model.eval()

    def apply_spectral_subtraction(self, audio: torch.Tensor) -> torch.Tensor:
        """应用频谱减法进行降噪"""
        n_fft = self.config.model.n_fft
        hop_length = self.config.model.hop_length
        win_length = self.config.model.win_length

        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        audio = audio.squeeze(1)

        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hamming_window(win_length).to(self.device),
            return_complex=True,
            normalized=True,
            center=True
        )

        noise_frames = int(spec.size(1) * 0.1)
        noise_spec = spec[:, :noise_frames].mean(dim=1, keepdim=True)

        mag = torch.abs(spec)
        phase = torch.angle(spec)
        mag_reduced = torch.clamp(mag - torch.abs(noise_spec), min=0.0)
        enhanced_spec = mag_reduced * torch.exp(1j * phase)

        enhanced = torch.istft(
            enhanced_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hamming_window(win_length).to(self.device),
            normalized=True,
            center=True,
            length=audio.size(-1)
        )

        rms = torch.sqrt(torch.mean(enhanced ** 2))
        if rms > 1e-6:
            enhanced = enhanced * (0.2 / rms)

        return enhanced.unsqueeze(1)

    def evaluate_sample(self, scene_id: str, root_dir: str, output_dir: str) -> Dict[str, float]:
        """评估单个样本"""
        dataset = AVSEDataset(
            root_dir=root_dir,
            split="train",
            config=self.config,
            monitor=None
        )

        try:
            scene_idx = dataset.scene_ids.index(scene_id)
        except ValueError:
            raise ValueError(f"Scene {scene_id} not found in dataset")

        sample = dataset[scene_idx]

        if any(torch.all(sample[k] == 0) for k in ['mixed_audio', 'target_audio']):
            raise ValueError(f"Invalid sample for scene {scene_id}: contains all-zero data")

        batch = custom_collate([sample])
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            enhanced_audio = self.model(batch)

        if torch.all(enhanced_audio == 0):
            raise ValueError("Model output is all zeros")

        denoised_audio = self.apply_spectral_subtraction(enhanced_audio)

        audios = {
            'mixed': batch['mixed_audio'],
            'target': batch['target_audio'],
            'enhanced': enhanced_audio,
            'denoised': denoised_audio
        }

        scene_output_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_output_dir, exist_ok=True)

        metrics = {}
        target_np = batch['target_audio'].squeeze().cpu().numpy()

        for name, audio in audios.items():
            save_audio = audio.squeeze().cpu()
            if save_audio.dim() == 1:
                save_audio = save_audio.unsqueeze(0)

            output_path = os.path.join(scene_output_dir, f"{name}.wav")
            torchaudio.save(
                output_path,
                save_audio,
                self.config.preprocess.sample_rate
            )

            if name != 'target':
                audio_np = audio.squeeze().cpu().numpy()
                length = min(len(audio_np), len(target_np))
                audio_np = audio_np[:length]
                target_np_trim = target_np[:length]

                try:
                    metrics[f"{name}_pesq"] = pesq(
                        self.config.preprocess.sample_rate,
                        target_np_trim,
                        audio_np,
                        'wb'
                    )
                    metrics[f"{name}_stoi"] = stoi(
                        target_np_trim,
                        audio_np,
                        self.config.preprocess.sample_rate,
                        extended=False
                    )
                    metrics[f"{name}_si_snr"] = self.calculate_si_snr(
                        torch.from_numpy(audio_np),
                        torch.from_numpy(target_np_trim)
                    )
                except Exception as e:
                    print(f"Warning: Failed to calculate metrics for {name} in scene {scene_id}: {str(e)}")
                    metrics[f"{name}_pesq"] = float('nan')
                    metrics[f"{name}_stoi"] = float('nan')
                    metrics[f"{name}_si_snr"] = float('nan')

        return metrics

    @staticmethod
    def calculate_si_snr(enhanced: torch.Tensor, target: torch.Tensor) -> float:
        """改进版SI-SNR计算，包含自动对齐和幅度归一化"""

        def _align_signals(x, y):
            """通过互相关进行自动对齐"""
            # 计算互相关（频域加速）
            x = x - x.mean()
            y = y - y.mean()
            X = torch.fft.rfft(x, n=2 * x.shape[0])
            Y = torch.fft.rfft(y, n=2 * y.shape[0])
            cross_corr = torch.fft.irfft(X * Y.conj()).real
            # 寻找最大相关点
            max_lag = torch.argmax(cross_corr)
            # 计算实际偏移量（循环相关处理）
            if max_lag >= x.shape[0]:
                max_lag -= 2 * x.shape[0]
            # 对齐信号
            return torch.roll(x, shifts=max_lag.item())

        # 信号预处理
        enhanced = enhanced.squeeze()
        target = target.squeeze()

        # 自动对齐
        aligned_enhanced = _align_signals(enhanced, target)

        # 幅度归一化（匹配目标信号RMS）
        target_rms = torch.sqrt(torch.mean(target ** 2)) + 1e-8
        enhanced_rms = torch.sqrt(torch.mean(aligned_enhanced ** 2)) + 1e-8
        aligned_enhanced = (aligned_enhanced / enhanced_rms) * target_rms

        # 截断处理（保持长度一致）
        min_len = min(aligned_enhanced.shape[0], target.shape[0])
        aligned_enhanced = aligned_enhanced[:min_len]
        target = target[:min_len]

        # 核心计算
        alpha = (aligned_enhanced * target).sum() / (target * target).sum() + 1e-8
        s_target = alpha * target
        e_noise = aligned_enhanced - s_target

        # 计算能量比
        signal_power = (s_target ** 2).sum() + 1e-8
        noise_power = (e_noise ** 2).sum() + 1e-8

        si_snr = 10 * torch.log10(signal_power / noise_power)
        return si_snr.item()


def generate_scene_ids(start_id: str, end_id: str) -> List[str]:
    """生成场景ID列表"""
    start_num = int(start_id[1:])
    end_num = int(end_id[1:])
    return [f"S{str(i).zfill(5)}" for i in range(start_num, end_num + 1)]


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="AVSE Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset")
    args = parser.parse_args()

    # 配置参数
    CONFIG_PATH = "config/avse_base.yaml"
    CHECKPOINT_PATH = args.checkpoint  # 使用命令行传入的参数
    DATA_ROOT = args.data_root        # 使用命令行传入的参数
    START_SCENE_ID = "S00040"
    END_SCENE_ID = "S00050"

    try:
        current_time = datetime.now().strftime("D%Y%m%d_T%H%M")
        main_output_dir = os.path.join("E:\School_Work\PhD_1\Transformer-AVSE-V5\Test_Output", current_time)
        os.makedirs(main_output_dir, exist_ok=True)

        evaluator = ModelEvaluator(CONFIG_PATH, CHECKPOINT_PATH, main_output_dir)
        scene_ids = generate_scene_ids(START_SCENE_ID, END_SCENE_ID)

        all_metrics = []

        with alive_bar(len(scene_ids), title="Processing Scenes", bar="smooth", spinner="classic") as bar:
            for scene_id in scene_ids:
                try:
                    metrics = evaluator.evaluate_sample(scene_id, DATA_ROOT, main_output_dir)
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error evaluating scene {scene_id}: {str(e)}")
                bar()

        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if not np.isnan(m[key])]
                avg_metrics[key] = np.mean(values) if values else float('nan')

            table_data = [
                ["Target (GT)", "4.500", "1.000", "∞"]
            ]
            headers = ["Audio Type", "PESQ", "STOI", "SI-SNR"]

            for audio_type in ['mixed', 'enhanced', 'denoised']:
                row = [
                    audio_type.capitalize(),
                    f"{avg_metrics[f'{audio_type}_pesq']:.3f}" if not np.isnan(avg_metrics[f'{audio_type}_pesq']) else "N/A",
                    f"{avg_metrics[f'{audio_type}_stoi']:.3f}" if not np.isnan(avg_metrics[f'{audio_type}_stoi']) else "N/A",
                    f"{avg_metrics[f'{audio_type}_si_snr']:.3f}" if not np.isnan(avg_metrics[f'{audio_type}_si_snr']) else "N/A"
                ]
                table_data.append(row)

            print("\nAverage Evaluation Results:")
            print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))
        else:
            print("No valid metrics were collected.")

    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()