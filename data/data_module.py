import os
import torch
import numpy as np
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, List, Dict
from utils.monitor import DataFlowMonitor
from typing import Optional


def custom_collate(batch):
    """自定义的collate函数，确保批次中的所有样本大小一致"""
    # 检查batch是否为空
    if len(batch) == 0:
        return {}

    # 初始化结果字典
    collated = {
        'landmarks': [],
        'lip_images': [],
        'confidence': [],
        'mixed_audio': [],
        'target_audio': []
    }

    # 收集每个样本的数据
    for sample in batch:
        for key in collated.keys():
            if key in sample:
                collated[key].append(sample[key])

    # 将列表转换为张量，并确保数据类型正确
    for key in collated.keys():
        if len(collated[key]) > 0:
            try:
                # 先堆叠为张量
                collated[key] = torch.stack(collated[key], dim=0)

                # 根据不同类型的数据设置正确的数据类型
                if key in ['landmarks', 'lip_images']:
                    collated[key] = collated[key].float()
                elif key in ['mixed_audio', 'target_audio']:
                    collated[key] = collated[key].float()
                elif key == 'confidence':
                    collated[key] = collated[key].float()

            except Exception as e:
                print(f"Error stacking {key}: {str(e)}")
                print(f"Shapes of items in {key}:")
                for i, item in enumerate(collated[key]):
                    print(f"Item {i} shape: {item.shape}")
                raise

    return collated

class AVSEDataset(Dataset):
    def __init__(self, root_dir: str, split: str, config, monitor: Optional[DataFlowMonitor] = None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.config = config
        self.monitor = monitor

        # 初始化预处理器
        from data.preprocessing import FacialPreprocessor
        # 初始化预处理器
        self.facial_processor = FacialPreprocessor(config.preprocess, monitor)

        # 设置路径
        self.scenes_dir = os.path.join(root_dir, split, 'scenes')
        self.npy_dir = os.path.join(root_dir, split, 'npy')

        # 获取场景列表
        self.scene_ids = self._get_valid_scenes()

    def _get_valid_scenes(self) -> List[str]:
        """获取有效的场景ID列表，确保ID格式正确"""
        scenes = []
        for file in os.listdir(self.npy_dir):
            if file.endswith('_silent.npy'):
                scene_id = file.replace('_silent.npy', '')
                # 确保场景ID格式正确（以S开头）
                if scene_id.startswith('S') and self._check_files(scene_id):
                    scenes.append(scene_id)
        return sorted(scenes)  # 排序确保顺序一致

    def _check_files(self, scene_id: str) -> bool:
        """检查所需文件是否都存在"""
        files = [
            os.path.join(self.npy_dir, f"{scene_id}_silent.npy"),
            os.path.join(self.scenes_dir, f"{scene_id}_mixed.wav"),
            os.path.join(self.scenes_dir, f"{scene_id}_target.wav")
        ]
        return all(os.path.exists(f) for f in files)

    def _load_and_preprocess_audio(self, path: str) -> torch.Tensor:
        """加载并预处理音频数据，RMS 归一化到 0.2，峰值超标时动态压缩"""
        try:
            # 1. 加载原始音频
            waveform, sr = torchaudio.load(path)

            if self.monitor:
                self.monitor.log_data(
                    data=waveform,
                    location="AVSEDataset",
                    data_type="raw_audio",
                    processing_step="load",
                    additional_info={"path": path, "sample_rate": sr}
                )

            # 2. 确保单声道
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

                if self.monitor:
                    self.monitor.log_data(
                        data=waveform,
                        location="AVSEDataset",
                        data_type="mono_audio",
                        processing_step="to_mono"
                    )

            # 3. 处理长度
            target_length = int(self.config.data.max_duration * self.config.preprocess.sample_rate)
            current_length = waveform.size(1)

            if current_length > target_length:
                waveform = waveform[:, :target_length]
            elif current_length < target_length:
                num_repeats = (target_length + current_length - 1) // current_length
                waveform = waveform.repeat(1, num_repeats)
                waveform = waveform[:, :target_length]

            if self.monitor:
                self.monitor.log_data(
                    data=waveform,
                    location="AVSEDataset",
                    data_type="length_adjusted_audio",
                    processing_step="length_adjustment",
                    additional_info={"target_length": target_length}
                )

            # 4. 归一化处理
            if self.config.preprocess.normalize_audio:
                rms = torch.sqrt(torch.mean(waveform ** 2))

                if rms < 1e-6:
                    # 极小值处理，避免无效音频
                    waveform = waveform * (1e-2 / (rms + 1e-8))
                else:
                    # 第一次归一化到 RMS = 0.2
                    target_rms = 0.2
                    waveform = waveform * (target_rms / rms)

                    # 检查峰值并应用动态压缩
                    threshold = 0.98  # 最大峰值容忍度
                    compression_start = 0.95  # 开始压缩的幅度
                    max_amplitude = torch.max(torch.abs(waveform))

                    if max_amplitude > compression_start:
                        # 对超过 compression_start 的部分应用软限幅
                        exceed_mask = torch.abs(waveform) > compression_start
                        if exceed_mask.any():
                            exceed_values = waveform[exceed_mask]
                            # 使用 tanh 进行软限幅，限制到 threshold
                            soft_clipped = threshold * torch.tanh(
                                (exceed_values - compression_start) / (threshold - compression_start))
                            # 平滑融合压缩部分和原始部分
                            waveform[exceed_mask] = compression_start + soft_clipped
                            # 保持符号一致
                            waveform[exceed_mask] = torch.sign(exceed_values) * torch.abs(waveform[exceed_mask])

                        # 压缩后重新调整 RMS 到 0.2
                        new_rms = torch.sqrt(torch.mean(waveform ** 2))
                        if new_rms > 1e-6:
                            waveform = waveform * (target_rms / new_rms)

                    # 记录归一化后的数据
                    if self.monitor:
                        self.monitor.log_data(
                            data=waveform,
                            location="AVSEDataset",
                            data_type="normalized_audio",
                            processing_step="normalization",
                            additional_info={
                                "rms": rms.item(),
                                "max_amplitude": max_amplitude.item(),
                                "reference_rms": None  # 不再使用 reference_rms
                            }
                        )

            return waveform

        except Exception as e:
            print(f"Error processing audio {path}: {str(e)}")
            return torch.zeros(1, target_length)


    def _load_facial_data(self, scene_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """加载并预处理面部数据"""
        npy_path = os.path.join(self.npy_dir, f"{scene_id}_silent.npy")
        frames_data = np.load(npy_path, allow_pickle=True)
        return self.facial_processor.process(frames_data)

    def _synchronize_av_data(self, audio_path: str, frames_data: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
        """同步音频和视频数据"""
        # 获取音频时长和帧率
        audio_info = torchaudio.info(audio_path)
        audio_duration = audio_info.num_frames / audio_info.sample_rate
        frame_duration = audio_duration / len(frames_data)

        # 计算每帧对应的音频样本索引
        frame_indices = []
        for i in range(len(frames_data)):
            start_time = i * frame_duration
            end_time = (i + 1) * frame_duration
            start_sample = int(start_time * audio_info.sample_rate)
            end_sample = int(end_time * audio_info.sample_rate)
            frame_indices.append((start_sample, end_sample))

        # 将时间戳信息添加到帧数据中
        for i, frame_data in enumerate(frames_data):
            frame_data['audio_indices'] = frame_indices[i]

        return frame_indices, frames_data

    def _get_empty_batch(self) -> Dict[str, torch.Tensor]:
        """返回空批次数据"""
        print(f"Warning: Loading failed, returning all-zero batch")
        return {
            'scene_id': '',
            'landmarks': torch.zeros(75, 68, 2),
            'lip_images': torch.zeros(75, 3, 96, 96),
            'confidence': torch.zeros(75),
            'mixed_audio': torch.zeros(1, int(self.config.preprocess.sample_rate * self.config.data.max_duration)),
            'target_audio': torch.zeros(1, int(self.config.preprocess.sample_rate * self.config.data.max_duration)),
            'frame_indices': torch.zeros(75, 2)
        }

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_id = self.scene_ids[idx]
        try:
            # 加载音频路径
            mixed_path = os.path.join(self.scenes_dir, f"{scene_id}_mixed.wav")
            target_path = os.path.join(self.scenes_dir, f"{scene_id}_target.wav")

            # 检查文件是否存在
            if not os.path.exists(mixed_path):
                print(f"Mixed audio file not found: {mixed_path}")
                return self._get_empty_batch()
            if not os.path.exists(target_path):
                print(f"Target audio file not found: {target_path}")
                return self._get_empty_batch()

            # 加载面部数据
            npy_path = os.path.join(self.npy_dir, f"{scene_id}_silent.npy")
            frames_data = np.load(npy_path, allow_pickle=True)

            # 同步音视频数据
            frame_indices, frames_data = self._synchronize_av_data(mixed_path, frames_data)

            # 处理面部数据
            landmarks, lip_images, confidence = self.facial_processor.process(frames_data)

            # 加载音频数据 - 独立处理 mixed 和 target
            mixed_audio = self._load_and_preprocess_audio(mixed_path)
            target_audio = self._load_and_preprocess_audio(target_path)

            if self.monitor:

                self.monitor.log_data(
                        data=landmarks,
                        location="AVSEDataset",
                        data_type="landmarks",
                        processing_step="load",
                        additional_info={"scene_id": scene_id}
                    )

                self.monitor.log_data(
                        data=lip_images,
                        location="AVSEDataset",
                        data_type="lip_images",
                        processing_step="load"
                    )

            # 检查音频是否全为零
            if torch.all(mixed_audio == 0):
                print(f"Warning: Mixed audio is all zeros for scene {scene_id}")
            if torch.all(target_audio == 0):
                print(f"Warning: Target audio is all zeros for scene {scene_id}")

            return {
                'landmarks': landmarks,  # [T, 68, 2]
                'lip_images': lip_images,  # [T, 3, 96, 96]
                'confidence': confidence,  # [T]
                'mixed_audio': mixed_audio,  # [1, samples]
                'target_audio': target_audio  # [1, samples]
            }

        except Exception as e:
            print(f"Error loading scene {scene_id}: {str(e)}")
            return self._get_empty_batch()


class AVSEDataModule(pl.LightningModule):
    def __init__(self, config, monitor: Optional[DataFlowMonitor] = None):
        """
        初始化数据模块
        Args:
            config: 配置对象
            monitor: 可选的数据监控器
        """
        super().__init__()
        # 保存所有参数为实例变量
        self.config = config
        self.monitor = monitor

        # 初始化数据集为 None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = AVSEDataset(
                self.config.data.root_dir,
                "train",
                self.config,
                self.monitor
            )
            self.val_dataset = AVSEDataset(
                self.config.data.root_dir,
                "dev",
                self.config,
                self.monitor
            )

            if self.monitor:
                self.monitor.log_data(
                        data=torch.tensor([0]),  # 占位数据
                        location="AVSEDataModule",
                        data_type="dataset_info",
                        processing_step="setup",
                        additional_info={
                            "train_size": len(self.train_dataset),
                            "val_size": len(self.val_dataset)
                        }
                    )

    def train_dataloader(self):

        if self.train_dataset is None:
            raise RuntimeError("Please call setup() before accessing train_dataloader")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=True,
            collate_fn=custom_collate
        )

    def val_dataloader(self):

        """返回验证数据加载器"""
        if self.val_dataset is None:
            raise RuntimeError("Please call setup() before accessing val_dataloader")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False,
            collate_fn=custom_collate
        )