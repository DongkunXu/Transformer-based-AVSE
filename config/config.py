from dataclasses import dataclass, field
from typing import Tuple, List
import yaml

@dataclass
class PreprocessConfig:
    """音视频预处理配置"""
    # 视频预处理
    video_fps: int = 25
    lip_size: Tuple[int, int] = (96, 96)
    normalize_landmarks: bool = True
    max_frames: int = 75  # 3秒@25fps
    enhance_contrast = True  # 启用对比度增强
    contrast_factor = 1.5  # 对比度增强因子，可调整

    # 音频预处理
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    normalize_audio: bool = True
    audio_max_length: int = 48000  # 3秒@16kHz


@dataclass
class ModelConfig:
    """模型配置"""
    # 视频编码器
    video_channels: int = 256
    landmark_hidden_dim: int = 256

    # 面部关键点编码器参数
    landmark_point_dim: int = 32
    landmark_point_hidden: int = 64
    landmark_num_heads: int = 4
    landmark_dropout: float = 0.1

    # 唇部编码器参数
    lip_init_channels: int = 64
    lip_mid_channels: int = 128
    lip_kernel_size: Tuple[int, int, int] = (1, 3, 3)
    lip_padding: Tuple[int, int, int] = (0, 1, 1)
    lip_pool_size: Tuple[int, int, int] = (1, 2, 2)

    # 时序处理器参数
    temporal_num_heads: int = 8
    temporal_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    temporal_dropout: float = 0.1

    # Conformer块参数
    conformer_expansion: int = 4
    conformer_kernel_size: int = 31
    conformer_num_heads: int = 8
    conformer_dropout: float = 0.1

    # 音频处理器
    audio_channels: int = 256
    audio_layers: int = 4
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512

    # 特征融合
    fusion_dim: int = 256
    fusion_heads: int = 8
    fusion_expansion: int = 4
    fusion_dropout: float = 0.1


@dataclass
class DataConfig:
    """数据配置"""
    root_dir: str = "path/to/dataset"
    max_duration: float = 3.0
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    precision: str = "16-mixed"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class LossConfig:
    """损失函数配置"""
    mag_loss_weight: float = 0.1
    phase_loss_weight: float = 0.05
    si_snr_weight: float = 1
    noise_loss_weight : float = 0.1
    perc_loss_weight : float = 0.1
    highF_loss_weight : float = 0.1
    pesq_loss_weight : float = 0.1
    volume_loss_weight: float = 0.4
    fft_sizes: List[int] = field(default_factory=lambda: [512])
    hop_sizes: List[int] = field(default_factory=lambda: [128])
    win_lengths: List[int] = field(default_factory=lambda: [512])


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """从YAML加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        config = cls()

        # 更新所有配置
        for section in ['data', 'model', 'training', 'preprocess', 'loss']:
            if section in config_dict:
                section_config = getattr(config, section)
                for k, v in config_dict[section].items():
                    setattr(section_config, k, v)

        return config