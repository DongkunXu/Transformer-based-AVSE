import torch
import numpy as np
import os
from typing import Union, Dict, Any
from datetime import datetime


class DataFlowMonitor:
    """数据流监控工具"""

    def __init__(self, root_dir: str, max_samples: int = 100):
        """
        初始化监控器
        Args:
            root_dir: 日志保存根目录
            max_samples: 每个阶段记录的最大样本数
        """
        self.root_dir = root_dir
        self.max_samples = max_samples
        self.sample_count = 0
        self.log_file = None
        self.log_path = None
        self._initialize_log_file()

    def _initialize_log_file(self):
        """初始化日志文件路径"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(self.root_dir, f'dataflow_monitor_{timestamp}.txt')

    def _ensure_file_open(self):
        """确保文件是打开的"""
        if self.log_file is None or self.log_file.closed:
            self.log_file = open(self.log_path, 'a', encoding='utf-8')

    def _calculate_stats(self, tensor: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """计算张量的统计信息"""
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        # 确保转换为浮点类型
        if tensor.dtype not in [torch.float32, torch.float64]:
            tensor = tensor.float()

        with torch.no_grad():
            stats = {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'norm': torch.norm(tensor).item(),
                'is_nan': torch.isnan(tensor).any().item(),
                'is_inf': torch.isinf(tensor).any().item(),
                'near_zero': (torch.abs(tensor) < 1e-6).float().mean().item()
            }
        return stats

    def log_data(self, data: Union[torch.Tensor, np.ndarray],
                 location: str,
                 data_type: str,
                 processing_step: str,
                 additional_info: Dict[str, Any] = None):
        """记录数据信息
        if self.sample_count >= self.max_samples:
            return"""

        if self.sample_count >= 50:
            return

        self.sample_count += 1

        # 计算统计信息
        stats = self._calculate_stats(data)

        # 构建日志消息
        log_msg = f"\n{'=' * 50}\n"
        log_msg += f"Sample Count: {self.sample_count}\n"
        log_msg += f"Location: {location}\n"
        log_msg += f"Data Type: {data_type}\n"
        log_msg += f"Processing Step: {processing_step}\n"
        log_msg += f"Shape: {data.shape if hasattr(data, 'shape') else None}\n"
        log_msg += f"Dtype: {data.dtype if hasattr(data, 'dtype') else type(data)}\n"

        # 添加统计信息
        log_msg += "\nStatistics:\n"
        for key, value in stats.items():
            log_msg += f"  {key}: {value}\n"

        # 添加额外信息
        if additional_info:
            log_msg += "\nAdditional Info:\n"
            for key, value in additional_info.items():
                log_msg += f"  {key}: {value}\n"

        # 写入日志
        self._ensure_file_open()
        self.log_file.write(log_msg)
        self.log_file.flush()

    def log_gradient(self,
                     model_name: str,
                     param_name: str,
                     gradient: torch.Tensor):
        """记录梯度信息"""
        if gradient is None:
            return

        with torch.no_grad():
            grad_stats = {
                'norm': torch.norm(gradient).item(),
                'mean': gradient.mean().item(),
                'std': gradient.std().item(),
                'max': gradient.max().item(),
                'min': gradient.min().item(),
                'is_zero': (gradient == 0).float().mean().item()
            }

        log_msg = f"\n{'=' * 50}\n"
        log_msg += f"Gradient Info:\n"
        log_msg += f"Model: {model_name}\n"
        log_msg += f"Parameter: {param_name}\n"

        for key, value in grad_stats.items():
            log_msg += f"  {key}: {value}\n"

        self.log_file.write(log_msg)
        self.log_file.flush()

    def close(self):
        """安全关闭文件"""
        if hasattr(self, 'log_file') and self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        """析构时确保文件关闭"""
        self.close()

    def __getstate__(self):
        """控制序列化行为"""
        state = self.__dict__.copy()
        # 不序列化文件对象
        state['log_file'] = None
        return state

    def __setstate__(self, state):
        """控制反序列化行为"""
        self.__dict__.update(state)
        # 需要时重新打开文件
        if self.log_path is not None:
            self._ensure_file_open()