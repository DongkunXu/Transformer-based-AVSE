import torch
import torchaudio
import numpy as np
from typing import Tuple, Dict, List
import logging
from typing import Optional
from utils.monitor import DataFlowMonitor

class FacialPreprocessor:
    """简化的面部特征预处理器"""

    def __init__(self, config, monitor: Optional[DataFlowMonitor] = None):
        self.config = config
        self.monitor = monitor
        self.target_frames = 75  # 固定帧数
        # 从配置中读取对比度增强参数，默认为不启用
        self.enhance_contrast = getattr(config, 'enhance_contrast', False)
        self.contrast_factor = getattr(config, 'contrast_factor', 1.5)  # 默认增强因子为 1.5

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """标准化面部关键点坐标"""
        # 计算面部边界框
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)

        # 计算范围并避免除零
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_x = range_x if range_x > 1e-6 else 1.0
        range_y = range_y if range_y > 1e-6 else 1.0

        # 归一化到[-1, 1]范围
        normalized = landmarks.copy()
        normalized[:, 0] = (landmarks[:, 0] - min_x) / range_x * 2 - 1
        normalized[:, 1] = (landmarks[:, 1] - min_y) / range_y * 2 - 1

        return normalized

    def _process_lip_image(self, lip_image: np.ndarray) -> torch.Tensor:
        """简化的唇部图像处理，新增对比度调整"""
        # 转换为 float 类型
        lip_tensor = torch.from_numpy(lip_image).float()

        # 如果原始范围是 0-255，归一化到 [0, 1]
        if lip_tensor.max() > 1.0:
            lip_tensor = lip_tensor / 255.0

        # 可选的对比度增强
        if self.enhance_contrast:
            # 计算当前图像的最小值和最大值
            min_val = lip_tensor.min()
            max_val = lip_tensor.max()
            if max_val > min_val:  # 避免除零
                # 线性对比度增强
                lip_tensor = (lip_tensor - min_val) * self.contrast_factor / (max_val - min_val)
                # 限制范围到 [0, 1]
                lip_tensor = torch.clamp(lip_tensor, 0.0, 1.0)

            # 记录增强后的数据
            if self.monitor:
                self.monitor.log_data(
                    data=lip_tensor,
                    location="FacialPreprocessor",
                    data_type="lip_image_enhanced",
                    processing_step="contrast_adjustment",
                    additional_info={"contrast_factor": self.contrast_factor}
                )

        # 调整通道顺序 [H, W, C] -> [C, H, W]
        if lip_tensor.ndim == 3:
            lip_tensor = lip_tensor.permute(2, 0, 1)

        return lip_tensor

    def process(self, frames_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理帧序列，简化的处理逻辑"""
        # 初始化输出列表
        landmarks_list = []
        lip_images_list = []
        confidence_list = []

        # 找到第一个有效帧的索引
        last_valid_idx = None
        for i, frame in enumerate(frames_data):
            if frame['confidence'] > 0:
                last_valid_idx = i
                break

        # 如果没有任何有效帧，返回全零数据
        if last_valid_idx is None:
            zero_landmarks = torch.zeros((self.target_frames, 68, 2))
            zero_lip = torch.zeros((self.target_frames, 3, 96, 96))
            zero_conf = torch.zeros(self.target_frames)
            return zero_landmarks, zero_lip, zero_conf

        # 处理每一帧
        for i, frame in enumerate(frames_data[:self.target_frames]):
            if frame['confidence'] > 0:
                # 处理有效帧
                landmarks = self._normalize_landmarks(frame['landmarks'])
                lip_image = self._process_lip_image(frame['lip'])
                confidence = frame['confidence']
                last_valid_idx = i
            else:
                # 使用最近的有效帧
                landmarks = self._normalize_landmarks(frames_data[last_valid_idx]['landmarks'])
                lip_image = self._process_lip_image(frames_data[last_valid_idx]['lip'])
                confidence = 0.0  # 置信度设为0

            landmarks_list.append(torch.from_numpy(landmarks).float())
            lip_images_list.append(lip_image)
            confidence_list.append(confidence)

        # 如果帧数不足，使用最后一帧的数据填充
        while len(landmarks_list) < self.target_frames:
            landmarks_list.append(landmarks_list[-1])
            lip_images_list.append(lip_images_list[-1])
            confidence_list.append(confidence_list[-1])

        # 堆叠成批量张量
        landmarks_tensor = torch.stack(landmarks_list)  # [T, 68, 2]
        lip_images_tensor = torch.stack(lip_images_list)  # [T, 3, 96, 96]
        confidence_tensor = torch.tensor(confidence_list)  # [T]

        return landmarks_tensor, lip_images_tensor, confidence_tensor