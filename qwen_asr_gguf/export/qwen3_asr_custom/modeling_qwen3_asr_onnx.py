# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3ASRFrontendOnnx(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        
    def forward(self, input_features: torch.Tensor):
        """
        这里我们实现与官方一致的单块卷积逻辑
        Args:
            input_features: (Batch, Mel_Bins, Time) -> 推荐固定 Time 为 100 (1秒)
        Returns:
            projected_embeds: (Batch, T_downsampled, Hidden_Size) -> 若 Time=100, T_downsampled=13
        """
        # 1. 强制 4D 输入 (B, 1, 128, T)
        x = input_features.unsqueeze(1)
        
        # 2. 卷积下采样
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        
        # 3. 维度重排 (B, C, F, T) -> (B, T, C, F) -> (B, T, C*F)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, c * f)
        
        # 4. 线性投影
        x = self.conv_out(x)
        
        return x
