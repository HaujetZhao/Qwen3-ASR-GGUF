# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3ASRFrontendFullOnnx(nn.Module):
    """
    Qwen3-ASR 完整前端 (Precision Wrapper)
    支持一次性投喂整段音频，内部自动实现‘切块-卷积-拼接’逻辑，
    并精确计算输出长度，确保输出结果与官方 bit-exact 且维度完全对齐。
    """
    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        self.pos_embed_table = audio_tower.positional_embedding.positional_embedding
        
    def _get_feat_extract_output_lengths(self, input_lengths):
        """参考官方计算公式计算卷积后的序列长度"""
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def forward(self, input_features: torch.Tensor):
        """
        Args:
            input_features: (Batch, Mel_Bins, Time) -> (1, 128, T)
        """
        b, mel, t = input_features.shape
        chunk_size = 100
        
        # 1. 计算官方预期的最终长度（用于最后的切片对齐）
        expected_len = self._get_feat_extract_output_lengths(t)
        
        # 2. 内部补齐到 100 的倍数以便矩阵化处理
        pad_len = (chunk_size - (t % chunk_size)) % chunk_size
        if pad_len > 0:
            x = F.pad(input_features, (0, pad_len))
        else:
            x = input_features
            
        num_chunks = x.shape[2] // chunk_size
        
        # 3. 维度重排进入分块处理模式
        x = x.transpose(1, 2).contiguous() # (1, T_pad, 128)
        x = x.view(num_chunks, chunk_size, mel).transpose(1, 2).unsqueeze(1) # (N, 1, 128, 100)
        
        # 4. 卷积下采样
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        
        # 5. 投影到 HiddenSize (N, 13, 896)
        b_c, c, f, t_c = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(num_chunks, t_c, c * f)
        x = self.conv_out(x)
        
        # 6. 添加位置编码 (每个块都固定加一次 13 维的 Positional Embedding)
        pos_embed = self.pos_embed_table[:t_c, :].unsqueeze(0)
        x = x + pos_embed
        
        # 7. 展平并精确切片裁剪
        # 还原后的序列长度是 N * 13，但我们需要裁剪到 expected_len
        x = x.view(1, -1, x.shape[-1])
        x = x[:, :expected_len, :]
        
        return x
