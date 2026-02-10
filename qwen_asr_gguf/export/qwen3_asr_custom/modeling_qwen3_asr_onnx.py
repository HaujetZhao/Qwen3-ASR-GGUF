# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3ASRFrontendFullOnnx(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        
        # 关键：获取位置编码矩阵
        # 官方位置编码是 SinusoidsPositionEmbedding，保存在 positional_embedding.positional_embedding 中
        self.pos_embed_table = audio_tower.positional_embedding.positional_embedding
        
    def forward(self, input_features: torch.Tensor):
        """
        Args:
            input_features: (Batch, Mel_Bins, Time) -> (1, 128, T)
        """
        b, mel, t = input_features.shape
        chunk_size = 100
        
        # 1. 补齐到 100 的倍数 (对齐官方逻辑)
        pad_len = (chunk_size - (t % chunk_size)) % chunk_size
        if pad_len > 0:
            x = F.pad(input_features, (0, pad_len))
        else:
            x = input_features
            
        num_chunks = x.shape[2] // chunk_size
        
        # 2. 切块：(1, 128, N*100) -> (N, 1, 128, 100)
        # 先转置并确保内存连续，避免 Reshape 出错
        x = x.transpose(1, 2).contiguous() # (1, T_pad, 128)
        x = x.view(num_chunks, chunk_size, mel).transpose(1, 2).unsqueeze(1) # (N, 1, 128, 100)
        
        # 3. 卷积
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        
        # 4. 投影：(N, 112, 16, 13) -> (N, 13, 896)
        b_c, c, f, t_c = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(num_chunks, t_c, c * f)
        x = self.conv_out(x)
        
        # 5. 【核心修正】添加位置编码
        # 官方逻辑：每一个 chunk (长 13) 都加一遍前 13 个位置编码向量
        pos_embed = self.pos_embed_table[:t_c, :].unsqueeze(0) # (1, 13, 896)
        x = x + pos_embed
        
        # 6. 展平回序列：(1, N*13, 896)
        x = x.view(1, -1, x.shape[-1])
        
        return x
