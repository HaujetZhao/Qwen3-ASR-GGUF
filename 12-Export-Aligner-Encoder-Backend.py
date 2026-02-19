# coding=utf-8
import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "qwen_asr_gguf" / "export"))

from export_config import ALIGNER_MODEL_DIR, EXPORT_DIR
from qwen_asr import Qwen3ASRModel
from qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASRBackendOnnx

def export_aligner_backend():
    model_path = str(ALIGNER_MODEL_DIR) # 使用 ALIGNER 路径
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # 目标文件名: qwen3_aligner_encoder_backend.fp32.onnx
    backend_path = os.path.join(EXPORT_DIR, "qwen3_aligner_encoder_backend.fp32.onnx")
    
    print(f"Loading ALIGNER model from {model_path}...")
    asr_model = Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    audio_tower = asr_model.model.thinker.audio_tower
    
    print("\n" + "="*50)
    print("Exporting [Aligner Backend] (Long-Sequence Transformer)...")
    backend_model = Qwen3ASRBackendOnnx(audio_tower)
    backend_model.eval()
    
    # 动态获取 hidden_size (Aligner 通常是 1024, ASR 是 896)
    # 通过查看 conv_out 层的输出维度来确定
    hidden_size = audio_tower.conv_out.out_features
    print(f"Detected Hidden Size: {hidden_size}")
    
    # 输入: 拼接后的特征序列
    # 维度: [Batch=1, Time=Dynamic, Dim=hidden_size]
    seq_len = 64
    dummy_hidden = torch.randn(1, seq_len, hidden_size) 
    
    # 关键修正：Mask 必须是 4D (batch, 1, time, time) 且为 Additive Mask (0.0)
    dummy_mask = torch.zeros(1, 1, seq_len, seq_len)
    
    torch.onnx.export(
        backend_model,
        (dummy_hidden, dummy_mask),
        backend_path,
        input_names=["hidden_states", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "time"},
            # 兼容 4D Mask 的动态轴
            "attention_mask": {0: "batch", 2: "time_q", 3: "time_k"}, 
            "last_hidden_state": {0: "batch", 1: "time"},
        },
        opset_version=18,
        do_constant_folding=True, 
        dynamo=True
    )
    print(f"✅ Aligner Backend exported to: {backend_path}")

if __name__ == "__main__":
    export_aligner_backend()
