# coding=utf-8
import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "qwen_asr_gguf" / "export"))

from export_config import ASR_MODEL_DIR, EXPORT_DIR
from qwen_asr import Qwen3ASRModel
from qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASRBackendOnnx

def export_backend():
    model_path = str(ASR_MODEL_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_backend.onnx")
    
    print(f"Loading official model for Backend export...")
    asr_model = Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    audio_tower = asr_model.model.thinker.audio_tower
    
    backend_model = Qwen3ASRBackendOnnx(audio_tower)
    backend_model.eval()
    
    # Dummy 输入
    # 前端输出维度是 896，长度对应 2850 帧前端输入时是 371
    seq_len = 371
    dummy_hidden = torch.randn(1, seq_len, 896)
    # 模拟一个全 0 的 Mask (代表 Full Attention，验证时再传入分块 Mask)
    dummy_mask = torch.zeros(1, 1, seq_len, seq_len)
    
    print(f"Exporting Backend to ONNX: {onnx_path}...")
    
    torch.onnx.export(
        backend_model,
        (dummy_hidden, dummy_mask),
        onnx_path,
        input_names=["hidden_states", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "time"},
            "attention_mask": {0: "batch", 2: "time_q", 3: "time_k"},
            "last_hidden_state": {0: "batch", 1: "time"},
        },
        opset_version=18,
        do_constant_folding=True
    )
    
    print(f"✅ Backend ONNX export complete!")

if __name__ == "__main__":
    export_backend()
