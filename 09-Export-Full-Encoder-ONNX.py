# coding=utf-8
import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "qwen_asr_gguf" / "export"))

from export_config import ASR_MODEL_DIR, EXPORT_DIR
from qwen_asr import Qwen3ASRModel
from qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASREncoderFullOnnx

def export_full_encoder():
    model_path = str(ASR_MODEL_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_encoder.onnx")
    
    print(f"Loading official model for Full Encoder export...")
    asr_model = Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    audio_tower = asr_model.model.thinker.audio_tower
    
    full_model = Qwen3ASREncoderFullOnnx(audio_tower)
    full_model.eval()
    
    # 使用典型的长音频 dummy 输入
    dummy_mel = torch.randn(1, 128, 2850)
    # 模拟一个对应的全屏注意力掩码
    # 2850 帧前端输入对应 371 轴
    dummy_mask = torch.zeros(1, 1, 371, 371)
    
    print(f"Exporting FULL ENCODER to ONNX: {onnx_path}...")
    
    torch.onnx.export(
        full_model,
        (dummy_mel, dummy_mask),
        onnx_path,
        input_names=["input_features", "attention_mask"],
        output_names=["audio_embeds"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "time_mel"},
            "attention_mask": {0: "batch", 2: "time_q", 3: "time_k"},
            "audio_embeds": {0: "batch", 1: "time_out"},
        },
        opset_version=18,
        do_constant_folding=True
    )
    
    print(f"✅ Full Encoder ONNX export complete!")

if __name__ == "__main__":
    export_full_encoder()
