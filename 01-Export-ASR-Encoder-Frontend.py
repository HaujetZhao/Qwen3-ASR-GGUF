# coding=utf-8
import os
import sys
import torch
from pathlib import Path

# 添加相关路径
sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "qwen_asr_gguf" / "export"))

from export_config import ASR_MODEL_DIR, EXPORT_DIR
from qwen_asr import Qwen3ASRModel
from qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASRFrontendAtomicOnnx

def export_frontend():
    model_path = str(ASR_MODEL_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    frontend_path = os.path.join(EXPORT_DIR, "qwen3_asr_encoder_frontend.fp32.onnx")
    
    print(f"Loading PyTorch model from {model_path}...")
    # 强制 float32
    asr_model = Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    audio_tower = asr_model.model.thinker.audio_tower
    
    # ========================================================================
    # 导出原子前端 (Atomic Frontend)
    # ========================================================================
    print("\n" + "="*50)
    print("Exporting [Frontend] (Atomic Chunk-based, Static Shape)...")
    frontend_model = Qwen3ASRFrontendAtomicOnnx(audio_tower)
    frontend_model.eval()
    
    # 输入: 单个 Chunk (1s, 100帧)
    # 维度: [Batch=1, Freq=128, Time=100]
    dummy_input = torch.randn(1, 128, 100)
    
    torch.onnx.export(
        frontend_model,
        (dummy_input,),
        frontend_path,
        input_names=["chunk_mel"],
        output_names=["chunk_out"],
        # 纯静态模型
        opset_version=18,
        do_constant_folding=False, 
        dynamo=True
    )
    print(f"✅ Frontend exported to: {frontend_path}")

if __name__ == "__main__":
    export_frontend()
