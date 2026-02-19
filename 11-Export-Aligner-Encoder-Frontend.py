# coding=utf-8
import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "qwen_asr_gguf" / "export"))

from export_config import ALIGNER_MODEL_DIR, EXPORT_DIR
from qwen_asr import Qwen3ASRModel
from qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASRFrontendAtomicOnnx

def export_aligner_frontend():
    model_path = str(ALIGNER_MODEL_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # 目标文件名: qwen3_aligner_encoder_frontend.fp32.onnx
    frontend_path = os.path.join(EXPORT_DIR, "qwen3_aligner_encoder_frontend.fp32.onnx")
    
    print(f"Loading ALIGNER model from {model_path}...")
    asr_model = Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    audio_tower = asr_model.model.thinker.audio_tower
    
    print("\n" + "="*50)
    print("Exporting [Aligner Frontend] (Atomic Chunk-based, Static Shape)...")
    frontend_model = Qwen3ASRFrontendAtomicOnnx(audio_tower)
    frontend_model.eval()
    
    # 输入: 单个 Chunk (1s, 100帧)
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
    print(f"✅ Aligner Frontend exported to: {frontend_path}")

if __name__ == "__main__":
    export_aligner_frontend()
