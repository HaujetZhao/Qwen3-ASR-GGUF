# coding=utf-8
import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "qwen_asr_gguf" / "export"))

from export_config import MODEL_DIR, EXPORT_DIR
from qwen_asr import Qwen3ASRModel
from qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASRFrontendFullOnnx

def export_frontend():
    model_path = str(MODEL_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_frontend.onnx")
    
    # 强制 float32
    asr_model = Qwen3ASRModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    audio_tower = asr_model.model.thinker.audio_tower
    
    frontend_model = Qwen3ASRFrontendFullOnnx(audio_tower)
    frontend_model.eval()
    
    # Dummy 输入：使用 2850 帧进行导出，并设置动态轴
    dummy_input = torch.randn(1, 128, 2850)
    
    print(f"Exporting PRECISION Frontend to ONNX...")
    
    torch.onnx.export(
        frontend_model,
        (dummy_input,),
        onnx_path,
        input_names=["input_features"],
        output_names=["frontend_output"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "time"},
            "frontend_output": {0: "batch", 1: "time"},
        },
        opset_version=18,
        do_constant_folding=True
    )
    
    print(f"✅ Precision Frontend ONNX export complete!")

if __name__ == "__main__":
    export_frontend()
