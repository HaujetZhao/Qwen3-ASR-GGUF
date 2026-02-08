import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
from export_config import EXPORT_DIR

# Configuration
PROJECT_ROOT = Path(__file__).parent.absolute()
EXPORT_DIR = EXPORT_DIR / "onnx"

FP32_MODELS = [
    str(EXPORT_DIR / "qwen3_asr_encoder_frontend.fp32.onnx"),
    str(EXPORT_DIR / "qwen3_asr_encoder_backend.fp32.onnx")
]

def convert_to_fp16(input_path):
    output_path = input_path.replace(".fp32.onnx", ".fp16.onnx")
    print(f"\n[FP16] Converting {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        model = onnx.load(input_path)
        # Use ORT Transformers conversion for better DML compatibility
        # Block ops that are sensitive to precision changes or shape calculation
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=False,
            min_positive_val=1e-7,
            max_finite_val=65504,
            op_block_list=['LayerNormalization']
        )
        onnx.save(model_fp16, output_path)
        print(f"   ✅ [Success] Saved FP16 model.")
    except Exception as e:
        print(f"   ❌ [Failed] FP16 conversion error: {e}")

def convert_to_int8(input_path):
    output_path = input_path.replace(".fp32.onnx", ".int8.onnx")
    print(f"\n[INT8] Quantizing {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        quantize_dynamic(
            input_path,
            output_path,
            op_types_to_quantize=["MatMul"], # Primary target for weight compression
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8
        )
        print(f"   ✅ [Success] Saved INT8 model.")
    except Exception as e:
        print(f"   ❌ [Failed] INT8 quantization error: {e}")

def main():
    print("--- 正在开始针对 Qwen3-ASR Encoder 的批量量化/转换 ---")
    
    if not EXPORT_DIR.exists():
        print(f"错误: 目录 {EXPORT_DIR} 不存在。")
        return

    for model_path in FP32_MODELS:
        if not os.path.exists(model_path):
            print(f"\n[跳过] 找不到模型文件: {model_path}")
            continue
            
        # 1. 转换为 FP16
        convert_to_fp16(model_path)
        
        # 2. 转换为 INT8
        convert_to_int8(model_path)

    print("\n--- 所有转换工作已完成 ---")

if __name__ == "__main__":
    main()
