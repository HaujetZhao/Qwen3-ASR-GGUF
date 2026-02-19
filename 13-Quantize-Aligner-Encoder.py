# coding=utf-8
import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
from export_config import EXPORT_DIR

def convert_to_fp16(input_path):
    output_path = input_path.replace(".fp32.onnx", ".fp16.onnx")
    print(f"\n[FP16] Converting {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        model = onnx.load(input_path)
        # 使用 ORT Transformers 转换以获得更好的 DML 兼容性
        # 屏蔽对精度敏感或涉及形状计算的算子
        # 对于 Backend Transformer，保留 LayerNorm 和 Softmax 为 FP32 极其重要
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
            op_types_to_quantize=["MatMul", "Attention", "Conv"], # 权重量化的核心目标
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8
        )
        print(f"   ✅ [Success] Saved INT8 model.")
    except Exception as e:
        print(f"   ❌ [Failed] INT8 quantization error: {e}")

def main():
    print("--- 正在开始针对 Qwen3-Aligner 的批量量化/转换 ---")
    
    export_path = Path(EXPORT_DIR)
    if not export_path.exists():
        print(f"错误: 目录 {export_path} 不存在。")
        return

    # 定义目标文件列表
    targets = [
        "qwen3_aligner_encoder_frontend.fp32.onnx",
        "qwen3_aligner_encoder_backend.fp32.onnx"
    ]
    
    for target in targets:
        model_path = str(export_path / target)
        
        if not os.path.exists(model_path):
            print(f"\n❌ 跳过: 找不到基准 FP32 模型文件: {target}")
            continue
            
        print(f"\n>>> 处理模型: {target}")
        
        # 1. 转换为 FP16 (适用于 GPU/DirectML)
        convert_to_fp16(model_path)
        
        # 2. 动态量化为 INT8 (适用于 CPU)
        convert_to_int8(model_path)

    print("\n--- 所有转换工作已完成 ---")

if __name__ == "__main__":
    main()
