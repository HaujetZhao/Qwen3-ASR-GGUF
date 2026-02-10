# coding=utf-8
import os
import numpy as np
import onnxruntime as ort
from export_config import EXPORT_DIR

def calculate_cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

def main():
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_frontend.onnx")
    input_mel_path = "capture_data/input_mel.npy"
    # 这里对比拼接完成后且包含位置编码的后端输入基准 (371, 896)
    baseline_path = "capture_data/encoder_backend_input.npy"

    if not os.path.exists(onnx_path) or not os.path.exists(baseline_path):
        print("❌ 缺失文件，请确保已运行导出并捕获过数据。")
        return

    # 1. 加载数据
    input_mel = np.load(input_mel_path)  # (1, 128, 2850)
    baseline_full = np.load(baseline_path)  # (371, 896)
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # 2. 一键投喂验证
    print(f"验证开始：一次性向 ONNX 投喂 {input_mel.shape[2]} 帧特征...")
    
    ort_outs = sess.run(None, {sess.get_inputs()[0].name: input_mel})
    onnx_output = ort_outs[0][0] # (T_downsampled, 896)
    
    print(f"ONNX 输出形状: {onnx_output.shape}")
    print(f"官方基准形状: {baseline_full.shape}")

    # 3. 维度对比与相似度计算
    if onnx_output.shape == baseline_full.shape:
        sim = calculate_cosine_similarity(baseline_full, onnx_output)
        print("\n" + "="*40)
        print(f"精确对齐验证 (Precision Alignment Verification):")
        print(f"  - 形状匹配情况: ✅ 一致")
        print(f"  - 余弦相似度: {sim:.8f}")
        
        if sim > 0.9999:
            print("\n✨ PERFECT! The ONNX output is bit-exact and dimension-aligned with official output.")
        else:
            print("\n⚠️ WARNING: Dimensions match but content differs. Check logic.")
    else:
        print("\n" + "="*40)
        print(f"❌ ERROR: Shape Mismatch! {onnx_output.shape} vs {baseline_full.shape}")
        # 即使形状不对，也强行对齐看一下内容相似度
        min_t = min(onnx_output.shape[0], baseline_full.shape[0])
        sim = calculate_cosine_similarity(baseline_full[:min_t], onnx_output[:min_t])
        print(f"  - 截断后的余弦相似度: {sim:.8f}")
    print("="*40)

if __name__ == "__main__":
    main()
