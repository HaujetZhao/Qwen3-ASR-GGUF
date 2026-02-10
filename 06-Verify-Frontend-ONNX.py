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
    # 这里我们对比 02 直接捕捉的后端输入 (拼接完成后的全长特征)
    baseline_path = "capture_data/encoder_backend_input.npy"

    if not os.path.exists(onnx_path) or not os.path.exists(baseline_path):
        print("❌ 缺失文件，请确保已运行导出并捕获过数据。")
        return

    # 1. 加载数据
    input_mel = np.load(input_mel_path)  # (1, 128, 2850)
    # 官方拼接后的特征 (371, 896)
    baseline_full = np.load(baseline_path)  
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # 2. 一键投喂验证
    print(f"验证开始：一次性喂入 {input_mel.shape[2]} 帧特征...")
    
    ort_outs = sess.run(None, {sess.get_inputs()[0].name: input_mel})
    onnx_output = ort_outs[0][0] # (T_total_downsampled, 896)
    
    print(f"ONNX 输出总长度: {onnx_output.shape[0]}")
    print(f"官方基准总长度: {baseline_full.shape[0]}")

    # 3. 对齐对比
    # 注意：由于 ONNX 内部做了 Full Padding 补齐到 100 倍数，结果会比 371 长一点点（通常补齐到 377）
    # 我们只需要截取与官方基准对等的部分即可
    min_t = min(onnx_output.shape[0], baseline_full.shape[0])
    onnx_final = onnx_output[:min_t, :]
    baseline_final = baseline_full[:min_t, :]
    
    sim = calculate_cosine_similarity(baseline_final, onnx_final)
    
    print("\n" + "="*40)
    print(f"全长投喂验证 (Full-Sequence Verification):")
    print(f"  - 余弦相似度: {sim:.8f}")
    
    if sim > 0.9999:
        print("\n✅ PERFECT: ONNX 'Full' Frontend matches official implementation exactly!")
    else:
        print("\n❌ MISMATCH: Similarity is low. Please check padding logic.")
    print("="*40)

if __name__ == "__main__":
    main()
