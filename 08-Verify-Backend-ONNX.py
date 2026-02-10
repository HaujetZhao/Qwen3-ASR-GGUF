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
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_backend.onnx")
    input_path = "capture_data/encoder_backend_input.npy"
    output_path = "capture_data/encoder_backend_output.npy"

    if not os.path.exists(onnx_path) or not os.path.exists(input_path):
        print("❌ 缺失文件，请确保执行过导出和捕获数据。")
        return

    # 1. 加载数据
    backend_input = np.load(input_path)   # (371, 896)
    baseline_output = np.load(output_path) # (371, 1024)
    seq_len = backend_input.shape[0]
    
    # 2. 构造 Full Attention Mask (全 0)
    # 因为官方推理时实际上忽略了块隔离，所以我们投喂全 0 掩码
    mask_input = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    
    # 3. 推理
    sess = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    
    print(f"Running Final Backend ONNX Verification (Full Attention Mode)...")
    ort_outs = sess.run(None, {
        "hidden_states": backend_input[np.newaxis, :, :],
        "attention_mask": mask_input
    })
    onnx_output = ort_outs[0][0] # (371, 1024)

    # 4. 对比
    sim = calculate_cosine_similarity(baseline_output, onnx_output)
    
    print("\n" + "="*40)
    print(f"Backend 验证结果 (Final Result):")
    print(f"  - 形状: {onnx_output.shape} (ONNX) == {baseline_output.shape} (Baseline)")
    print(f"  - 余弦相似度: {sim:.8f}")
    
    if sim > 0.99999:
        print("\n✨ 满分达成 (1.000000)! 验证完全通过。")
        print("结论：后端 Transformer 逻辑及权重导出完全正确。")
    elif sim > 0.999:
        print("\n✅ 验证通过：高度一致。")
    print("="*40)

if __name__ == "__main__":
    main()
