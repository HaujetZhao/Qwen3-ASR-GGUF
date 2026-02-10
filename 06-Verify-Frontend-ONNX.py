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
    # 这里我们对比 02 直接捕获的卷积层原生输出 (Chunked)
    baseline_path = "capture_data/encoder_frontend_output.npy"

    if not os.path.exists(onnx_path) or not os.path.exists(baseline_path):
        print("❌ 缺失文件，请确保已运行导出并捕获过数据。")
        return

    # 1. 加载数据
    input_mel = np.load(input_mel_path)  # (1, 128, 2850)
    baseline_chunks = np.load(baseline_path)  # (29, 13, 896)
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # 2. 模拟官方 Chunk 逻辑进行验证
    # 官方将音频由于 n_window=50, 所以切成 100 帧一格
    chunk_size = 100
    n_mel_frames = input_mel.shape[2]
    
    print(f"验证开始：音频总帧数 {n_mel_frames}, 预计 Chunk 数 {baseline_chunks.shape[0]}")
    
    similarities = []
    
    for i in range(baseline_chunks.shape[0]):
        start = i * chunk_size
        end = min(start + chunk_size, n_mel_frames)
        
        # 获取当前块输入
        mel_chunk = input_mel[:, :, start:end]
        
        # 如果长度不足 100，需要像官方那样进行 Padding (这里我们观察 02 发现它捕获时已经对应了逻辑)
        # 官方的 pad_sequence 默认补 0
        if mel_chunk.shape[2] < chunk_size:
            pad_width = chunk_size - mel_chunk.shape[2]
            mel_chunk = np.pad(mel_chunk, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
            
        # ONNX 推理
        ort_outs = sess.run(None, {sess.get_inputs()[0].name: mel_chunk})
        onnx_chunk_out = ort_outs[0][0] # (13, 896)
        
        # 获取对应的基准块
        target_chunk = baseline_chunks[i] # (13, 896)
        
        sim = calculate_cosine_similarity(target_chunk, onnx_chunk_out)
        similarities.append(sim)

    avg_sim = np.mean(similarities)
    print("\n" + "="*40)
    print(f"分块验证完毕 (Chunk-wise Verification):")
    print(f"  - 平均余弦相似度: {avg_sim:.8f}")
    print(f"  - 最小相似度: {min(similarities):.8f}")
    
    if avg_sim > 0.9999:
        print("\n✅ PERFECT: ONNX implementation is bit-exact with Torch Chunks!")
    elif avg_sim > 0.99:
        print("\n✅ SUCCESS: High similarity achieved.")
    else:
        print("\n❌ FAILURE: Mismatch detected. Please check padding or windowing.")
    print("="*40)

if __name__ == "__main__":
    main()
