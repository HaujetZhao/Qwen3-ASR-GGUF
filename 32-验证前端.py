# coding=utf-8
import os
import numpy as np
import onnxruntime as ort
from export_config import EXPORT_DIR

def calculate_cosine_similarity(a, b):
    # 展平 + 归一化点积
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a_flat, b_flat) / (norm_a * norm_b)

def get_feat_extract_output_lengths(input_lengths):
    """
    完全复刻官方 Qwen3 前端逻辑，计算最终有效的输出帧数。
    用于从拼接好的 (N*13) 结果中切出有效部分。
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return int(output_lengths)

def main():
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_encoder_frontend.fp32.onnx")

    # 我们比对的目标是捕捉到的 encoder_backend_input.npy (它是前端的完整输出)
    input_mel_path = "capture_data/input_mel.npy" 
    baseline_path = "capture_data/encoder_backend_input.npy"

    if not os.path.exists(baseline_path):
        print("❌ 缺失捕获数据，请先运行 30-捕捉数据.py")
        return

    # 1. 加载数据
    input_mel = np.load(input_mel_path) # (1, 128, T)
    baseline = np.load(baseline_path)   # (T_out, 896)
    
    t_input = input_mel.shape[2]
    print(f"输入 Mel 形状: {input_mel.shape}, T={t_input}")
    
    # 2. 手动 Pad (复刻官方 behavior)
    chunk_size = 100
    pad_len = (chunk_size - (t_input % chunk_size)) % chunk_size
    if pad_len > 0:
        # np.pad: ((batch_L, batch_R), (freq_L, freq_R), (time_L, time_R))
        input_mel_padded = np.pad(input_mel, ((0,0), (0,0), (0, pad_len)), mode='constant', constant_values=0)
    else:
        input_mel_padded = input_mel
        
    num_chunks = input_mel_padded.shape[2] // chunk_size
    print(f"Pad 后形状: {input_mel_padded.shape}, 分块数: {num_chunks}")

    # 3. 启动 ONNX Runtime
    print(f"正在载入原子前端模型: {onnx_path} ...")
    sess = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    
    # 2.5 预热 (Warmup)
    print("正在预热 ONNX Runtime (Warmup)...")
    # 使用符合模型输入签名的随机数据
    dummy_in = np.random.randn(1, 128, 100).astype(np.float32)
    dummy_idx = np.array([0], dtype=np.int64)
    sess.run(None, {"chunk_mel": dummy_in})

    # 4. 循环推理
    outputs = []
    # print("开始循环推理...")
    import time
    t_start = time.perf_counter()
    
    for i in range(num_chunks):
        start = i * chunk_size
        chunk = input_mel_padded[:, :, start : start + chunk_size]
        
        # 运行单块推理
        # 输入名需与导出脚本一致: chunk_mel, chunk_idx
        out = sess.run(None, {"chunk_mel": chunk})[0] # (1, 13, 896)
        outputs.append(out)
        
    t_end = time.perf_counter()
    total_time = t_end - t_start
    print(f"推理完成! 总耗时: {total_time:.4f}s")
    if num_chunks > 0:
        print(f"平均单 Chunk (1s) 耗时: {total_time/num_chunks*1000:.2f}ms")
        
    # 5. 拼接与后处理
    # 拼接 -> (1, N*13, 896)
    full_out = np.concatenate(outputs, axis=1)
    
    # 根据官方逻辑计算有效长度并切片
    expected_len = get_feat_extract_output_lengths(t_input)
    # 官方输出 baseline 是 (T, 896)，没有 Batch 维度，且已经 squeeze
    final_out = full_out[0, :expected_len, :] # (T_expected, 896)

    print(f"ONNX 推理拼接并切片后形状: {final_out.shape}")
    print(f"官方基准形状: {baseline.shape}")
    
    # 6. 对比
    if final_out.shape == baseline.shape:
        sim = calculate_cosine_similarity(final_out, baseline)
        print("\n" + "="*40)
        print(f"精确对齐验证 (Precision Alignment Verification):")
        print(f"  - 形状匹配情况: ✅ 一致")
        print(f"  - 余弦相似度: {sim:.8f}")
        
        if sim > 0.9999:
             print("\n✨ Perfect Match! Python Loop Logic Verified.")
        else:
             print("\n⚠️ 相似度不足！与基准存在偏差。")
    else:
        print("\n" + "="*40)
        print(f"❌ ERROR: Shape Mismatch! {final_out.shape} vs {baseline.shape}")
    
    print("="*40)

if __name__ == "__main__":
    main()
