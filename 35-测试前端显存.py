import os
import time
import numpy as np
import onnxruntime as ort
import pynvml
from export_config import EXPORT_DIR

# ============================================================================
# 配置与初始化
# ============================================================================
# 前端现在是原子模型
# 前端现在是原子模型
MODEL_PATH = os.path.join(EXPORT_DIR, "qwen3_asr_encoder_frontend.fp32.onnx")

try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def get_gpu_used():
    if not HAS_NVML: return 0
    info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
    return info.used / 1024 / 1024  # MB

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到前端模型 {MODEL_PATH}")
        return

    print("\n" + "="*50)
    print(" 脚本 35: 原子前端 (Atomic Frontend) FP32 循环推理显存测试")
    print("="*50)

    # 1. 载入模型
    vram_start = get_gpu_used()
    sess_opts = ort.SessionOptions()
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    
    print(f"正在载入原子前端模型...")
    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    vram_after_load = get_gpu_used()
    print(f"载入后显存: {vram_after_load:.2f} MB (净开销: {vram_after_load - vram_start:.2f} MB)")

    # 2. 准备 40s 音频输入 (4000 帧 Mel)
    # 2. 准备 5.2s 音频输入 (520 帧 Mel)
    # T = 520
    t_mel = 520
    chunk_size = 100
    
    # 模拟真实推理中的 Padding 逻辑
    if t_mel % chunk_size != 0:
        pad_len = chunk_size - (t_mel % chunk_size)
        total_len = t_mel + pad_len
        print(f"输入长度 {t_mel} 不是 {chunk_size} 的倍数，Padding {pad_len} 帧 (Total: {total_len})...")
    else:
        pad_len = 0
        total_len = t_mel
        
    num_chunks = total_len // chunk_size
    
    # 模拟输入 (Batch=1)
    full_mel = np.zeros((1, 128, total_len), dtype=np.float32)
    
    print(f"\n正在模拟 {t_mel} 帧音频前端推理 (Loop Mode, {num_chunks} Chunks)...")
    
    # 热身 (送两个 chunk)
    for k in range(2):
        dummy_chunk = full_mel[:, :, :100]
        session.run(None, {"chunk_mel": dummy_chunk})
        
    vram_post_warmup = get_gpu_used()
    
    # 正式循环推理
    t_start = time.perf_counter()
    
    outputs = []
    # 模拟 Python 端的循环调度
    for i in range(num_chunks):
        # 1. 切片
        start = i * chunk_size
        chunk = full_mel[:, :, start : start + chunk_size]
        
        # 2. 推理 (显存只负责这 1 个 Chunk)
        out = session.run(None, {
            "chunk_mel": chunk
        })[0]
        
        # 4. 收集 (放到 List 中，List 会占用 CPU/系统内存，不占显存)
        outputs.append(out)
        
    t_end = time.perf_counter()
    
    # 拼装结果
    final_out = np.concatenate(outputs, axis=1)
    
    vram_final = get_gpu_used()
    increment = vram_final - vram_after_load
    
    print(f"推理总耗时: {t_end - t_start:.4f}s")
    print(f"平均单块耗时: {(t_end - t_start)/num_chunks*1000:.2f}ms")
    print(f"推理后总显存: {vram_final:.2f} MB")
    print(f"显存增量: {increment:.2f} MB (预期应在 30-50MB 级)")
    print(f"输出形状: {final_out.shape}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
