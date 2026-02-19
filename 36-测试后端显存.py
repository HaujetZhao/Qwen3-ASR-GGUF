import os
import time
import numpy as np
import onnxruntime as ort
import pynvml
from export_config import EXPORT_DIR

# ============================================================================
# 配置与初始化
# ============================================================================
MODEL_PATH = os.path.join(EXPORT_DIR, "qwen3_asr_encoder_backend.fp32.onnx")

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
        print(f"❌ 错误: 找不到后端模型 {MODEL_PATH}")
        return

    print("\n" + "="*50)
    print(" 脚本 36: 后端 (Backend) FP32 显存压力测试")
    print("="*50)

    # 1. 载入模型
    vram_start = get_gpu_used()
    sess_opts = ort.SessionOptions()
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    
    print(f"正在载入后端模型...")
    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    vram_after_load = get_gpu_used()
    print(f"载入后显存: {vram_after_load:.2f} MB (净开销: {vram_after_load - vram_start:.2f} MB)")

    # 2. 准备 40s 音频对应的后端输入
    # 40s -> T_out = 520
    t_out = 520
    dummy_hidden = np.zeros((1, t_out, 896), dtype=np.float32)
    # Full Attention Mask
    dummy_mask = np.zeros((1, 1, t_out, t_out), dtype=np.float32)
    
    print(f"\n正在模拟 {t_out} 步序列的后端 Transformer 推理...")
    
    # 准备输入字典
    input_feed = {
        "hidden_states": dummy_hidden,
        "attention_mask": dummy_mask
    }
    
    # 热身
    session.run(None, input_feed)
    vram_post_warmup = get_gpu_used()
    
    # 正式推理并统计
    t_start = time.perf_counter()
    session.run(None, input_feed)
    t_end = time.perf_counter()
    
    vram_final = get_gpu_used()
    increment = vram_final - vram_after_load
    
    print(f"推理耗时: {t_end - t_start:.4f}s")
    print(f"推理后总显存: {vram_final:.2f} MB")
    print(f"本次后端推理阶段显存增量: {increment:.2f} MB")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
