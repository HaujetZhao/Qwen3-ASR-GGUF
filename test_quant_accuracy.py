import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference.encoder import QwenAudioEncoder
from qwen_asr_gguf.inference.audio import load_audio


# ===== 对比配置 =====
# 只需修改精度名即可，支持: fp32, fp16, int8, int4
REF_PRECISION = "fp16"
TGT_PRECISION = "int4"
# ===================


def calculate_cosine_similarity(v1, v2):
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    return np.dot(v1_flat, v2_flat) / (np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat))


def _make_name(prefix: str, precision: str) -> str:
    return f"qwen3_asr_encoder_{prefix}.{precision}.onnx"

def main():
    ref_label = REF_PRECISION.upper()
    tgt_label = TGT_PRECISION.upper()
    ref_frontend = _make_name("frontend", REF_PRECISION)
    ref_backend  = _make_name("backend", REF_PRECISION)
    tgt_frontend = _make_name("frontend", TGT_PRECISION)
    tgt_backend  = _make_name("backend", TGT_PRECISION)

    audio_file = "test.mp3"
    if not os.path.exists(audio_file):
        print(f"Error: 找不到 {audio_file}")
        sys.exit(1)

    model_dir = os.path.join(Path(__file__).parent.absolute(), "model")

    print("[1/4] 载入音频文件...")
    audio = load_audio(audio_file, start_second=0, duration=None)
    print(f"  音频长度: {len(audio)/16000:.2f} 秒")

    # ----- 参考模型 -----
    print(f"\n[2/4] 载入 {ref_label} Encoder 并推理...")
    ref_encoder = QwenAudioEncoder(
        frontend_path=os.path.join(model_dir, ref_frontend),
        backend_path=os.path.join(model_dir, ref_backend),
        onnx_provider='cpu',
        verbose=False
    )
    ref_embd, ref_time = ref_encoder.encode(audio)
    print(f"  {ref_label} 推理完成，耗时: {ref_time:.2f}s, 输出形状: {ref_embd.shape}")

    # ----- 目标模型 -----
    print(f"\n[3/4] 载入 {tgt_label} Encoder 并推理...")
    tgt_encoder = QwenAudioEncoder(
        frontend_path=os.path.join(model_dir, tgt_frontend),
        backend_path=os.path.join(model_dir, tgt_backend),
        onnx_provider='cpu',
        verbose=False
    )
    tgt_embd, tgt_time = tgt_encoder.encode(audio)
    print(f"  {tgt_label} 推理完成，耗时: {tgt_time:.2f}s, 输出形状: {tgt_embd.shape}")

    del ref_encoder
    del tgt_encoder

    print(f"\n[4/4] 计算 {ref_label} vs {tgt_label} 相似度...")
    if ref_embd.shape != tgt_embd.shape:
        print("  ⚠️ 形状不完全一致，将对齐最小长度")
        min_len = min(ref_embd.shape[0], tgt_embd.shape[0])
        ref_embd = ref_embd[:min_len]
        tgt_embd = tgt_embd[:min_len]

    sim = calculate_cosine_similarity(ref_embd, tgt_embd)
    print(f"  🎯 余弦相似度 (Cosine Similarity): {sim:.5f}")

    mae = np.mean(np.abs(ref_embd - tgt_embd))
    print(f"  🎯 平均绝对误差 (MAE): {mae:.5f}")

    speed_ratio = ref_time / tgt_time if tgt_time > 0 else float('inf')
    print(f"  🎯 速度比 ({ref_label}/{tgt_label}): {speed_ratio:.2f}x")


if __name__ == '__main__':
    main()
