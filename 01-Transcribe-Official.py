# coding=utf-8
import os
import torch
from export_config import ASR_MODEL_DIR
from qwen_asr import Qwen3ASRModel

def main():
    # 1. 确保模型路径为字符串
    model_path = str(ASR_MODEL_DIR)
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}")
        return

    # 2. 加载官方推理模型 (Transformers 后端)
    # 根据技术报告，Qwen3-ASR 使用 bfloat16 性能最佳
    asr_model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",  # 自动选择 GPU/CPU
    )

    # 3. 指定音频路径
    audio_path = "test.mp3"
    print(f"Transcribing audio: {audio_path}")

    # 4. 执行推理
    # language=None 表示自动语种识别
    results = asr_model.transcribe(
        audio=audio_path,
        language=None,
        return_time_stamps=False,
    )

    # 5. 输出结果
    print("\n" + "="*30)
    print("Transcription Result:")
    print("="*30)
    for i, res in enumerate(results):
        print(f"Language: {res.language}")
        print(f"Text: {res.text}")
    print("="*30)

if __name__ == "__main__":
    main()
