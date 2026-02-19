# coding=utf-8
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import QwenForcedAligner, load_audio, AlignerConfig

def main():
    # 路径配置
    model_dir = "model"
    audio_path = "test.mp3"
    text_path = "test.txt"
    
    if not os.path.exists(text_path):
        print(f"❌ 找不到文本文件: {text_path}")
        return
        
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 1. 初始化对齐器 (使用标准化 Config)
    # 默认会自动加载 Split Encoder (frontend/backend)
    config = AlignerConfig(
        model_dir="model", 
        use_dml=True
    )
    aligner = QwenForcedAligner(config=config)
    
    # 2. 加载音频 (基于 pydub)
    print(f"加载音频: {audio_path}")
    audio = load_audio(audio_path)
    
    # 3. 执行对齐 (标准化结果)
    results = aligner.align(audio, text)
    
    # 4. 输出对齐预览
    print("\n" + "="*50)
    print(f"{'Text':<20} | {'Start':<8} | {'End':<8}")
    print("-" * 50)
    for it in results[:20]: # 预览前 20 个
        print(f"{it.text:<20} | {it.start_time:7.3f}s | {it.end_time:7.3f}s")
    print(f"...\n总计对齐词数: {len(results)}")
    print("="*50)

if __name__ == "__main__":
    main()
