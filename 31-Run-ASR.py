# coding=utf-8
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import QwenASREngine, itn, load_audio, ASREngineConfig

def main():
    # 路径配置
    model_dir = "model"
    audio_path = "睡前消息.m4a"
    
    # 1. 初始化引擎 (使用标准化 Config)
    config = ASREngineConfig(
        model_dir=model_dir, 
        use_dml=False,
        enable_aligner=True # 开启对齐功能
    )
    engine = QwenASREngine(config=config)
    
    # 2. 加载音频
    print(f"\n加载音频: {audio_path}\n")
    audio = load_audio(audio_path, start_second=0, duration=120)
    
    # 3. 执行转录
    res = engine.transcribe(
        audio=audio,
        context="这是1004期睡前消息，主持人叫督工，助理叫静静。",
        language="Chinese",
        chunk_size_sec=40.0
    )
    
    # 4. ITN 后处理与输出
    print("\n" + "="*20 + " ITN 处理后 " + "="*20)
    print(itn(res.text))
    print("="*52)

    # 5. 对齐预览 (如果有)
    if res.alignment:
        print("\n" + "="*15 + " 对齐结果预览 (前10个) " + "="*15)
        for it in res.alignment[:10]:
            print(f"{it.text:<10} | {it.start_time:7.3f}s | {it.end_time:7.3f}s")
        print("="*52)
    
    # 6. 优雅退出
    engine.shutdown()

if __name__ == "__main__":
    main()
