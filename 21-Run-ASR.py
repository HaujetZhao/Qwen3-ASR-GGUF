# coding=utf-8
import os
import sys
import re
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import QwenASREngine, itn, load_audio, ASREngineConfig, AlignerConfig
from qwen_asr_gguf.inference import exporters

def main():
    
    audio_path = "睡前消息.m4a"
    context = "这是1004期睡前消息，主持人叫督工，助理叫静静。"

    # 配置引擎
    config = ASREngineConfig(
        model_dir="model", 
        use_dml = True, 
        encoder_fn = "qwen3_asr_encoder.fp16.onnx" ,
        enable_aligner = True, 
        align_config = AlignerConfig(
            use_dml=True, 
            model_dir="model", 
            encoder_fn = "qwen3_aligner_encoder.fp16.onnx" 
        )
    )

    # 初始化引擎
    engine = QwenASREngine(config=config)
    
    # 执行转录
    res = engine.transcribe(
        audio_file=audio_path,
        context=context,
        language="Chinese",
        chunk_size=40.0, 
        memory_num=1,
        start_second=0,
        duration=0
    )
    
    
    # 导出文本（每行一句）
    txt_path = str(Path(audio_path).with_suffix('.txt'))
    exporters.export_to_txt(txt_path, res)

    # 导出 SRT（仅当有对齐时间戳时，才会有内容输出）
    srt_path = str(Path(audio_path).with_suffix('.srt'))
    exporters.export_to_srt(srt_path, res)

    # 导出 JSON（仅当有对齐时间戳时，才会有内容输出）
    json_path = str(Path(audio_path).with_suffix('.json'))
    exporters.export_to_json(json_path, res)

    # 对齐预览 (仅当有结果时)
    if res.alignment:
        print("\n" + "="*15 + " 对齐结果预览 (前10个) " + "="*15)
        for it in res.alignment.items[:10]:
            print(f"{it.text:<10} | {it.start_time:7.3f}s | {it.end_time:7.3f}s")
        print("="*52)
    
    # 优雅退出
    engine.shutdown()

if __name__ == "__main__":
    main()
