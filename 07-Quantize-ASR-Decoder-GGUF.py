import os
import subprocess
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.absolute()
QUANTIZE_EXE = PROJECT_ROOT / "qwen_asr_gguf" / "inference" / "bin" / "llama-quantize.exe"

QUANTIZE_TYPE = "q5_k"
MODEL_DIR = PROJECT_ROOT / "model"
INPUT_MODEL = MODEL_DIR / "qwen3_asr_llm.f16.gguf"
OUTPUT_MODEL = MODEL_DIR / f"qwen3_asr_llm.{QUANTIZE_TYPE}.gguf"

def main():
    print("---------------------------------------------------------")
    print("             执行 ASR Decoder 量化")
    print("---------------------------------------------------------")
    
    if not QUANTIZE_EXE.exists():
        print(f"❌ 找不到量化工具: {QUANTIZE_EXE}")
        return
        
    if not INPUT_MODEL.exists():
        print(f"❌ 找不到输入模型，请先运行 05 脚本生成 f16 模型: {INPUT_MODEL}")
        return
        
    print(f"🔹 输入模型: {INPUT_MODEL.name}")
    print(f"🔹 输出模型: {OUTPUT_MODEL.name}")
    print(f"🔹量化类型: {QUANTIZE_TYPE}")
    
    cmd = [
        str(QUANTIZE_EXE),
        str(INPUT_MODEL),
        str(OUTPUT_MODEL),
        QUANTIZE_TYPE
    ]
    
    print(f"\n🚀 正在启动 llama-quantize.exe...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ ASR Decoder {QUANTIZE_TYPE} 量化成功！")
        print(f"📁 产物保存在: {OUTPUT_MODEL}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 量化失败，错误码: {e.returncode}")
    except Exception as e:
        print(f"\n❌ 执行时发生未知错误: {e}")

if __name__ == '__main__':
    main()
