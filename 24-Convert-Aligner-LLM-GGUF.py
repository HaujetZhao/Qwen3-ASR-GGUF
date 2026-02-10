# coding=utf-8
import sys
import os
import json
from pathlib import Path

# 1. 路径设置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT_LIB_DIR = os.path.join(PROJECT_ROOT, "qwen_asr_gguf", 'export')
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "aligner_decoder_hf")

# 确保可以导入转换库
if CONVERT_LIB_DIR not in sys.path:
    sys.path.insert(0, CONVERT_LIB_DIR)

try:
    import convert_hf_to_gguf
    from convert_hf_to_gguf import ModelBase, TextModel
except ImportError as e:
    print(f"❌ 导入 convert_hf_to_gguf 失败: {e}")
    sys.exit(1)

# 2. 定义 Monkey Patches (猴子补丁)
# 目的：强制转换器读取本地 config.json 并返回正确的词表/架构标识
def patched_load_hparams(dir_model: Path, is_mistral_format: bool):
    """
    直接从磁加载 config.json，绕过 AutoConfig 的加载问题。
    """
    print(f"💉 [补丁] 拦截 load_hparams。正在从 {dir_model / 'config.json'} 加载...")
    
    with open(dir_model / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 针对对齐器模型的配置做归一化
    if "text_config" not in config:
        config["text_config"] = config
        
    return config

def patched_get_vocab_base_pre(self, tokenizer) -> str:
    """
    强制识别为 qwen2 分词器。
    """
    print(f"💉 [补丁] 拦截 get_vocab_base_pre。强制返回 'qwen2'。")
    return "qwen2"

# 应用补丁
ModelBase.load_hparams = staticmethod(patched_load_hparams)
TextModel.get_vocab_base_pre = patched_get_vocab_base_pre

def main():
    print(f"--- 正在将 Qwen3-ForcedAligner LLM 转换为 GGUF ---")
    print(f"输入目录: {MODEL_DIR}")

    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"❌ 错误: 在 {MODEL_DIR} 中未找到 config.json")
        return

    # 定义转换任务列表: (输出精度类型, 后缀名)
    tasks = [
        ("f16", "f16"),
        ("q8_0", "q8_0")
    ]

    for out_type, suffix in tasks:
        output_file = os.path.join(PROJECT_ROOT, "model", f"qwen3_aligner_llm.{suffix}.gguf")
        print(f"\n--- 正在转换 {out_type} 格式 -> {output_file} ---")

        # 3. 准备转换器参数
        sys.argv = [
            "convert_hf_to_gguf.py",
            MODEL_DIR,
            "--outfile", output_file,
            "--outtype", out_type,
            "--verbose"
        ]

        # 4. 执行转换
        print(f"正在启动 {out_type} 转换流程...\n")
        try:
            convert_hf_to_gguf.main()
            print(f"\n✅ {out_type} 转换成功！")
            print(f"GGUF 模型路径: {output_file}")
        except Exception as e:
            print(f"\n❌ {out_type} 转换失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
