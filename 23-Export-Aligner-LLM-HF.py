# coding=utf-8
import os
import sys
import json
import torch
from pathlib import Path
from safetensors.torch import save_file

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_asr import Qwen3ForcedAligner
from export_config import ALIGNER_MODEL_DIR, EXPORT_DIR

def extract_aligner_llm():
    print(f"--- 正在准备导出对齐器 LLM (Thinker) ---")
    print(f"源路径: {ALIGNER_MODEL_DIR}")
    
    # 1. 加载对齐模型
    try:
        aligner = Qwen3ForcedAligner.from_pretrained(
            str(ALIGNER_MODEL_DIR),
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    output_dir = Path(EXPORT_DIR) / "aligner_decoder_hf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thinker = aligner.model.thinker
    
    # 2. 处理配置
    print("正在生成配置文件...")
    # 提取 text_config
    text_config = thinker.config.text_config
    llm_config_dict = text_config.to_dict()
    
    # 伪装逻辑
    llm_config_dict["architectures"] = ["Qwen3VLForConditionalGeneration"]
    llm_config_dict["model_type"] = "qwen3_vl"
    
    # 强制修正词表大小为 152064 (对齐 Embedding 层)
    target_vocab_size = 152064
    llm_config_dict["vocab_size"] = target_vocab_size
    
    # 保存 config
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(llm_config_dict, f, indent=2, ensure_ascii=False)
    
    # 3. 提取并转换权重
    print("正在处理权重矩阵...")
    full_state_dict = thinker.state_dict()
    new_state_dict = {}
    
    for key, weight in full_state_dict.items():
        if key.startswith("model."):
            # 基础模型部分直接复制
            new_state_dict[key] = weight
        elif key == "lm_head.weight":
            # 核心难点：lm_head.weight 的补齐
            # 原始 [5000, 1024] -> 目标 [152064, 1024]
            original_shape = weight.shape
            print(f"补齐 lm_head.weight: {original_shape} -> [{target_vocab_size}, {original_shape[1]}]")
            
            # 使用一个极大的负数填充剩余部分，确保推理时不会误选
            padded_weight = torch.full((target_vocab_size, original_shape[1]), -100.0, dtype=weight.dtype)
            padded_weight[:original_shape[0], :] = weight
            new_state_dict[key] = padded_weight

    # 4. 保存 Safetensors
    print(f"正在保存权重文件到: {output_dir}")
    save_file(new_state_dict, output_dir / "model.safetensors")
    
    # 5. 保存分词器
    print("正在保存分词器...")
    aligner.processor.tokenizer.save_pretrained(output_dir)
    
    print("\n✅ 对齐器 LLM 已成功提取并伪装为 HF 格式！")
    print(f"文件位置: {output_dir}")

if __name__ == "__main__":
    extract_aligner_llm()
