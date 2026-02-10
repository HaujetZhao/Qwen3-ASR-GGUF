# coding=utf-8
import torch
import os
import sys
import numpy as np
import librosa
from pathlib import Path

# 将当前目录加入 path
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr import Qwen3ASRModel
from export_config import ASR_MODEL_DIR

def save_tensor(name, tensor, folder):
    """将 Tensor 转换为 Numpy 并保存"""
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        data = tensor
    else:
        # 处理可能的封装类型
        try:
            data = tensor.detach().cpu().numpy()
        except:
            data = np.array(tensor)
    
    path = os.path.join(folder, f"{name}.npy")
    np.save(path, data)
    print(f"✅ 已保存 {name}: {data.shape} -> {path}")

def main():
    # 1. 环境准备
    output_folder = "capture_data"
    os.makedirs(output_folder, exist_ok=True)
    
    model_path = str(ASR_MODEL_DIR)
    audio_path = "test.mp3"
    
    # 2. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 正在初始化模型 (Device: {device}) ---")
    
    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.float32,  # 使用 float32 捕捉数据更精确
        device_map=device,
    )
    
    captured_data = {}

    # 3. 核心组件定位
    # asr.model 是 Qwen3ASRForConditionalGeneration
    # asr.model.thinker 是 Qwen3ASRThinkerForConditionalGeneration
    # asr.model.thinker.audio_tower 是 Qwen3ASRAudioEncoder
    audio_tower = asr.model.thinker.audio_tower

    # 4. 挂载钩子 (Hooks)
    
    # A. 捕获 Encoder 前端 (CNN) 的输出 (进入 Transformer 之前的特征)
    def hook_conv_out(module, input, output):
        # output shape: [b, t, d]
        captured_data["encoder_frontend_output"] = output

    # B. 捕获 Encoder 后端 (Transformer) 的输入
    def hook_backend_input(module, input, kwargs):
        # input[0] 是 hidden_states
        # input[1] 是 cu_seqlens
        captured_data["encoder_backend_input"] = input[0]
        if len(input) > 1:
            captured_data["encoder_cu_seqlens"] = input[1]

    # C. 捕获 Encoder 后端 (Transformer) 的最终输出
    def hook_audio_tower_final(module, input, output):
        # output 是 BaseModelOutput, 包含 last_hidden_state
        if hasattr(output, "last_hidden_state"):
            captured_data["encoder_backend_output"] = output.last_hidden_state
        else:
            captured_data["encoder_backend_output"] = output[0]

    # 挂载
    handles = []
    handles.append(audio_tower.conv_out.register_forward_hook(hook_conv_out))
    # 拦截第一层 EncoderLayer 的输入作为后端输入
    handles.append(audio_tower.layers[0].register_forward_pre_hook(hook_backend_input, with_kwargs=True))
    # 拦截整个 audio_tower 的输出作为后端最终输出
    handles.append(audio_tower.register_forward_hook(hook_audio_tower_final))

    print(f"已成功挂载 {len(handles)} 个钩子")

    # 5. 执行推理
    print("\n--- 执行推理并捕捉数据... ---")
    try:
        # 为了捕捉 Mel 输入，我们手动调用一次 processor
        audio_array, sr = librosa.load(audio_path, sr=16000)
        # Qwen3-ASR 内部会把音频切成 chunk，这里我们只捕捉第一个 chunk 的 Processor 输出作为参考
        inputs = asr.processor(
            text=["<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio|><|im_end|>\n<|im_start|>assistant\n"],
            audio=audio_array,
            return_tensors="pt",
            padding=True,
        )
        captured_data["input_mel"] = inputs["input_features"]
        captured_data["feature_attention_mask"] = inputs["feature_attention_mask"]

        # 执行正式转录 (会触发 Hooks)
        results = asr.transcribe(
            audio=audio_path,
            language=None,
            return_time_stamps=False,
        )
        
        print("\n识别结果:")
        for res in results:
            print(f"[{res.language}] {res.text[:50]}...")

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 移除钩子
        for h in handles:
            h.remove()

    # 6. 保存数据
    print("\n--- 开始保存捕捉到的数据 ---")
    for name, tensor in captured_data.items():
        save_tensor(name, tensor, output_folder)

    print(f"\n✨ 数据捕捉完成，全部文件保存在: {output_folder}/")

if __name__ == "__main__":
    main()
