# coding=utf-8
import os
import numpy as np
from transformers import WhisperFeatureExtractor
from export_config import ASR_MODEL_DIR, EXPORT_DIR

# 目标输出路径，确保 model 文件夹存在
os.makedirs(EXPORT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(EXPORT_DIR, "mel_filters.npy")

def main():
    print(f"Loading feature extractor from: {ASR_MODEL_DIR}")
    
    # Qwen3-ASR 使用的是 WhisperFeatureExtractor
    try:
        fe = WhisperFeatureExtractor.from_pretrained(ASR_MODEL_DIR)
        
        # 提取 mel_filters
        if hasattr(fe, 'mel_filters'):
            filters = np.array(fe.mel_filters)
            print(f"✅ Found mel_filters in feature extractor. Shape: {filters.shape}")
        else:
            # 如果没有预设的 filters，则手动计算 (Qwen3-ASR 标准参数: sr=16000, n_fft=400, n_mels=128)
            print("⚠️ mel_filters not found in object, calculating manually...")
            from transformers.models.whisper.feature_extraction_whisper import mel_filter_bank
            
            filters = mel_filter_bank(
                num_frequency_bins=400 // 2 + 1,
                num_mel_filters=128,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=16000,
                mel_scale="slaney",
            )
            print(f"✅ Calculated filters shape: {filters.shape}")

        # 保存结果
        np.save(OUTPUT_FILE, filters)
        print(f"🚀 Successfully saved mel_filters to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")

if __name__ == "__main__":
    main()
