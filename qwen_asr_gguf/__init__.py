"""
FunASR-GGUF: 混合 ASR 推理引擎

使用 ONNX Runtime (encoder/CTC) + llama.cpp (GGUF decoder) 进行语音识别

API 兼容 sherpa-onnx，可直接替换使用。
"""

import logging
import sys
import os

def setup_logging(level: int = logging.WARNING, log_file: str = os.path.join("logs", "latest.log")):
    """
    配置全局日志

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名

    Returns:
        配置好的 logger 实例
    """
    # 获取根 logger
    root_logger = logging.getLogger('qwen_asr_gguf')
    root_logger.setLevel(logging.DEBUG)  # 接收所有级别的日志
    root_logger.handlers.clear()  # 清除已有处理器


    # 文件处理器
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG) # 文件通常记录更详细的信息
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# 初始化默认日志配置（默认 WARNING 级别，记录到 logs/latest.log）
logger = setup_logging(level=logging.DEBUG)

