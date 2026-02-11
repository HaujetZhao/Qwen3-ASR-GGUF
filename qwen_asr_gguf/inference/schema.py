# coding=utf-8
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional
import numpy as np

class MsgType(Enum):
    CMD_ENCODE = auto()   # 主进程 -> Encoder: 编码请求
    CMD_STOP = auto()     # 主进程 -> Encoder: 停止请求
    MSG_EMBD = auto()     # Encoder -> 主进程: 返回特征
    MSG_READY = auto()    # Encoder -> 主进程: 就绪信号
    MSG_DONE = auto()     # Encoder -> 主进程: 已退出信号

@dataclass
class StreamingMessage:
    """音频编码进程通信协议"""
    msg_type: MsgType
    data: Any = None         # 存放音频 chunk 或 embedding 结果
    is_last: bool = False    # 标记是否为最后一段音频
    encode_time: float = 0.0 # 编码器实际耗时

@dataclass
class DecodeResult:
    """LLM 解码内核输出标准化"""
    text: str = ""           # 包含前缀的完整文本
    new_text: str = ""       # 本次增量生成的文本
    stable_tokens: List[int] = field(default_factory=list)
    t_prefill: float = 0.0   # 预填充耗时 (ms)
    t_generate: float = 0.0  # 生成耗时 (ms)
    n_prefill: int = 0       # 预填充 token 数
    n_generate: int = 0      # 生成 token 数
    is_aborted: bool = False # 是否因重复或其他原因熔断中断

@dataclass(frozen=True)
class ForcedAlignItem:
    """单个词/字符的对齐结果"""
    text: str
    start_time: float        # 单位：秒
    end_time: float          # 单位：秒

@dataclass
class ForcedAlignResult:
    """对齐结果标准化集合 (官方结构化输出格式)"""
    items: List[ForcedAlignItem]
    performance: Optional[dict] = None

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]

@dataclass
class AlignerConfig:
    """对齐引擎配置"""
    model_dir: str
    encoder_fn: str = "qwen3_aligner_encoder.int8.onnx"
    llm_fn: str = "qwen3_aligner_llm.q8_0.gguf"
    mel_fn: str = "mel_filters.npy"
    use_dml: bool = False
    n_ctx: int = 8192
    verbose: bool = True

@dataclass
class ASREngineConfig:
    """ASR 识别引擎配置"""
    model_dir: str
    encoder_fn: str = "qwen3_asr_encoder.int8.onnx"
    llm_fn: str = "qwen3_asr_llm.q8_0.gguf"
    mel_fn: str = "mel_filters.npy"
    use_dml: bool = False
    n_ctx: int = 4096
    verbose: bool = True
    enable_aligner: bool = False
    align_config: Optional[AlignerConfig] = None

    def __post_init__(self):
        if self.align_config is None:
            self.align_config = AlignerConfig(
                model_dir=self.model_dir,
                use_dml=self.use_dml,
                verbose=self.verbose
            )

@dataclass
class TranscribeResult:
    """ASR 转录结果 (含可选的对齐信息)"""
    text: str
    alignment: Optional[ForcedAlignResult] = None
    performance: Optional[dict] = None
