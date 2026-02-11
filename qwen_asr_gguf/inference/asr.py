# coding=utf-8
import os
import time
import re
import codecs
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import deque
from typing import Optional, List

from .schema import MsgType, StreamingMessage, DecodeResult, ASREngineConfig, TranscribeResult, ForcedAlignItem, ForcedAlignResult
from .encoder import encoder_worker_proc
from .utils import normalize_language_name, validate_language
from .aligner import QwenForcedAligner
from . import llama

class QwenASREngine:
    """Qwen3-ASR 流式转录引擎 (GGUF 后端)"""
    def __init__(self, config: ASREngineConfig):
        self.verbose = config.verbose
        if self.verbose: print(f"--- [QwenASR] 初始化引擎 (DML: {config.use_dml}) ---")
        
        # 路径解析
        encoder_onnx = os.path.join(config.model_dir, config.encoder_fn)
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)
        mel_filters = os.path.join(config.model_dir, config.mel_fn)

        # 1. 加载识别 LLM
        self.model = llama.LlamaModel(llm_gguf)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=4096, embeddings=False)
        
        # 2. 启动音频编码器子进程
        self.to_enc_q = mp.Queue()
        self.from_enc_q = mp.Queue()
        self.enc_proc = mp.Process(
            target=encoder_worker_proc, 
            args=(self.to_enc_q, self.from_enc_q, encoder_onnx, mel_filters, 40.0, config.use_dml), 
            daemon=True
        )
        self.enc_proc.start()
        
        # 3. 按需加载对齐器 (主进程)
        self.aligner = None
        if config.enable_aligner:
            self.aligner = QwenForcedAligner(config.align_config)

        # 等待编码器预热及就绪信号
        msg = self.from_enc_q.get()
        if msg.msg_type == MsgType.MSG_READY and self.verbose:
            print("--- [QwenASR] 编码器已就绪 ---")

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def shutdown(self):
        self.to_enc_q.put(StreamingMessage(MsgType.CMD_STOP))
        self.enc_proc.join()
        if self.verbose: print("--- [QwenASR] 引擎已关闭 ---")

    def _build_prompt_embd(self, audio_embd: np.ndarray, prefix_text: str, context: Optional[str], language: Optional[str]):
        """构造用于 LLM 输入的 Embedding 序列 (区块化打包模式)"""
        def tk(t): return self.model.tokenize(t)

        # 1. 区块 A: 音频之前的所有内容 (System + User Header)
        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = [self.ID_IM_START] + tk(prefix_str) + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]
        
        # 2. 区块 B: 音频之后的所有内容 (Instruction + Assistant Header + History)
        suffix_head = f"assistant\n"
        if language: suffix_head += f"language {language}"
        
        suffix_tokens = [self.ID_AUDIO_END] + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(suffix_head) + [self.ID_ASR_TEXT] + tk(prefix_text)

        # 3. 统计并拼接
        n_pre, n_aud, n_suf = len(prefix_tokens), audio_embd.shape[0], len(suffix_tokens)
        total_embd = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32)
        
        total_embd[:n_pre] = self.embedding_table[prefix_tokens]
        total_embd[n_pre : n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud:] = self.embedding_table[suffix_tokens]
        
        return total_embd

    def _run_llm_buffered(
        self, 
        full_embd: np.ndarray,
        prefix_text: str, 
        rollback_num: int,
        is_last_chunk: bool = False, 
        temperature: float = 0.4
    ) -> DecodeResult:
        """核心推理循环：实现了回滚缓冲区、实时流式输出和重复熔断机制"""
        res = DecodeResult()
        t_len = full_embd.shape[0]
        
        # 1. Prefill
        pos_base = np.arange(0, t_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(t_len, dtype=np.int32)])
        batch = llama.LlamaBatch(max(t_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        self.ctx.clear_kv_cache()
        t0 = time.time()
        self.ctx.decode(batch)
        res.t_prefill = (time.time() - t0) * 1000
        res.n_prefill = t_len

        # 2. Generation Loop
        t1 = time.time()
        n_gen = 0
        display_queue = deque()
        stable_text_acc = ""
        cur_pos = t_len
        gen_batch = llama.LlamaBatch(4, 0, 1)
        utf8_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        
        # 采样初始化
        seed = int(np.random.randint(0, 2**31 - 1))
        sampler = llama.LlamaSampler(temperature=temperature, seed=seed)
        token = sampler.sample(self.ctx.ptr)
        
        for _ in range(150): # Max new tokens per chunk
            if token in [self.model.eos_token, self.ID_IM_END]: break
            
            gen_batch.set_token(token, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            self.ctx.decode(gen_batch)
            
            display_queue.append(token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                res.stable_tokens.append(ready_token)
                piece = utf8_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    # 使用正则处理换行输出
                    print(re.sub('([，。？！])', '\\1\n', piece), end='', flush=True)
                    stable_text_acc += piece
            
            # 熔断检测
            if len(res.stable_tokens) > 30 and len(set(res.stable_tokens[-30:])) <= 3:
                res.is_aborted = True; break
            
            cur_pos += 1
            n_gen += 1
            token = sampler.sample(self.ctx.ptr)
            
        # 最后一块音频的强制刷新
        if is_last_chunk and not res.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                res.stable_tokens.append(t)
                piece = utf8_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    print(re.sub('([，。？！])', '\\1\n', piece), end='', flush=True)
                    stable_text_acc += piece
            final_p = utf8_decoder.decode(b"", final=True)
            if final_p:
                print(final_p, end='', flush=True)
                stable_text_acc += final_p
        
        res.t_generate = (time.time() - t1) * 1000
        res.n_generate = n_gen
        res.text = prefix_text + stable_text_acc
        res.new_text = stable_text_acc
        return res

    def transcribe(
        self, 
        audio: np.ndarray,
        context: str = "",
        language: str = "Chinese",
        chunk_size_sec: float = 40.0,
        memory_chunks: int = 2,
        temperature: float = 0.4,
        rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线"""
        # 语言归一化与校验
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        num_chunks = int(np.ceil(total_len / samples_per_chunk))
        
        history_segments = deque(maxlen=memory_chunks)
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []
        last_char_end_time = 0.0
        
        # 统计指标 (与 21 脚本一致)
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "wait_time": 0.0, "encode_time": 0.0,
            "align_enc_time": 0.0, "align_dec_time": 0.0
        }
        t_main_start = time.time()

        def send_chunk(idx):
            s, e = idx * samples_per_chunk, min((idx + 1) * samples_per_chunk, total_len)
            data = audio[s:e]
            if len(data) < samples_per_chunk: 
                data = np.pad(data, (0, samples_per_chunk - len(data)))
            self.to_enc_q.put(StreamingMessage(MsgType.CMD_ENCODE, data=data, is_last=(idx == num_chunks - 1)))

        if num_chunks > 0: send_chunk(0)

        for i in range(num_chunks):
            # 获取特征 (生产者-消费者)
            t_w_start = time.time()
            msg = self.from_enc_q.get()
            stats["wait_time"] += (time.time() - t_w_start)
            stats["encode_time"] += msg.encode_time
            
            current_embd = msg.data
            was_last = msg.is_last
            
            # 提前触发下一块
            if not was_last: send_chunk(i + 1)
            
            # 构建记忆
            prefix_text = "".join([seg['text'] for seg in history_segments])
            combined_audio = np.concatenate([seg['embd'] for seg in history_segments] + [current_embd], axis=0)
            
            full_embd = self._build_prompt_embd(combined_audio, prefix_text, context, language)
            
            # LLM 推理 (带熔断重启机制)
            temp = temperature
            for retry in range(6):
                res = self._run_llm_buffered(full_embd, prefix_text, rollback_num, is_last_chunk=was_last, temperature=temp)
                if not res.is_aborted: break
                temp += 0.3
                if self.verbose: print(f"\n[ASR] 熔断重启 (Temp={temp:.1f})")
            
            # 更新文本
            new_text_part = res.text[len(prefix_text):]
            history_segments.append({'embd': current_embd, 'text': new_text_part})
            total_full_text += new_text_part
            
            # --- 集成对齐逻辑 (渐进式) ---
            if self.aligner and new_text_part.strip():
                # 计算音频切片：以上一个字的结束时间为起点
                start_sample = int(last_char_end_time * sr)
                end_sample = min((i + 1) * samples_per_chunk, total_len)
                
                # 如果当前已经是最后一块，确保包含了全部音频
                if was_last: end_sample = total_len
                
                if end_sample > start_sample:
                    audio_slice = audio[start_sample:end_sample]
                    align_res = self.aligner.align(
                        audio_slice, 
                        new_text_part, 
                        language=language, 
                        offset_sec=last_char_end_time
                    )
                    if align_res.items:
                        all_aligned_items.extend(align_res.items)
                        last_char_end_time = align_res.items[-1].end_time
                        if align_res.performance:
                            stats["align_enc_time"] += align_res.performance.get("encoder_time", 0)
                            stats["align_dec_time"] += align_res.performance.get("decoder_time", 0)

            stats["prefill_tokens"] += res.n_prefill; stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate; stats["decode_time"] += res.t_generate

        t_total = time.time() - t_main_start
        audio_duration = total_len / sr

        if self.verbose:
            rtf = t_total / audio_duration if audio_duration > 0 else 0
            pre_speed = stats["prefill_tokens"] / (stats["prefill_time"]/1000) if stats["prefill_time"] > 0 else 0
            gen_speed = stats["decode_tokens"] / (stats["decode_time"]/1000) if stats["decode_time"] > 0 else 0
            
            print(f"\n\n📊 性能统计:")
            print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
            print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
            print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
            print(f"  🔹 编码等待    : {stats['wait_time']:.2f} 秒")
            if self.aligner:
                print(f"  🔹 对齐耗时    : {stats['align_enc_time']+stats['align_dec_time']:.2f} 秒 (Enc: {stats['align_enc_time']:.2f}s, Dec: {stats['align_dec_time']:.2f}s)")
            print(f"  🔹 LLM 预填充  : {stats['prefill_time']/1000:.3f} 秒 ({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)")
            print(f"  🔹 LLM 生成    : {stats['decode_time']/1000:.3f} 秒 ({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)")
            
        return TranscribeResult(
            text=total_full_text,
            alignment=ForcedAlignResult(items=all_aligned_items) if all_aligned_items else None,
            performance=stats
        )
