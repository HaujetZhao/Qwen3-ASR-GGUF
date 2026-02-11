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
from .asr_worker import asr_helper_worker_proc
from .utils import normalize_language_name, validate_language
from . import llama

class QwenASREngine:
    """Qwen3-ASR 流式转录引擎 (GGUF 后端) - 统一辅助进程架构"""
    def __init__(self, config: ASREngineConfig):
        self.verbose = config.verbose
        if self.verbose: print(f"--- [QwenASR] 初始化引擎 (DML: {config.use_dml}) ---")

        from qwen_asr_gguf.inference import llama
        self.llama_mod = llama # keep reference
        
        # 路径解析
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)

        # 1. 加载识别 LLM
        self.model = llama.LlamaModel(llm_gguf)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=4096, embeddings=False)
        
        # 2. 启动统一辅助子进程 (编码 + 对齐)
        self.to_worker_q = mp.Queue()
        self.from_enc_q = mp.Queue()
        self.from_align_q = mp.Queue()
        
        self.helper_proc = mp.Process(
            target=asr_helper_worker_proc, 
            args=(self.to_worker_q, self.from_enc_q, self.from_align_q, config), 
            daemon=True
        )
        self.helper_proc.start()
        
        # 3. 等待子进程就绪信号 (包含 Encoder 预热完成)
        msg = self.from_enc_q.get()
        if msg.msg_type == MsgType.MSG_READY and self.verbose:
            print("--- [QwenASR] 辅助进程已就绪 ---")

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def shutdown(self):
        # 向辅助进程发送停止信号
        if self.helper_proc:
            self.to_worker_q.put(StreamingMessage(MsgType.CMD_STOP))
            self.helper_proc.join()
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
        """内部方法：执行单次 LLM 生成循环（仅负责推理）"""
        result = DecodeResult()
        
        total_len = full_embd.shape[0]
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        batch = self.llama_mod.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        # 1. Prefill
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start
        
        # 2. Generation Loop（使用新采样器和随机种子）
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        cur_pos = total_len
        gen_batch = self.llama_mod.LlamaBatch(4, 0, 1)
        text_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        
        # 每次解码使用新的随机种子
        seed = int(np.random.randint(0, 2**31 - 1))
        sampler = self.llama_mod.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)
        for _ in range(150): # Max new tokens per chunk
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break
            
            gen_batch.set_token(last_sampled_token, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            self.ctx.decode(gen_batch)
            
            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    print(re.sub('([，。？！])', '\\1\n', piece), end='', flush=True)
                    stable_text_acc += piece
            
            # 熔断检查：检测重复循环
            if len(stable_tokens) > 15:
                if len(set(stable_tokens[-15:])) <= 3:
                    result.is_aborted = True
                    break
            
            cur_pos += 1
            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1
            
        gen_time = time.time() - t_gen_start
        del sampler  # 释放采样器资源
            
        if is_last_chunk and not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    print(re.sub('([，。？！])', '\\1\n', piece), end="", flush=True)
                    stable_text_acc += piece
            final_p = text_decoder.decode(b"", final=True)
            if final_p: 
                print(final_p, end='', flush=True)
                stable_text_acc += final_p
        
        # 填充结果（内核输出标准化）
        result.text = prefix_text + stable_text_acc
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        return result

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
        """运行完整转录流水线 (异步对齐 - 单通道版)"""
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
        align_tasks_count = 0
        
        # 统计指标
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "wait_time": 0.0, "encode_time": 0.0,
            "align_enc_time": 0.0, "align_dec_time": 0.0
        }
        t_main_start = time.time()

        def send_enc_chunk(idx):
            s, e = idx * samples_per_chunk, min((idx + 1) * samples_per_chunk, total_len)
            data = audio[s:e]
            if len(data) < samples_per_chunk: 
                data = np.pad(data, (0, samples_per_chunk - len(data)))
            self.to_worker_q.put(StreamingMessage(MsgType.CMD_ENCODE, data=data, is_last=(idx == num_chunks - 1)))

        def send_align_task(idx, text, is_last):
            nonlocal align_tasks_count
            if (self.helper_proc and self.helper_proc.is_alive()) and text.strip():
                s, e = idx * samples_per_chunk, min((idx + 1) * samples_per_chunk, total_len)
                audio_slice = audio[s:e]
                
                self.to_worker_q.put(StreamingMessage(
                    msg_type=MsgType.CMD_ALIGN,
                    data=audio_slice,
                    text=text,
                    offset_sec=float(idx * chunk_size_sec),
                    language=language,
                    is_last=is_last
                ))
                align_tasks_count += 1

        if num_chunks > 0: send_enc_chunk(0)

        for i in range(num_chunks):
            # 1. 获取特征
            t_w_start = time.time()
            msg = self.from_enc_q.get()
            stats["wait_time"] += (time.time() - t_w_start)
            stats["encode_time"] += msg.encode_time
            
            current_embd = msg.data
            was_last = msg.is_last
            
            # 提前触发下一块特征提取
            if not was_last: send_enc_chunk(i + 1)
            
            # 2. 构建记忆并推理
            prefix_text = "".join([seg['text'] for seg in history_segments])
            combined_audio_embd = np.concatenate([seg['embd'] for seg in history_segments] + [current_embd], axis=0)
            full_embd = self._build_prompt_embd(combined_audio_embd, prefix_text, context, language)
            
            temp = temperature
            for retry in range(6):
                res = self._run_llm_buffered(full_embd, prefix_text, rollback_num, is_last_chunk=was_last, temperature=temp)
                if not res.is_aborted: break
                temp += 0.3
                if self.verbose: print(f"\n[ASR] 熔断重启 (Temp={temp:.1f})")
            
            new_text_part = res.text[len(prefix_text):]
            history_segments.append({'embd': current_embd, 'text': new_text_part})
            total_full_text += new_text_part
            
            # --- 异步下发对齐任务 ---
            send_align_task(i, new_text_part, was_last)

            stats["prefill_tokens"] += res.n_prefill; stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate; stats["decode_time"] += res.t_generate

        # 3. 回收所有对齐结果
        if align_tasks_count > 0:
            if self.verbose: print(f"\n--- [QwenASR] 正在回收 {align_tasks_count} 个对齐结果... ---")
            for _ in range(align_tasks_count):
                align_msg = self.from_align_q.get()
                if align_msg.msg_type == MsgType.MSG_ALIGN and align_msg.data:
                    align_res: ForcedAlignResult = align_msg.data
                    all_aligned_items.extend(align_res.items)
                    if align_res.performance:
                        stats["align_enc_time"] += align_res.performance.get("encoder_time", 0)
                        stats["align_dec_time"] += align_res.performance.get("decoder_time", 0)

        # 4. 排序结果 (防止子进程回收乱序)
        all_aligned_items.sort(key=lambda x: x.start_time)

        t_total = time.time() - t_main_start
        audio_duration = total_len / sr

        if self.verbose:
            rtf = t_total / audio_duration if audio_duration > 0 else 0
            pre_speed = stats["prefill_tokens"] / (stats["prefill_time"]) if stats["prefill_time"] > 0 else 0
            gen_speed = stats["decode_tokens"] / (stats["decode_time"]) if stats["decode_time"] > 0 else 0
            
            print(f"\n\n📊 性能统计:")
            print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
            print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
            print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
            print(f"  🔹 编码等待    : {stats['wait_time']:.2f} 秒")
            if self.helper_proc:
                print(f"  🔹 对齐总时    : {stats['align_enc_time']+stats['align_dec_time']:.2f} 秒 (子进程并行 Enc:{stats['align_enc_time']:.2f}s, Dec:{stats['align_dec_time']:.2f}s)")
            print(f"  🔹 LLM 预填充  : {stats['prefill_time']:.3f} 秒 ({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)")
            print(f"  🔹 LLM 生成    : {stats['decode_time']:.3f} 秒 ({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)")
            
        return TranscribeResult(
            text=total_full_text,
            alignment=ForcedAlignResult(items=all_aligned_items) if all_aligned_items else None,
            performance=stats
        )
