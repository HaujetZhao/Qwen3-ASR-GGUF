import os
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

def profile_onnx_model(model_path, input_dict):
    print(f"\\n{'-'*60}")
    print(f"正在 Profiling {os.path.basename(model_path)}...")
    print(f"{'-'*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ 无法找到模型文件: {model_path}")
        return
        
    sess_opts = ort.SessionOptions()
    sess_opts.enable_profiling = True
    
    providers = ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    except Exception as e:
        print(f"❌ 无法载入模型: {e}")
        return

    RUNS = 2
    print(f"启动预热和 Profile 推理 (运行 {RUNS} 次)...")
    for _ in range(RUNS):
        session.run(None, input_dict)
        
    profile_file = session.end_profiling()
    fixed_profile_name = os.path.basename(model_path) + ".profile.json"
    if os.path.exists(fixed_profile_name):
        os.remove(fixed_profile_name)
    os.rename(profile_file, fixed_profile_name)
    profile_file = fixed_profile_name
    print(f"✅ Profile 结果已提取并固定保存至: {profile_file}\\n")
    
    with open(profile_file, "r") as f:
        data = json.load(f)
        
    op_durations = {}
    for entry in sorted(data, key=lambda x: x.get("ts", 0)):
        if "cat" in entry and entry["cat"] == "Node":
            op_name = entry.get("args", {}).get("op_name", entry.get("name", "Unknown"))
            dur = entry.get("dur", 0) # microseconds
            if op_name not in op_durations:
                op_durations[op_name] = []
            op_durations[op_name].append(dur)
            
    filtered_ops = {}
    for op_name, durs in op_durations.items():
        if len(durs) >= RUNS:
            ops_per_run = len(durs) // RUNS
            valid_durs = durs[ops_per_run:]  # 丢弃第一遍预热的记录
            avg_dur = sum(valid_durs) / (RUNS - 1) / 1000.0  # ms
            count_per_run = len(valid_durs) // (RUNS - 1)
        else:
            avg_dur = sum(durs) / len(durs) / 1000.0
            count_per_run = len(durs)
        filtered_ops[op_name] = (avg_dur, count_per_run)

    filtered_ops = {k: v for k, v in filtered_ops.items() if v[0] > 0}
    sorted_ops = sorted(filtered_ops.items(), key=lambda x: x[1][0], reverse=True)
    total_dur = sum(v[0] for v in filtered_ops.values())

    print(f"📊 [算子耗时分布] {os.path.basename(model_path)}")
    print(f"{'算子类型 (OpType)':<25} | {'次数':>5} | {'总耗时 (ms)':<15} | {'占比 (%)':<10}")
    print("-" * 72)
    for op_name, (dur_ms, count) in sorted_ops:
        pct = (dur_ms / total_dur) * 100.0 if total_dur > 0 else 0
        if pct < 0.1: continue
        print(f"{op_name:<25} | {count:>4}次 | {dur_ms:<15.3f} | {pct:>5.1f}%")
    print("-" * 72)
    print(f"{'总计 (Total)':<25} |       | {total_dur:<15.3f} | 100.0%")
    print("(注: DML 在异步执行时 CPU/GPU 时间线存在交叉，统计到的通常是驱动分发或同步时间，但百分比极其精准地揭示了黑洞算子)")


def main():
    model_dir = os.path.join(Path(__file__).parent.absolute(), "model")
    
    frontend_path = os.path.join(model_dir, "qwen3_asr_encoder_frontend.int8.onnx")
    backend_path = os.path.join(model_dir, "qwen3_asr_encoder_backend.int8.onnx")

    data_type = np.float32
        
    if os.path.exists(frontend_path):
        dummy_frontend = {
            "chunk_mel": np.random.randn(1, 128, 100).astype(data_type)
        }
        profile_onnx_model(frontend_path, dummy_frontend)
        
    if os.path.exists(backend_path):
        hidden_size = 1024
        print(f"\\n🔧 检测到 Backend 隐藏层维度 (hidden_size): {hidden_size}")
        seq_len = 2000
        dummy_backend = {
            "hidden_states": np.random.randn(1, seq_len, hidden_size).astype(data_type),
            "attention_mask": np.zeros((1, 1, seq_len, seq_len), dtype=data_type)
        }
        profile_onnx_model(backend_path, dummy_backend)

if __name__ == '__main__':
    main()
