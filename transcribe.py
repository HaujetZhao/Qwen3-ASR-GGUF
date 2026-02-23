# coding=utf-8
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# 获取项目根目录 (适配打包环境)
if getattr(sys, 'frozen', False):
    # 打包环境：sys.executable 位于 dist/Project/ 根目录
    PROJ_DIR = Path(sys.executable).parent
else:
    # 源码环境
    PROJ_DIR = Path(__file__).parent


import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, AlignerConfig, exporters

app = typer.Typer(help="Qwen3-ASR GGUF 命令行转录工具", add_completion=False)
console = Console()

def get_model_filenames(precision: str, is_aligner: bool = False):
    """根据精度返回对应的模型文件名"""
    prefix = "qwen3_aligner" if is_aligner else "qwen3_asr"
    return {
        "frontend": f"{prefix}_encoder_frontend.{precision}.onnx",
        "backend": f"{prefix}_encoder_backend.{precision}.onnx"
    }

def check_model_files(config: ASREngineConfig):
    """检查模型文件完整性"""
    missing_files = []
    
    # ASR 核心文件
    asr_llm = Path(config.model_dir) / config.llm_fn
    asr_frontend = Path(config.model_dir) / config.encoder_frontend_fn
    asr_backend = Path(config.model_dir) / config.encoder_backend_fn
    
    for f in [asr_llm, asr_frontend, asr_backend]:
        if not f.exists():
            missing_files.append(str(f))
            
    # Aligner 文件
    if config.enable_aligner and config.align_config:
        align_llm = Path(config.align_config.model_dir) / config.align_config.llm_fn
        align_frontend = Path(config.align_config.model_dir) / config.align_config.encoder_frontend_fn
        align_backend = Path(config.align_config.model_dir) / config.align_config.encoder_backend_fn
        
        for f in [align_llm, align_frontend, align_backend]:
            if not f.exists():
                missing_files.append(str(f))
    
    if missing_files:
        console.print("\n[bold red]错误：找不到以下所需模型文件：[/bold red]")
        for f in missing_files:
            console.print(f"  - {f}")
        console.print("\n[bold yellow]请到以下链接下载模型文件，并解压到 model 目录：[/bold yellow]")
        console.print("[blue]https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models[/blue]\n")
        raise typer.Exit(code=1)

@app.command()
def transcribe(
    files: List[Path] = typer.Argument(..., help="要转录的音频文件列表"),
    
    # 组 1: 模型与硬件
    model_dir: str = typer.Option(str(PROJ_DIR / "model"), "--model-dir", "-m", help="模型权重根目录", rich_help_panel="模型配置"),
    precision: str = typer.Option("int4", "--prec", help="编码器精度: fp32, fp16, int8, int4", rich_help_panel="模型配置"),
    timestamp: bool = typer.Option(True, "--timestamp/--no-ts", help="是否开启时间戳引擎", rich_help_panel="模型配置"),
    use_dml: bool = typer.Option(True, "--dml/--no-dml", help="是否使用 DirectML 加速", rich_help_panel="模型配置"),
    use_vulkan: bool = typer.Option(True, "--vulkan/--no-vulkan", help="是否开启 Vulkan 加速 (设置 GGML_VULKAN=1)", rich_help_panel="模型配置"),
    n_ctx: int = typer.Option(2048, "--n-ctx", help="LLM 上下文窗口大小", rich_help_panel="模型配置"),
    
    # 组 2: 转录逻辑
    language: Optional[str] = typer.Option(None, "--language", "-l", help="强制指定语种 (例: Chinese, English)", rich_help_panel="转录设置"),
    context: str = typer.Option("", "--context", "-p", help="上下文提示词 (Prompt)", rich_help_panel="转录设置"),
    temperature: float = typer.Option(0.4, "--temperature", help="采样温度", rich_help_panel="转录设置"),


    seek_start: float = typer.Option(0.0, "--seek-start", "-ss", help="音频开始位置 (秒)", rich_help_panel="音频切片"),
    duration: Optional[float] = typer.Option(None, "--duration", "-t", help="处理音频的时长 (秒)", rich_help_panel="音频切片"),
    
    # 组 3: 音频裁剪与性能
    chunk_size: float = typer.Option(40.0, "--chunk-size", "-c", help="分段识别时长 (秒)", rich_help_panel="流式配置"),
    memory_num: int = typer.Option(1, "--memory-num", help="记忆的历史片段数量", rich_help_panel="流式配置"),
    
    # 组 4: 其他
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="是否打印详细日志", rich_help_panel="其他选项"),
    yes: bool = typer.Option(False, "--yes", "-y", help="覆盖已存在的输出文件", rich_help_panel="其他选项"),
):
    """
    使用 Qwen3-ASR GGUF 模型对音频进行高精度转录。
    """
    
    # 1. 环境准备
    if not use_vulkan:
        os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan

    # 2. 构造配置
    asr_files = get_model_filenames(precision, is_aligner=False)
    align_files = get_model_filenames(precision, is_aligner=True)

    align_config = None
    if timestamp:
        align_config = AlignerConfig(
            model_dir=model_dir,
            use_dml=use_dml,
            encoder_frontend_fn=align_files["frontend"],
            encoder_backend_fn=align_files["backend"],
            n_ctx=n_ctx
        )

    config = ASREngineConfig(
        model_dir=model_dir,
        use_dml=use_dml,
        encoder_frontend_fn=asr_files["frontend"],
        encoder_backend_fn=asr_files["backend"],
        n_ctx=n_ctx,
        chunk_size=chunk_size,
        memory_num=memory_num,
        enable_aligner=timestamp,
        align_config=align_config,
        verbose=verbose
    )

    # 3. 打印配置面板
    config_table = Table(show_header=False, box=None)
    config_table.add_row("模型目录", f"[green]{model_dir}[/green]")
    config_table.add_row("编码精度", f"[cyan]{precision}[/cyan]")
    config_table.add_row("加速设备", f"DML:{'[green]ON[/green]' if use_dml else '[red]OFF[/red]'} | Vulkan:{'[green]ON[/green]' if use_vulkan else '[red]OFF[/red]'}")
    config_table.add_row("时间戳对齐", f"{'[green]启用[/green]' if timestamp else '[red]禁用[/red]'}")
    config_table.add_row("语言设定", f"{language or '自动识别'}")
    
    console.print(Panel(config_table, title="[bold cyan]Qwen3-ASR 配置选项[/bold cyan]", expand=False))

    # 4. 检查模型文件是否存在
    check_model_files(config)

    # 5. 初始化引擎
    with console.status("[bold yellow]正在初始化引擎，请稍候...[/bold yellow]") as status:
        try:
            t0 = time.time()
            engine = QwenASREngine(config=config)
            init_duration = time.time() - t0
            console.print(f"--- [QwenASR] 引擎初始化耗时: {init_duration:.2f} 秒 ---")
        except Exception as e:
            console.print(f"[bold red]引擎初始化失败: {e}[/bold red]")
            raise typer.Exit(code=1)

    # 6. 循环处理文件
    try:
        for audio_path in files:
            if not audio_path.exists():
                console.print(f"[yellow]跳过不存在的文件: {audio_path}[/yellow]")
                continue

            console.print(f"\n[bold blue]开始处理:[/bold blue] {audio_path.name}\n")
            
            # 检查输出文件冲突
            base_out = audio_path.with_suffix("")
            txt_out = f"{base_out}.txt"
            if Path(txt_out).exists() and not yes:
                if not typer.confirm(f"文件 {txt_out} 已存在，是否覆盖?"):
                    console.print("[yellow]已跳过。[/yellow]")
                    continue

            res = engine.transcribe(
                audio_file=str(audio_path),
                language=language,
                context=context,
                start_second=seek_start,
                duration=duration,
                temperature=temperature,
            )

            # 7. 导出结果
            exporters.export_to_txt(txt_out, res)

            if timestamp and res.alignment:
                srt_out = f"{base_out}.srt"
                json_out = f"{base_out}.json"
                exporters.export_to_srt(srt_out, res)
                exporters.export_to_json(json_out, res)


    finally:
        engine.shutdown()
        console.print("\n[bold green]所有任务已完成。[/bold green]")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app()
