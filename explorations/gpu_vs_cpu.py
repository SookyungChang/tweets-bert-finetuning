import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import torch 
import cpuinfo
import platform
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from src.training.train_bert import train
from dotenv import load_dotenv
load_dotenv()
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_system_info():
    info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "OS Release": platform.release(),
        "Architecture": platform.machine(),
        "Python Version": platform.python_version(),
        "CPU": cpuinfo.get_cpu_info()['brand_raw'],
        "CPU Cores": psutil.cpu_count(logical=False),
        "CPU Threads": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available",  
    }
    return info

def display_system_info(info):
    table = Table(show_header=False, box=None)
    for key, value in info.items():
        table.add_row(f"[bold cyan]{key}[/bold cyan]", f"{value}")
    panel = Panel(table, title="[bold magenta]System Information[/bold magenta]", border_style="magenta")
    console.print(panel)

def model_train(sample_size: int, device: str):
    bert = train(log_name=f"bert-{device}-{sample_size}", sample_size=sample_size)

def exptract_from_wandb(project_path: str = "amd-beast-owner/bert-finetuning"):
    api = wandb.Api()
    cpu = api.run("amd-beast-owner-personal/bert-finetuning/uyxynbpg")
    gpu = api.run("amd-beast-owner-personal/bert-finetuning/sfnubzah")
    df_cpu, df_gpu = cpu.history(), gpu.history()
    df_cpu['device'] = 'AMD 9950X (CPU)'
    df_gpu['device'] = 'AMD 9070 (GPU)'
    df = pd.concat([df_cpu, df_gpu], ignore_index=True, axis=0)
    return df

def plot(df):
    sns.set_theme(style="white")
    plt.rcParams['font.family'] = 'sans-serif'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
    fig.patch.set_facecolor('#f8f9fa') 

    colors = ['#2c3e50', '#d41717']

    avg_samples = df.groupby('device')['test/samples_per_second'].mean()
    bars1 = ax1.bar(avg_samples.index, avg_samples.values, color=colors, width=0.6)

    ax1.set_title('Inference Throughput: BERT-Base Fine-tuning', fontsize=18, fontweight='bold', pad=25)
    ax1.set_ylabel('Samples / Second', fontsize=13, fontweight='semibold')
    ax1.set_facecolor('#f8f9fa')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:,.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold', color=bar.get_facecolor())

    max_runtime = df.groupby('device')['_runtime'].max()
    bars2 = ax2.bar(max_runtime.index, max_runtime.values, color=colors, width=0.6)

    ax2.set_title('Total Execution Time (Single Epoch)', fontsize=18, fontweight='bold', pad=25)
    ax2.set_ylabel('Seconds', fontsize=13, fontweight='semibold')
    ax2.set_xlabel('Hardware Configuration', fontsize=14, fontweight='semibold', labelpad=20)
    ax2.set_facecolor('#f8f9fa')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{int(height)}s', ha='center', va='bottom', fontsize=14, fontweight='bold', color=bar.get_facecolor())

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6) # Subtle gridlines for context
        ax.tick_params(axis='x', labelsize=14)

    plt.tight_layout(pad=5.0)
    png_name = "inference_throughput_gpu_vs_cpu.png"
    save_path = os.path.join(current_dir, png_name)
    plt.savefig(save_path, dpi=300) 
    print(f"Plot saved successfully as {png_name}!")

if __name__ == "__main__":
    console = Console()
    display_system_info(info=get_system_info())
    # sample_size = 20000
    # device = "cpu"
    df = exptract_from_wandb()
    plot(df)