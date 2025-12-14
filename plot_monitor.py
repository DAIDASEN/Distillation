#!/usr/bin/env python
"""
训练曲线可视化工具
用法: python plot_monitor.py [--live] [--save]
"""

import json
import matplotlib.pyplot as plt
import os
import argparse
import time

def load_logs(log_path):
    """加载训练日志"""
    if not os.path.exists(log_path):
        return []
    logs = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except:
                pass
    return logs

def plot_training_curves(logs, save_path=None, show=True):
    """绘制训练曲线"""
    if not logs:
        print("No data to plot")
        return
    
    steps = [l['step'] for l in logs]
    losses = [l['loss'] for l in logs]
    # 兼容新旧日志格式
    student_accs = [l.get('student_acc', l.get('accuracy', 0)) for l in logs]
    teacher_accs = [(l.get('teacher_acc') or 0) for l in logs]  # None -> 0
    teacher_steps = [l['step'] for l in logs if l.get('teacher_acc') is not None]
    teacher_vals = [l['teacher_acc'] for l in logs if l.get('teacher_acc') is not None]
    ans_rates = [l.get('answer_found_rate', 0) for l in logs]
    trunc_rates = [l.get('truncation_rate', 0) for l in logs]
    avg_tokens = [l.get('avg_new_tokens', 0) for l in logs]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Training Progress (Step {steps[-1]})', fontsize=14, fontweight='bold')
    
    # 1. Loss
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, 'b-', alpha=0.3, label='Raw')
    window = min(20, len(losses))
    if window > 1:
        ma_loss = [sum(losses[max(0,i-window):i+1])/min(i+1, window) for i in range(len(losses))]
        ax1.plot(steps, ma_loss, 'b-', linewidth=2, label=f'MA-{window}')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Student vs Teacher Accuracy
    ax2 = axes[0, 1]
    ax2.plot(steps, student_accs, 'g-', alpha=0.3, label='Student (raw)')
    if window > 1:
        ma_stu = [sum(student_accs[max(0,i-window):i+1])/min(i+1, window) for i in range(len(student_accs))]
        ax2.plot(steps, ma_stu, 'g-', linewidth=2, label=f'Student MA-{window}')
    if teacher_steps:
        ax2.scatter(teacher_steps, teacher_vals, c='red', s=20, label='Teacher', zorder=5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Student vs Teacher Accuracy')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Answer Found Rate
    ax3 = axes[0, 2]
    ax3.plot(steps, ans_rates, 'orange', alpha=0.3, label='Raw')
    if window > 1:
        ma_ans = [sum(ans_rates[max(0,i-window):i+1])/min(i+1, window) for i in range(len(ans_rates))]
        ax3.plot(steps, ma_ans, 'orange', linewidth=2, label=f'MA-{window}')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Answer Found Rate')
    ax3.set_title('Answer Found Rate')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Truncation Rate
    ax4 = axes[1, 0]
    ax4.plot(steps, trunc_rates, 'r-', alpha=0.5)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Truncation Rate')
    ax4.set_title('Truncation Rate')
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    
    # 5. Average Tokens
    ax5 = axes[1, 1]
    ax5.plot(steps, avg_tokens, 'purple', alpha=0.5)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Avg New Tokens')
    ax5.set_title('Average Generated Tokens')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Stats
    ax6 = axes[1, 2]
    ax6.axis('off')
    recent_n = min(50, len(logs))
    recent_logs = logs[-recent_n:]
    
    # 计算教师平均准确率
    teacher_recent = [l['teacher_acc'] for l in recent_logs if l.get('teacher_acc') is not None]
    avg_teacher = sum(teacher_recent)/len(teacher_recent) if teacher_recent else 0
    avg_student = sum(l.get('student_acc', l.get('accuracy', 0)) for l in recent_logs)/recent_n
    
    stats_text = f"""
    Training Summary
    
    Total Steps: {steps[-1]}
    
    Last {recent_n} steps:
    ─────────────────
    Avg Loss: {sum(l['loss'] for l in recent_logs)/recent_n:.4f}
    Student Acc: {avg_student:.2%}
    Teacher Acc: {avg_teacher:.2%}
    Avg Answer Rate: {sum(l.get('answer_found_rate',0) for l in recent_logs)/recent_n:.2%}
    Avg Truncation: {sum(l.get('truncation_rate',0) for l in recent_logs)/recent_n:.2%}
    Avg Tokens: {sum(l.get('avg_new_tokens',0) for l in recent_logs)/recent_n:.1f}
    """
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    if show:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="训练曲线可视化")
    parser.add_argument("--live", action="store_true", help="实时更新模式")
    parser.add_argument("--refresh", type=int, default=30, help="刷新间隔(秒)")
    parser.add_argument("--save", action="store_true", help="保存图片")
    parser.add_argument("--log-dir", type=str, default="runs/qwen2.5_gsm8k_distill")
    args = parser.parse_args()
    
    log_path = os.path.join(args.log_dir, "training_log.jsonl")
    save_path = os.path.join(args.log_dir, "training_curves.png") if args.save else None
    
    if args.live:
        print("Live mode, Ctrl+C to exit")
        plt.ion()
        try:
            while True:
                logs = load_logs(log_path)
                if logs:
                    plot_training_curves(logs, save_path=save_path, show=True)
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\nStopped")
    else:
        logs = load_logs(log_path)
        if logs:
            plot_training_curves(logs, save_path=save_path, show=True)
        else:
            print(f"Log not found: {log_path}")

if __name__ == "__main__":
    main()
