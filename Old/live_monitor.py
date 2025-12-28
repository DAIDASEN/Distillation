#!/usr/bin/env python
"""
实时训练监控工具
用法: python live_monitor.py [--refresh 5] [--show-samples]
"""

import json
import os
import time
import argparse
from datetime import datetime

def load_logs(log_path):
    """加载训练日志，并去重（保留最新的step）"""
    if not os.path.exists(log_path):
        return []
    
    logs_dict = {}
    with open(log_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                step = entry.get('step')
                if step is not None:
                    logs_dict[step] = entry
            except:
                pass
    
    # 按步数排序
    sorted_steps = sorted(logs_dict.keys())
    return [logs_dict[s] for s in sorted_steps]

def load_samples(samples_path, last_n=5):
    """加载最近的样本"""
    if not os.path.exists(samples_path):
        return []
    samples = []
    with open(samples_path, "r") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                pass
    return samples[-last_n:]

def print_stats(logs, window=20):
    """打印统计信息"""
    if not logs:
        print("⏳ 等待训练数据...")
        return
    
    recent = logs[-window:] if len(logs) >= window else logs
    
    # 计算移动平均
    avg_loss = sum(l['loss'] for l in recent) / len(recent)
    avg_acc = sum(l.get('student_acc', l.get('accuracy', 0)) for l in recent) / len(recent)
    avg_ans_rate = sum(l['answer_found_rate'] for l in recent) / len(recent)
    avg_trunc = sum(l['truncation_rate'] for l in recent) / len(recent)
    avg_len = sum(l['avg_new_tokens'] for l in recent) / len(recent)
    
    latest = logs[-1]
    latest_acc = latest.get('student_acc', latest.get('accuracy', 0))
    
    print("\n" + "="*70)
    print(f"📊 训练监控 - {datetime.now().strftime('%H:%M:%S')} | Step: {latest['step']}")
    print("="*70)
    
    # 当前值 vs 移动平均
    print(f"\n{'指标':<20} {'当前值':<15} {'近{window}步平均':<15}")
    print("-"*50)
    print(f"{'Loss':<20} {latest['loss']:<15.4f} {avg_loss:<15.4f}")
    print(f"{'Accuracy':<20} {latest_acc:<15.2%} {avg_acc:<15.2%}")
    print(f"{'Answer Found Rate':<20} {latest['answer_found_rate']:<15.2%} {avg_ans_rate:<15.2%}")
    print(f"{'Truncation Rate':<20} {latest['truncation_rate']:<15.2%} {avg_trunc:<15.2%}")
    print(f"{'Avg Tokens':<20} {latest['avg_new_tokens']:<15.1f} {avg_len:<15.1f}")
    
    # 进度条
    total_steps = 1000
    progress = latest['step'] / total_steps
    bar_len = 40
    filled = int(bar_len * progress)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n进度: [{bar}] {progress:.1%} ({latest['step']}/{total_steps})")
    
    # 趋势分析
    if len(logs) >= 20:
        old_acc = sum(l.get('student_acc', l.get('accuracy', 0)) for l in logs[-40:-20]) / 20 if len(logs) >= 40 else avg_acc
        trend = "📈 上升" if avg_acc > old_acc else ("📉 下降" if avg_acc < old_acc else "➡️ 稳定")
        print(f"准确率趋势: {trend}")

def print_samples(samples):
    """打印最近的样本"""
    if not samples:
        return
    
    print("\n" + "="*70)
    print("📝 最近生成样本")
    print("="*70)
    
    for i, s in enumerate(samples[-3:]):  # 只显示最近3个
        status = "✅" if s['correct'] else "❌"
        print(f"\n--- Sample {i+1} (Step {s['step']}) {status} ---")
        print(f"问题: {s['question'][:200]}..." if len(s['question']) > 200 else f"问题: {s['question']}")
        print(f"\n预测答案 (截取):")
        # 只显示最后500字符（包含答案部分）
        pred_text = s['prediction']
        if len(pred_text) > 600:
            pred_text = "..." + pred_text[-600:]
        print(pred_text)
        print(f"\n参考答案: {s['reference'][-100:]}" if len(s['reference']) > 100 else f"参考答案: {s['reference']}")
        print("-"*50)

def main():
    parser = argparse.ArgumentParser(description="实时训练监控")
    parser.add_argument("--refresh", type=int, default=10, help="刷新间隔(秒)")
    parser.add_argument("--show-samples", action="store_true", help="显示生成样本")
    parser.add_argument("--log-dir", type=str, default="runs/qwen2.5_gsm8k_distill", help="日志目录")
    parser.add_argument("--once", action="store_true", help="只运行一次（不循环）")
    args = parser.parse_args()
    
    log_path = os.path.join(args.log_dir, "training_log.jsonl")
    samples_path = os.path.join(args.log_dir, "samples.jsonl")
    
    print("🚀 启动训练监控...")
    print(f"日志路径: {log_path}")
    print(f"刷新间隔: {args.refresh}秒")
    print("按 Ctrl+C 退出\n")
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            logs = load_logs(log_path)
            print_stats(logs)
            
            if args.show_samples:
                samples = load_samples(samples_path)
                print_samples(samples)
            
            if args.once:
                break
                
            time.sleep(args.refresh)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")

if __name__ == "__main__":
    main()
