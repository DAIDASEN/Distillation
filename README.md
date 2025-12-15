# Math Reasoning Distillation Project

这是一个基于 On-Policy Distillation 的数学推理训练项目。目标是利用较大的 Teacher 模型 (Qwen2.5-1.5B) 指导较小的 Student 模型 (Qwen2.5-0.5B) 学习解决 GSM8K 数学问题。

## 📂 文件说明

### 核心代码
*   **`train_math_on_policy_distill.py`**: **主训练脚本**。包含了训练循环、生成采样 (Rollout)、Reward 计算、以及基于 KL 散度的蒸馏损失计算。
*   **`utils.py`**: 工具函数库。包含显存优化的 `compute_log_probs` (支持 Batch Chunking 防止 OOM) 和随机种子设置。
*   **`reward_math.py`**: 答案提取与评分逻辑。支持从模型输出中提取 `####` 或 `\boxed{}` 格式的答案并与标准答案比对。
*   **`data_loader.py`**: 负责加载和处理 GSM8K 数据集。

### 运行与监控
*   **`run_nohup.sh`**: **启动脚本**。用于在后台运行训练任务，配置了环境变量和启动参数。
*   **`live_monitor.py`**: **实时监控面板**。在终端中显示当前的 Loss、准确率、进度条以及最新的生成样本。
*   **`plot_monitor.py`**: **绘图工具**。读取日志文件并生成 Loss 和 Accuracy 的变化曲线图。

### 配置文件
*   **`requirements.txt`**: 项目依赖库列表。

---

## 🚀 如何使用

### 1. 启动训练
使用提供的 Shell 脚本在后台启动训练：
```bash
bash run_nohup.sh
```
*   日志会输出到 `nohup.out`。
*   默认配置适配 A100 80GB (Batch Size=4, K_Samples=16)。

### 2. 实时监控
在训练过程中，打开一个新的终端窗口，运行以下命令查看实时状态：
```bash
python live_monitor.py --refresh 5 --show-samples
```

### 3. 绘制曲线
生成训练过程的可视化图表（保存为图片）：
```bash
python plot_monitor.py --save
```

### 4. 终止与恢复
*   **终止训练**: 使用 `kill <PID>` 命令。程序会捕获信号并自动保存检查点（例如 `ckpt_interrupted_40`）。
*   **恢复训练**: 修改 `run_nohup.sh`，添加或更新 `--resume_from_checkpoint` 参数：
    ```bash
    --resume_from_checkpoint runs/qwen2.5_gsm8k_distill/ckpt_interrupted_40
    ```

---

## 💾 输出结果

所有训练产物都存储在 `runs/qwen2.5_gsm8k_distill/` 目录下：

| 文件/目录 | 说明 |
| :--- | :--- |
| **`training_log.jsonl`** | 详细的训练日志，包含每一步的 Loss, Accuracy, Token 长度等指标。 |
| **`samples.jsonl`** | Student 模型在训练过程中生成的样本（包含问题、预测、是否正确）。 |
| **`teacher_samples.jsonl`** | Teacher 模型的评估样本。 |
| **`ckpt_*/`** | 模型权重检查点（每 200 步或中断时保存）。 |
| **`training_curves.png`** | 由 `plot_monitor.py` 生成的训练曲线图。 |
| **`nohup.out`** | 程序的标准输出日志（在根目录下）。 |

## ⚙️ 显存优化说明
本项目针对显存进行了深度优化，以支持在 A100 上进行长序列 (4096 tokens) 训练：
1.  **Gradient Accumulation**: 通过切分 Mini-Batch 减少显存占用。
2.  **Logits Chunking**: 在 `utils.py` 中实现了 Logits 的分块计算，解决了大词表导致的 OOM 问题。
