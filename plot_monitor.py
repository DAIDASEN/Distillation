
import json
import matplotlib.pyplot as plt
import os

def main():
    log_file_path = "runs/qwen3_distill/training_log.jsonl"
    output_image_path = "training_curves.png"

    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}")
        return

    steps = []
    losses = []
    accuracies = []

    with open(log_file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                steps.append(data["step"])
                losses.append(data["loss"])
                accuracies.append(data["accuracy"])
            except json.JSONDecodeError:
                continue

    if not steps:
        print("No data found in log file.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(steps, losses, label="Training Loss", color="blue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(steps, accuracies, label="Batch Accuracy", color="green")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Batch Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Training curves saved to {output_image_path}")

if __name__ == "__main__":
    main()
