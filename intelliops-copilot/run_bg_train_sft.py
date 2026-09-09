import os
import sys

os.makedirs("experiments/logs", exist_ok=True)
log_file_path = os.path.join("experiments", "logs", "bg_train_sft.log")
log_file = open(log_file_path, "a", encoding="utf-8", buffering=1)

sys.stdout = log_file
sys.stderr = log_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from sft_train import train_sft

if __name__ == "__main__":
    print(f"\n--- Launching SFT Fine-Tuning Run (max_iters=1000, lr=3e-5) ---", flush=True)
    train_sft(max_iters=1000, learning_rate=3e-5)
