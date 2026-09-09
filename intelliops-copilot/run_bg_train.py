import os
import sys

# Ensure logs directory exists and redirect standard output/error to log file
os.makedirs("experiments/logs", exist_ok=True)
log_file_path = os.path.join("experiments", "logs", "bg_train.log")
log_file = open(log_file_path, "a", encoding="utf-8", buffering=1)

sys.stdout = log_file
sys.stderr = log_file

from minigpt_config import TrainConfig
from train import train

if __name__ == "__main__":
    print(f"\n--- Launching detached training run ---", flush=True)
    cfg = TrainConfig(max_iters=12000)
    train(train_cfg=cfg, resume_path="experiments/checkpoints/best.pt")
