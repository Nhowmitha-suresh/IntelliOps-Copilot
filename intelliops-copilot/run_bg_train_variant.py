import os
import sys

# Ensure logs directory exists and redirect standard output/error to log file
os.makedirs("experiments/logs", exist_ok=True)
log_file_path = os.path.join("experiments", "logs", "bg_train_variant.log")
log_file = open(log_file_path, "a", encoding="utf-8", buffering=1)

sys.stdout = log_file
sys.stderr = log_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from minigpt_config import ModelConfigVariant, TrainConfig
from train import train

if __name__ == "__main__":
    print(f"\n--- Launching variant training run (n_embd=256) ---", flush=True)
    m_cfg = ModelConfigVariant()
    t_cfg = TrainConfig(max_iters=12000)
    train(
        model_cfg=m_cfg,
        train_cfg=t_cfg,
        checkpoint_dir="experiments/checkpoints_variant",
        log_filename="training_log_variant.csv",
        plots_dir="experiments/plots_variant",
    )
