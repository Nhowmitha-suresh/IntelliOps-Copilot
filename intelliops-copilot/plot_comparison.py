import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("experiments/plots", exist_ok=True)

base_df = pd.read_csv("experiments/logs/training_log.csv")
var_df = pd.read_csv("experiments/logs/training_log_variant.csv")

plt.figure(figsize=(10, 6))

plt.plot(base_df["step"], base_df["train_loss"], label="Baseline Train Loss (n_embd=128)", color="#1f77b4", linestyle="--", alpha=0.6)
plt.plot(base_df["step"], base_df["val_loss"], label="Baseline Val Loss (n_embd=128)", color="#1f77b4", linewidth=2)

plt.plot(var_df["step"], var_df["train_loss"], label="Variant Train Loss (n_embd=256)", color="#d62728", linestyle="--", alpha=0.6)
plt.plot(var_df["step"], var_df["val_loss"], label="Variant Val Loss (n_embd=256)", color="#d62728", linewidth=2)

plt.xlabel("Step", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("MiniGPT Architecture Comparison: Baseline (n_embd=128) vs Variant (n_embd=256)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)

plot_path = "experiments/plots/comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Comparison plot saved to {plot_path}")
