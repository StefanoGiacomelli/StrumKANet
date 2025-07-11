import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ CONFIG ------------------------
CSV_PATH = "./experiments/results/training_results.csv"
OUTPUT_DIR = "./experiments/results/training_figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ LOAD CSV ------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["test_status"] == True] if "test_status" in df.columns else df
df = df[["experiment_id", "test_loss", "test_metric", "stopped_epoch"]]

# ------------------------ PARSE ID ------------------------
def parse_exp_id(exp_id):
    tokens = exp_id.split("_")
    parsed = {"id": exp_id}

    for i in range(len(tokens)):
        if tokens[i] == "chs16":
            parsed["channels"] = 16
        elif tokens[i] == "chs32":
            parsed["channels"] = 32
        elif tokens[i].startswith("ks"):
            parsed["kernel"] = tokens[i + 1]
        elif tokens[i] == "att":
            att = tokens[i + 1]
            parsed["attention"] = "CBAM" if att == "convattn" else (att if att == "None" else att.upper())
        elif tokens[i] == "smooth":
            parsed["smoothing"] = tokens[i + 1]
        elif i == len(tokens) - 1:
            feature = tokens[i].upper()
            if feature == "RATIO":
                parsed["feature"] = "TEMPOGRAM_RATIO"
            elif feature == "STRENGTH":
                parsed["feature"] = "ONSET_STRENGTH"
            else:
                parsed["feature"] = feature

    parsed["encoder"] = "2D" if "[" in parsed.get("kernel", "") else "1D"
    return parsed

parsed = [parse_exp_id(eid) for eid in df["experiment_id"]]
df_meta = pd.DataFrame(parsed)
df_full = pd.concat([df, df_meta], axis=1)

# Convert accuracy to percentage
df_full["accuracy_pct"] = df_full["test_metric"] * 100.0

# Update matplotlib defaults
plt.rcParams.update({
    "font.size": 12,
    "boxplot.medianprops.linewidth": 2.0,
})

# ------------------------ PLOT: ACC vs LOSS + STOPPED EPOCH ------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

attention_types = df_full["attention"].unique()
cmap = plt.get_cmap("tab10")
color_map = {attn: cmap(i) for i, attn in enumerate(attention_types)}
legend_labels = set()

# Scatter: Accuracy vs Loss
for idx, row in df_full.iterrows():
    color = color_map.get(row["attention"], "gray")
    edge_color = "black" if row["encoder"] == "1D" else "red"
    label = row["attention"] if row["attention"] not in legend_labels else None
    axes[0].scatter(row["test_loss"], row["accuracy_pct"],
                    color=color, edgecolors=edge_color, linewidths=1.5,
                    s=140, alpha=0.9, label=label)
    legend_labels.add(row["attention"])

axes[0].set_xlabel("Test Loss")
axes[0].set_ylabel("Test Accuracy (in %)")
axes[0].set_title("Accuracy vs Loss\n(Black edge = 1D, Red edge = 2D)")
axes[0].legend(title="Attention", fontsize=10, title_fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.3)

# Scatter: Stopped Epoch vs Accuracy
for idx, row in df_full.iterrows():
    color = color_map.get(row["attention"], "gray")
    edge_color = "black" if row["encoder"] == "1D" else "red"
    axes[1].scatter(row["stopped_epoch"], row["accuracy_pct"],
                    color=color, edgecolors=edge_color, linewidths=1.5,
                    s=140, alpha=0.9)

axes[1].set_xlabel("Stopped Epoch")
axes[1].set_ylabel("Test Accuracy (in %)")
axes[1].set_title("Accuracy vs Stopping Epoch")
axes[1].grid(True, linestyle="--", alpha=0.3)

# Scatter: Stopped Epoch vs Loss
for idx, row in df_full.iterrows():
    color = color_map.get(row["attention"], "gray")
    edge_color = "black" if row["encoder"] == "1D" else "red"
    axes[2].scatter(row["stopped_epoch"], row["test_loss"],
                    color=color, edgecolors=edge_color, linewidths=1.5,
                    s=140, alpha=0.9)

axes[2].set_xlabel("Stopped Epoch")
axes[2].set_ylabel("Test Loss")
axes[2].set_title("Loss vs Stopping Epoch")
axes[2].grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_loss_vs_stopping_epoch.png"), dpi=600)
plt.close()

# ------------------------ Summary ------------------------
best_acc_row = df_full.sort_values("accuracy_pct", ascending=False).iloc[0]
print("\nBEST MODEL (by Test Accuracy):")
print(f"- Experiment_ID  : {best_acc_row['experiment_id']}")
print(f"- Test Accuracy  : {best_acc_row['accuracy_pct']:.2f}%")
print(f"- Test Loss      : {best_acc_row['test_loss']:.4f}")
print(f"- Stopped Epoch  : {best_acc_row['stopped_epoch']}")
print(f"- Attention      : {best_acc_row['attention']}")
print(f"- Feature        : {best_acc_row['feature']}")
print(f"- Encoder        : {best_acc_row['encoder']}")
print(f"- Kernel Size    : {best_acc_row['kernel']}")

best_loss_row = df_full.sort_values("test_loss", ascending=True).iloc[0]
print("\nBEST MODEL (by Loss Minimization):")
print(f"- Experiment_ID  : {best_loss_row['experiment_id']}")
print(f"- Test Accuracy  : {best_loss_row['accuracy_pct']:.2f}%")
print(f"- Test Loss      : {best_loss_row['test_loss']:.4f}")
print(f"- Stopped Epoch  : {best_loss_row['stopped_epoch']}")
print(f"- Attention      : {best_loss_row['attention']}")
print(f"- Feature        : {best_loss_row['feature']}")
print(f"- Encoder        : {best_loss_row['encoder']}")
print(f"- Kernel Size    : {best_loss_row['kernel']}")