import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
CSV_PATH = "./experiments/results/hp_search_results.csv"
OUTPUT_DIR = "./experiments/results/hp_nas_seach_figures/"
TOP_K = 10
DPI = 600
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH)
df = df[df["test_status"] == True]
df["accuracy_pct"] = df["test_metric"] * 100.0

# === PARSE experiment_id ===
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

df_meta = pd.DataFrame([parse_exp_id(eid) for eid in df["experiment_id"]])
df_full = pd.concat([df, df_meta], axis=1)

# === Top-10 by Accuracy ===
top_acc = df.sort_values("test_metric", ascending=False).head(TOP_K)
print("Top-10 models by accuracy:")
print(top_acc[["experiment_id", "test_metric", "test_loss"]])
print("---------------------------------------------------------------------------------------------------------------------------")

# === Top-10 Accuracy Plot ===
plt.figure(figsize=(10, 6))
bars = plt.barh(top_acc["experiment_id"], top_acc["test_metric"],
                color="black", edgecolor="black")
plt.xlabel("Accuracy", fontsize=12)
plt.title(f"Top-{TOP_K} Models by Accuracy", fontsize=14)
plt.xscale("log")
plt.gca().invert_yaxis()
plt.grid(True, linestyle="--", alpha=0.3)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hp_top_accuracy.png", dpi=DPI)
plt.close()

# === Top-10 by Loss ===
top_loss = df.sort_values("test_loss", ascending=True).head(TOP_K)
print("\nTop-10 models by loss:")
print(top_loss[["experiment_id", "test_metric", "test_loss"]])
print("---------------------------------------------------------------------------------------------------------------------------")

# === Top-10 Loss Plot ===
plt.figure(figsize=(10, 6))
bars = plt.barh(top_loss["experiment_id"], top_loss["test_loss"],
                color="#d3d3d3", edgecolor="black")
plt.xlabel("Loss", fontsize=12)
plt.title(f"Top-{TOP_K} Models by Loss", fontsize=14)
plt.xscale("log")
plt.gca().invert_yaxis()
plt.grid(True, linestyle="--", alpha=0.3)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hp_top_loss.png", dpi=DPI)
plt.close()

# === Accuracy vs Loss Scatter ===
plt.figure(figsize=(10, 7))
cmap = plt.get_cmap("tab10")
color_map = {attn: cmap(i) for i, attn in enumerate(df_full["attention"].unique())}
legend_labels = set()
for _, row in df_full.iterrows():
    color = color_map.get(row["attention"], "gray")
    edge = "black" if row["encoder"] == "1D" else "red"
    label = row["attention"] if row["attention"] not in legend_labels else None
    plt.scatter(row["test_loss"], row["accuracy_pct"],
                color=color, edgecolors=edge, linewidths=1.2,
                s=70, alpha=0.9, label=label)
    legend_labels.add(row["attention"])
plt.xlabel("Test Loss")
plt.ylabel("Test Accuracy (%)")
plt.title("NAS-HPO: Accuracy vs Loss")
plt.legend(title="Attention", fontsize=10, title_fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hp_accuracy_vs_loss.png"), dpi=DPI)
plt.close()

# === Boxplot per Feature (Accuracy + Loss) ===
plt.figure(figsize=(14, 6))
feat_values = sorted(df_full["feature"].unique())
acc_data = [df_full[df_full["feature"] == feat]["accuracy_pct"] for feat in feat_values]
loss_data = [df_full[df_full["feature"] == feat]["test_loss"] for feat in feat_values]
plt.subplot(1, 2, 1)
plt.boxplot(acc_data, labels=feat_values, patch_artist=True)
plt.title("Accuracy (per Feature)")
plt.xlabel("Feature")
plt.ylabel("Test Accuracy (%)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.4)
plt.subplot(1, 2, 2)
plt.boxplot(loss_data, labels=feat_values, patch_artist=True)
plt.title("Loss (per Feature)")
plt.xlabel("Feature")
plt.ylabel("Test Loss")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hp_boxplot_feature_accuracy_loss.png"), dpi=DPI)
plt.close()

# === Boxplot per Attention (Accuracy + Loss) ===
plt.figure(figsize=(14, 6))
attn_values = sorted(df_full["attention"].unique())
acc_data = [df_full[df_full["attention"] == attn]["accuracy_pct"] for attn in attn_values]
loss_data = [df_full[df_full["attention"] == attn]["test_loss"] for attn in attn_values]
plt.subplot(1, 2, 1)
plt.boxplot(acc_data, labels=attn_values, patch_artist=True)
plt.title("Accuracy (per Attention)")
plt.xlabel("Attention")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.subplot(1, 2, 2)
plt.boxplot(loss_data, labels=attn_values, patch_artist=True)
plt.title("Loss (per Attention)")
plt.xlabel("Attention")
plt.ylabel("Test Loss")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hp_boxplot_attention_accuracy_loss.png"), dpi=DPI)
plt.close()

# === 2x2 Subplot on Smoothing ===
plt.figure(figsize=(14, 10))
smoothing_values = sorted(df_full["smoothing"].unique())
acc_data = [df_full[df_full["smoothing"] == val]["accuracy_pct"] for val in smoothing_values]
loss_data = [df_full[df_full["smoothing"] == val]["test_loss"] for val in smoothing_values]
colors = ["lightblue", "lightgreen"]
alphas = [0.8, 0.5]

# Subplot 1: Accuracy Boxplot with Median Legend
plt.subplot(2, 2, 1)
bp = plt.boxplot(acc_data, labels=smoothing_values, patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
medians = [round(data.median(), 2) for data in acc_data]
for i, (x, y) in enumerate(zip(range(1, len(medians)+1), medians)):
    plt.text(x + 0.1, y, f"Median: {y:.2f}%", va="center", fontsize=10)
plt.title("Accuracy Distribution by Smoothing")
plt.xlabel("Smoothing")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.4)

# Subplot 2: Loss Boxplot with Median Legend
plt.subplot(2, 2, 2)
bp = plt.boxplot(loss_data, labels=smoothing_values, patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
medians = [round(data.median(), 4) for data in loss_data]
for i, (x, y) in enumerate(zip(range(1, len(medians)+1), medians)):
    plt.text(x + 0.1, y, f"Median: {y:.4f}", va="center", fontsize=10)
plt.title("Loss Distribution by Smoothing")
plt.xlabel("Smoothing")
plt.ylabel("Test Loss")
plt.grid(True, linestyle="--", alpha=0.4)

# Subplot 3: Histogram of Accuracy by Smoothing
plt.subplot(2, 2, 3)
bins = np.linspace(df_full["accuracy_pct"].min(), df_full["accuracy_pct"].max(), 15)
for i, val in enumerate(smoothing_values):
    subset = df_full[df_full["smoothing"] == val]["accuracy_pct"]
    plt.hist(subset, bins=bins, alpha=alphas[i % len(alphas)], label=f"Smoothing={val}", edgecolor="black", color=colors[i % len(colors)])
plt.title("Accuracy Histogram by Smoothing")
plt.xlabel("Accuracy (%)")
plt.ylabel("Occurrences")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

# Subplot 4: Histogram of Loss by Smoothing
plt.subplot(2, 2, 4)
bins = np.linspace(df_full["test_loss"].min(), df_full["test_loss"].max(), 15)
for i, val in enumerate(smoothing_values):
    subset = df_full[df_full["smoothing"] == val]["test_loss"]
    plt.hist(subset, bins=bins, alpha=alphas[i % len(alphas)], label=f"Smoothing={val}", edgecolor="black", color=colors[i % len(colors)])
plt.title("Loss Histogram by Smoothing")
plt.xlabel("Loss")
plt.ylabel("Occurrences")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hp_smoothing_analysis.png"), dpi=DPI)
plt.close()

# === Smooth_T vs Smooth_F counterparts ===
print("\nSmooth_True vs Smooth_False counterpart results:\n")
def find_counterpart(exp_id):
    if "smooth_True" in exp_id:
        base_id = exp_id.replace("smooth_True", "smooth_False")
        match = df[df["experiment_id"] == base_id]
        return match.iloc[0] if not match.empty else None
    return None

for _, row in pd.concat([top_acc, top_loss]).drop_duplicates().iterrows():
    if "smooth_True" in row["experiment_id"]:
        cp = find_counterpart(row["experiment_id"])
        if cp is not None:
            print(f"Original    : {row['experiment_id']}")
            print(f" - Accuracy : {row['test_metric']:.4f}, Loss: {row['test_loss']:.4f}")
            print(f"Counterpart : {cp['experiment_id']}")
            print(f" - Accuracy : {cp['test_metric']:.4f}, Loss: {cp['test_loss']:.4f}")
            print("---")