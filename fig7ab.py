import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = r"D:\bio_project\fig\fig7\plotdata"

FIG7A_PATH = os.path.join(
    BASE_DIR,
    "Fig7A_Aset_ChickenThr_0.30_BinaryMetrics_LongTable.csv"
)

FIG7B_PATH = os.path.join(
    BASE_DIR,
    "Fig7_TunedBase_LGBMMeta_Integrated_ThreeClass_Metrics_BsetCV_vs_Aset.csv"
)

OUT_DIR = os.path.join(BASE_DIR, "Radar_Fig7AB_Thr030")
os.makedirs(OUT_DIR, exist_ok=True)

HOST_COLORS = {
    "human": "#E41A1C",
    "pig": "#377EB8",
    "chicken": "#FFD700",
}

DATASET_COLORS = {
    "B-set 5-fold CV": "#4D4D4D",
    "Independent A-set": "#984EA3",
}

FONT_SIZE_METRIC = 18
FONT_SIZE_RADIAL = 15
FONT_SIZE_TITLE = 22
FONT_SIZE_LEGEND = 17
FONT_SIZE_LEGEND_TITLE = 18

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE_METRIC,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "figure.titleweight": "normal",
    "legend.title_fontsize": FONT_SIZE_LEGEND_TITLE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def normalize_dataset_names(df):
    df = df.copy()

    if "Dataset" in df.columns:
        df["Dataset"] = df["Dataset"].astype(str).str.strip()

        df["Dataset"] = df["Dataset"].replace({
            "Filtered A-set": "Independent A-set",
            "Filtered_A-set": "Independent A-set",
            "Filtered_A_set": "Independent A-set",
            "A-set": "Independent A-set",
            "A set": "Independent A-set",
            "Independent A set": "Independent A-set",
            "Independent_A-set": "Independent A-set",
            "Independent_A_set": "Independent A-set",
            "B set 5 fold CV": "B-set 5-fold CV",
            "B-set CV": "B-set 5-fold CV",
            "Bset 5-fold CV": "B-set 5-fold CV",
            "B-set 5fold CV": "B-set 5-fold CV",
            "B-set_5-fold_CV": "B-set 5-fold CV",
        })

    return df


def check_nonempty(df, label):
    if len(df) == 0:
        raise ValueError(f"❌ {label} 数据为空。请检查 Dataset / Metric / 文件路径。")


def close_values(values):
    return values + values[:1]


def setup_radar_axis(ax, metric_labels, title):
    n = len(metric_labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(
        metric_labels,
        fontsize=FONT_SIZE_METRIC,
        fontweight="normal"
    )

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        fontsize=FONT_SIZE_RADIAL,
        fontweight="normal"
    )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)

    ax.spines["polar"].set_linewidth(1.4)
    ax.spines["polar"].set_color("#333333")

    ax.set_title(
        title,
        fontsize=FONT_SIZE_TITLE,
        fontweight="normal",
        pad=34
    )

    return angles_closed


def save_figure(fig, out_prefix):
    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"

    fig.savefig(png_path, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=800, bbox_inches="tight", facecolor="white")

    print(f"✅ 保存: {png_path}")
    print(f"✅ 保存: {pdf_path}")


def set_legend_normal_weight(legend):
    if legend is None:
        return

    legend.get_title().set_fontweight("normal")
    legend.get_title().set_fontsize(FONT_SIZE_LEGEND_TITLE)

    for text_obj in legend.get_texts():
        text_obj.set_fontweight("normal")
        text_obj.set_fontsize(FONT_SIZE_LEGEND)


def save_legend_only(handles, labels, title, out_prefix, ncol=1):
    fig_leg = plt.figure(figsize=(4.2, 2.8))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")

    legend = ax_leg.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=ncol,
        fontsize=FONT_SIZE_LEGEND,
        title=title,
        title_fontsize=FONT_SIZE_LEGEND_TITLE,
        handlelength=1.0,
        handleheight=1.0,
        borderpad=0.6,
        labelspacing=0.8,
        handletextpad=0.6,
        columnspacing=1.0
    )

    set_legend_normal_weight(legend)

    save_figure(fig_leg, out_prefix)
    plt.close(fig_leg)


def plot_fig7a_radar(fig7a_path):
    df = pd.read_csv(fig7a_path)
    df = normalize_dataset_names(df)

    print("\nFig7A columns:")
    print(df.columns.tolist())

    if "Dataset" in df.columns:
        print("\nFig7A Dataset values:")
        print(df["Dataset"].value_counts().to_string())

        df = df[
            df["Dataset"].astype(str).str.strip() == "Independent A-set"
        ].copy()

    check_nonempty(df, "Fig7A Independent A-set")

    required_cols = {"Target_Host", "Metric", "Value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fig7A 文件缺少这些列: {missing}")

    metric_order = ["AUC", "PR", "Accuracy", "Macro F1", "MCC", "Recall"]
    host_order = ["human", "pig", "chicken"]

    df["Target_Host"] = df["Target_Host"].astype(str).str.lower().str.strip()
    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    fig = plt.figure(figsize=(8.5, 8.2))
    ax = plt.subplot(111, polar=True)

    angles = setup_radar_axis(
        ax,
        metric_labels=metric_order,
        title="Fig. 7A  One-vs-rest binary performance"
    )

    for host in host_order:
        sub = df[df["Target_Host"] == host].copy()

        values = []
        for metric in metric_order:
            v = sub.loc[sub["Metric"] == metric, "Value"]
            values.append(float(v.iloc[0]) if len(v) > 0 else np.nan)

        print(f"Fig7A {host} values:", values)

        values_closed = close_values(values)

        ax.plot(
            angles,
            values_closed,
            linewidth=2.6,
            marker="o",
            markersize=6,
            color=HOST_COLORS[host],
            label=host.capitalize()
        )

        ax.fill(
            angles,
            values_closed,
            color=HOST_COLORS[host],
            alpha=0.10
        )

    leg = ax.legend(
        title="Target host",
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND_TITLE
    )
    set_legend_normal_weight(leg)

    plt.tight_layout()

    save_figure(
        fig,
        os.path.join(OUT_DIR, "Fig7A_OneVsRest_Binary_Radar")
    )

    plt.close(fig)


def plot_fig7b_radar(fig7b_path):
    df = pd.read_csv(fig7b_path)
    df = normalize_dataset_names(df)

    print("\nFig7B columns:")
    print(df.columns.tolist())

    required_cols = {"Dataset", "Metric", "Value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fig7B 文件缺少这些列: {missing}")

    print("\nFig7B Dataset values:")
    print(df["Dataset"].value_counts().to_string())

    metric_label_map = {
        "Accuracy": "Accuracy",
        "Macro F1": "Macro F1",
        "Weighted F1": "Weighted F1",
        "Macro Precision": "Macro Precision",
        "Macro Recall / Balanced Acc": "Recall",
        "Recall": "Recall",
        "MCC": "MCC",
    }

    metric_order_raw = [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall / Balanced Acc",
        "Recall",
        "MCC",
    ]

    metric_order_display = [
        "Weighted F1",
        "Macro Precision",
        "Accuracy",
        "Macro F1",
        "MCC",
        "Recall",
    ]

    dataset_order = ["B-set 5-fold CV", "Independent A-set"]

    df["Dataset"] = df["Dataset"].astype(str).str.strip()
    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Metric_Display"] = df["Metric"].map(metric_label_map)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    df = df[df["Metric"].isin(metric_order_raw)].copy()

    fig = plt.figure(figsize=(8.5, 8.2))
    ax = plt.subplot(111, polar=True)

    angles = setup_radar_axis(
        ax,
        metric_labels=metric_order_display,
        title="Fig. 7B  Integrated three-class performance"
    )

    for dataset in dataset_order:
        sub = df[df["Dataset"] == dataset].copy()

        if len(sub) == 0:
            print(f"⚠️ Fig7B 没有找到 dataset = {dataset} 的数据，跳过这条线。")
            continue

        values = []
        for metric in metric_order_display:
            v = sub.loc[sub["Metric_Display"] == metric, "Value"]
            values.append(float(v.iloc[0]) if len(v) > 0 else np.nan)

        print(f"Fig7B {dataset} values:", values)

        values_closed = close_values(values)

        ax.plot(
            angles,
            values_closed,
            linewidth=2.6,
            marker="o",
            markersize=6,
            color=DATASET_COLORS[dataset],
            label=dataset
        )

        ax.fill(
            angles,
            values_closed,
            color=DATASET_COLORS[dataset],
            alpha=0.10
        )

    leg = ax.legend(
        title="Dataset",
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND_TITLE
    )
    set_legend_normal_weight(leg)

    plt.tight_layout()

    save_figure(
        fig,
        os.path.join(OUT_DIR, "Fig7B_ThreeClass_Integrated_Radar")
    )

    plt.close(fig)


def plot_combined_radar(fig7a_path, fig7b_path):
    df_a = pd.read_csv(fig7a_path)
    df_b = pd.read_csv(fig7b_path)

    df_a = normalize_dataset_names(df_a)
    df_b = normalize_dataset_names(df_b)

    if "Dataset" in df_a.columns:
        df_a = df_a[
            df_a["Dataset"].astype(str).str.strip() == "Independent A-set"
        ].copy()

    check_nonempty(df_a, "Combined Fig7A Independent A-set")

    metric_order_a = ["AUC", "PR", "Accuracy", "Macro F1", "MCC", "Recall"]
    host_order = ["human", "pig", "chicken"]

    metric_label_map_b = {
        "Accuracy": "Accuracy",
        "Macro F1": "Macro F1",
        "Weighted F1": "Weighted F1",
        "Macro Precision": "Macro Precision",
        "Macro Recall / Balanced Acc": "Recall",
        "Recall": "Recall",
        "MCC": "MCC",
    }

    metric_order_raw_b = [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall / Balanced Acc",
        "Recall",
        "MCC",
    ]

    metric_order_display_b = [
        "Weighted F1",
        "Macro Precision",
        "Accuracy",
        "Macro F1",
        "MCC",
        "Recall",
    ]

    dataset_order = ["B-set 5-fold CV", "Independent A-set"]

    required_cols_a = {"Target_Host", "Metric", "Value"}
    missing_a = required_cols_a - set(df_a.columns)
    if missing_a:
        raise ValueError(f"Fig7A 文件缺少这些列: {missing_a}")

    required_cols_b = {"Dataset", "Metric", "Value"}
    missing_b = required_cols_b - set(df_b.columns)
    if missing_b:
        raise ValueError(f"Fig7B 文件缺少这些列: {missing_b}")

    df_a["Target_Host"] = df_a["Target_Host"].astype(str).str.lower().str.strip()
    df_a["Metric"] = df_a["Metric"].astype(str).str.strip()
    df_a["Value"] = pd.to_numeric(df_a["Value"], errors="coerce")

    df_b["Dataset"] = df_b["Dataset"].astype(str).str.strip()
    df_b["Metric"] = df_b["Metric"].astype(str).str.strip()
    df_b["Metric_Display"] = df_b["Metric"].map(metric_label_map_b)
    df_b["Value"] = pd.to_numeric(df_b["Value"], errors="coerce")
    df_b = df_b[df_b["Metric"].isin(metric_order_raw_b)].copy()

    fig, axes = plt.subplots(
        1,
        2,
        subplot_kw=dict(polar=True),
        figsize=(16.5, 8.0)
    )

    ax = axes[0]
    angles_a = setup_radar_axis(
        ax,
        metric_labels=metric_order_a,
        title=""
    )

    for host in host_order:
        sub = df_a[df_a["Target_Host"] == host].copy()

        values = []
        for metric in metric_order_a:
            v = sub.loc[sub["Metric"] == metric, "Value"]
            values.append(float(v.iloc[0]) if len(v) > 0 else np.nan)

        print(f"Combined Fig7A {host} values:", values)

        values_closed = close_values(values)

        ax.plot(
            angles_a,
            values_closed,
            linewidth=2.5,
            marker="o",
            markersize=5.8,
            color=HOST_COLORS[host],
            label=host.capitalize()
        )

        ax.fill(
            angles_a,
            values_closed,
            color=HOST_COLORS[host],
            alpha=0.10
        )

    ax = axes[1]
    angles_b = setup_radar_axis(
        ax,
        metric_labels=metric_order_display_b,
        title=""
    )

    for dataset in dataset_order:
        sub = df_b[df_b["Dataset"] == dataset].copy()

        if len(sub) == 0:
            print(f"⚠️ Combined Fig7B 没有找到 dataset = {dataset} 的数据，跳过这条线。")
            continue

        values = []
        for metric in metric_order_display_b:
            v = sub.loc[sub["Metric_Display"] == metric, "Value"]
            values.append(float(v.iloc[0]) if len(v) > 0 else np.nan)

        print(f"Combined Fig7B {dataset} values:", values)

        values_closed = close_values(values)

        ax.plot(
            angles_b,
            values_closed,
            linewidth=2.5,
            marker="o",
            markersize=5.8,
            color=DATASET_COLORS[dataset],
            label=dataset
        )

        ax.fill(
            angles_b,
            values_closed,
            color=DATASET_COLORS[dataset],
            alpha=0.10
        )

    plt.tight_layout(rect=[0, 0.16, 1, 1])

    pos_left = axes[0].get_position()
    pos_right = axes[1].get_position()

    fig.text(
        (pos_left.x0 + pos_left.x1) / 2,
        0.10,
        "One-vs-rest binary performance",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_TITLE,
        fontweight="normal"
    )

    fig.text(
        (pos_right.x0 + pos_right.x1) / 2,
        0.10,
        "Integrated three-class performance",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_TITLE,
        fontweight="normal"
    )

    save_figure(
        fig,
        os.path.join(OUT_DIR, "Fig7AB_Combined_Radar")
    )

    plt.close(fig)


def save_two_legends():
    handles_a = [
        Patch(facecolor=HOST_COLORS["human"], edgecolor="none"),
        Patch(facecolor=HOST_COLORS["pig"], edgecolor="none"),
        Patch(facecolor=HOST_COLORS["chicken"], edgecolor="none"),
    ]

    labels_a = ["Human", "Pig", "Chicken"]

    save_legend_only(
        handles=handles_a,
        labels=labels_a,
        title="Target host",
        out_prefix=os.path.join(OUT_DIR, "Fig7A_TargetHost_Legend"),
        ncol=1
    )

    handles_b = [
        Patch(facecolor=DATASET_COLORS["B-set 5-fold CV"], edgecolor="none"),
        Patch(facecolor=DATASET_COLORS["Independent A-set"], edgecolor="none"),
    ]

    labels_b = ["B-set 5-fold CV", "Independent A-set"]

    save_legend_only(
        handles=handles_b,
        labels=labels_b,
        title="Dataset",
        out_prefix=os.path.join(OUT_DIR, "Fig7B_Dataset_Legend"),
        ncol=1
    )


def main():
    print("正在检查文件...")

    if not os.path.exists(FIG7A_PATH):
        raise FileNotFoundError(f"找不到 Fig7A 文件:\n{FIG7A_PATH}")

    if not os.path.exists(FIG7B_PATH):
        raise FileNotFoundError(f"找不到 Fig7B 文件:\n{FIG7B_PATH}")

    print("✅ 找到 Fig7A 文件:")
    print(FIG7A_PATH)
    print("✅ 找到 Fig7B 文件:")
    print(FIG7B_PATH)

    plot_fig7a_radar(FIG7A_PATH)
    plot_fig7b_radar(FIG7B_PATH)
    plot_combined_radar(FIG7A_PATH, FIG7B_PATH)
    save_two_legends()

    print("\n🎉 全部完成！")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()