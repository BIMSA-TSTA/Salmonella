import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

base_dir = r"D:\bio_project\fig\fig5\2"

out_dir = os.path.join(base_dir, "Fig5E_redraw_3hosts_final_colors")
os.makedirs(out_dir, exist_ok=True)

file_map = {
    "human": "Fig5_E_human_Top1_Gene_Effect_Data.csv",
    "pig": "Fig5_E_pig_Top1_Gene_Effect_Data.csv",
    "chicken": "Fig5_E_chicken_Top1_Gene_Effect_Data.csv"
}


host_color_map = {
    "human": "#E41A1C",
    "pig": "#377EB8",
    "chicken": "#FFD700"
}

absent_color = "#D9D9D9"

def draw_one_host_plot(host_name, csv_path, out_dir):

    if not os.path.exists(csv_path):
        print(f"⚠️ 文件不存在，跳过: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    
    required_cols = [
        "Predicted_Probability",
        "Gene_State_Label",
        "Top1_Gene_Clean"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{host_name} 的 CSV 缺少必要列: {missing_cols}\n"
            f"当前列名为: {list(df.columns)}"
        )

    plot_df = df.copy()

    plot_df["Predicted_Probability"] = pd.to_numeric(
        plot_df["Predicted_Probability"],
        errors="coerce"
    )

    plot_df = plot_df[np.isfinite(plot_df["Predicted_Probability"])].copy()

    plot_df["Gene_State_Label"] = (
        plot_df["Gene_State_Label"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    plot_df = plot_df[
        plot_df["Gene_State_Label"].isin(["Absent", "Present"])
    ].copy()

    plot_df["Gene_State_Label"] = pd.Categorical(
        plot_df["Gene_State_Label"],
        categories=["Absent", "Present"],
        ordered=True
    )

    if plot_df.empty:
        print(f"⚠️ {host_name} 过滤后没有可用数据，跳过。")
        return

    target_gene = str(plot_df["Top1_Gene_Clean"].iloc[0])
    host_title = host_name.capitalize()

    present_color = host_color_map[host_name]

    print("\n分组计数：")
    print(plot_df["Gene_State_Label"].value_counts())

   
    sns.set_theme(style="whitegrid")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 21
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["legend.title_fontsize"] = 12

    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    palette = {
        "Absent": absent_color,
        "Present": present_color
    }

    
    sns.violinplot(
        data=plot_df,
        x="Gene_State_Label",
        y="Predicted_Probability",
        order=["Absent", "Present"],
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.2,
        saturation=1,
        width=0.76,
        ax=ax
    )

    for coll in ax.collections:
        try:
            coll.set_alpha(0.55)
        except Exception:
            pass

   
    sns.boxplot(
        data=plot_df,
        x="Gene_State_Label",
        y="Predicted_Probability",
        order=["Absent", "Present"],
        width=0.18,
        showfliers=False,
        showcaps=True,
        boxprops={
            "facecolor": "white",
            "edgecolor": "#444444",
            "linewidth": 1.5,
            "alpha": 0.92
        },
        whiskerprops={
            "color": "#444444",
            "linewidth": 1.4
        },
        capprops={
            "color": "#444444",
            "linewidth": 1.4
        },
        medianprops={
            "color": "black",
            "linewidth": 2.0
        },
        ax=ax
    )

    
    mean_df = (
        plot_df
        .groupby("Gene_State_Label", observed=True)["Predicted_Probability"]
        .mean()
        .reindex(["Absent", "Present"])
    )

    ax.scatter(
        [0, 1],
        mean_df.values,
        s=85,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=5
    )

    
    count_map = plot_df["Gene_State_Label"].value_counts().to_dict()

    ax.text(
        0, 0.985,
        f"n = {count_map.get('Absent', 0)}",
        ha="center",
        va="top",
        fontsize=12,
        color="#444444"
    )

    ax.text(
        1, 0.985,
        f"n = {count_map.get('Present', 0)}",
        ha="center",
        va="top",
        fontsize=12,
        color="#444444"
    )

    
    ax.set_ylim(-0.01, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.set_title(
        f"Fig.5E: Effect of Top1 SHAP Gene on Prediction\n"
        f"{host_title} | {target_gene}",
        fontsize=21,
        fontweight="bold",
        pad=10
    )

    ax.set_xlabel(
        f"{target_gene} Presence/Absence",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_ylabel(
        f"Predicted Probability ({host_title})",
        fontsize=18,
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=1.0,
        alpha=0.45,
        color="#BDBDBD"
    )
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    


    plt.tight_layout()

    out_png = os.path.join(
        out_dir,
        f"Fig5E_{host_name}_beautified_final_color.png"
    )
    out_pdf = os.path.join(
        out_dir,
        f"Fig5E_{host_name}_beautified_final_color.pdf"
    )

    plt.savefig(
        out_png,
        dpi=800,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.savefig(
        out_pdf,
        dpi=800,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()

    print(f"✅ 已保存: {out_png}")
    print(f"✅ 已保存: {out_pdf}")



if __name__ == "__main__":

    for host_name, file_name in file_map.items():
        csv_path = os.path.join(base_dir, file_name)
        draw_one_host_plot(
            host_name=host_name,
            csv_path=csv_path,
            out_dir=out_dir
        )

