import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable

base_dir = r"D:\bio_project\fig\fig5\1"

input_csv = os.path.join(
    base_dir,
    "Fig5_D_Top15_Union_SignedSHAP_Heatmap_CleanNames.csv"
)

out_dir = os.path.join(base_dir, "Fig5D_redraw_heatmap_vertical")
os.makedirs(out_dir, exist_ok=True)

out_png = os.path.join(out_dir, "Fig5D_SignedSHAP_Heatmap_Vertical.png")
out_pdf = os.path.join(out_dir, "Fig5D_SignedSHAP_Heatmap_Vertical.pdf")

legend_png = os.path.join(out_dir, "Fig5D_SignedSHAP_Heatmap_Legend.png")
legend_pdf = os.path.join(out_dir, "Fig5D_SignedSHAP_Heatmap_Legend.pdf")

df = pd.read_csv(input_csv)

print("读取文件：", input_csv)
print("原始列名：")
print(df.columns.tolist())
print(df.head())

possible_host_cols = ["Host", "host", "Target_Host", "target_host"]

host_col = None
for c in possible_host_cols:
    if c in df.columns:
        host_col = c
        break

if host_col is None:
    host_col = df.columns[0]

print("识别到 Host 列：", host_col)

df = df.set_index(host_col)

host_order = ["Human", "Pig", "Chicken"]

index_map = {str(idx).lower(): idx for idx in df.index}

ordered_index = []
for h in host_order:
    key = h.lower()
    if key in index_map:
        ordered_index.append(index_map[key])

if len(ordered_index) == 3:
    df = df.loc[ordered_index]
    df.index = host_order


heatmap_df = df.apply(pd.to_numeric, errors="coerce")


heatmap_df = heatmap_df.dropna(axis=1, how="all")


heatmap_plot_df = heatmap_df.T

mask = heatmap_plot_df.isna()

values = heatmap_plot_df.values.astype(float)

vmin = np.nanmin(values)
vmax = np.nanmax(values)


if vmin < 0 and vmax > 0:
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
else:
    norm = Normalize(vmin=vmin, vmax=vmax)


sns.set_theme(style="white")

plt.rcParams["font.family"] = "Arial"

plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20

plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 14


n_genes = heatmap_plot_df.shape[0]


fig_width = 6.2
fig_height = max(8, n_genes * 0.32)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

hm = sns.heatmap(
    heatmap_plot_df,
    ax=ax,
    cmap="RdBu_r",
    norm=norm,
    mask=mask,
    linewidths=0.35,
    linecolor="white",
    cbar=False
)

ax.set_title(
    "Fig.5D Signed SHAP Heatmap",
    fontsize=15,
    fontweight="bold",
    pad=10
)

ax.set_xlabel(
    "Host",
    fontsize=20,
    fontweight="bold",
    labelpad=8
)


ax.set_ylabel(
    "Union of Host-specific Top15 SHAP Genes",
    fontsize=20,
    fontweight="normal",
    labelpad=12
)

ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

ax.tick_params(
    axis="y",
    labelright=True,
    labelleft=False,
    right=True,
    left=False,
    pad=4
)


ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0,
    ha="center",
    fontsize=23,
    fontweight="normal"
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    ha="left",
    fontsize=18
)


for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color("black")

plt.tight_layout()

plt.savefig(out_png, dpi=800, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, dpi=800, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ 竖版主图已保存：")
print(out_png)
print(out_pdf)



fig, ax = plt.subplots(figsize=(1.2, 4.5))

sm = ScalarMappable(norm=norm, cmap="RdBu_r")
sm.set_array([])

cbar = fig.colorbar(
    sm,
    cax=ax,
    orientation="vertical"
)

cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels(["Low", "High"])

cbar.ax.tick_params(labelsize=23, length=0)
cbar.set_label(
    "Mean signed SHAP",
    fontsize=18,
    fontweight="normal",
    labelpad=10
)


for spine in cbar.ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color("black")

plt.savefig(legend_png, dpi=800, bbox_inches="tight", facecolor="white")
plt.savefig(legend_pdf, dpi=800, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ 单独 legend / colorbar 已保存：")
print(legend_png)
print(legend_pdf)