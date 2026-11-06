import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


base_dir = r"D:\bio_project\fig\fig5\3"

csv_candidates = glob.glob(os.path.join(base_dir, "*Clade*Probability*Table*.csv"))

if len(csv_candidates) == 0:
    raise FileNotFoundError(
        f"在 {base_dir} 没找到 *Clade*Probability*Table*.csv\n"
        f"请确认你的 CSV 是否叫 Scheme_0_Clade_Probability_Table.csv"
    )

input_csv = csv_candidates[0]

out_dir = os.path.join(base_dir, "Clade3D_redraw_3x1_big_legends")
os.makedirs(out_dir, exist_ok=True)

print("读取 CSV：", input_csv)


AXIS_LABEL_SIZE = 25
TICK_LABEL_SIZE = 22

LEGEND_FONT_SIZE = 28
LEGEND_TITLE_SIZE = 30

TITLE_SIZE = 25
SUPTITLE_SIZE = 32

HOST_COLOR_MAP = {
    "human": "#E41A1C",
    "pig": "#377EB8",
    "chicken": "#FFD700"
}

HOST_ORDER = ["human", "pig", "chicken"]
DEFAULT_COLOR = "#999999"


def get_host_color(host):
    return HOST_COLOR_MAP.get(str(host).lower(), DEFAULT_COLOR)


def clean_host_name(host):
    return str(host).strip().lower()


def beautify_3d_axis(ax):
    """
    美化 3D 坐标轴。
    """
    ax.set_facecolor("white")

    ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))

    ax.xaxis.pane.set_edgecolor("#BBBBBB")
    ax.yaxis.pane.set_edgecolor("#BBBBBB")
    ax.zaxis.pane.set_edgecolor("#BBBBBB")

    ax.xaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.35)
    ax.yaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.35)
    ax.zaxis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.35)

    ax.xaxis._axinfo["grid"]["linewidth"] = 0.8
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.8
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.8

    ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_SIZE, pad=4)
    ax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_SIZE, pad=4)
    ax.tick_params(axis="z", which="major", labelsize=TICK_LABEL_SIZE, pad=4)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def set_dynamic_limits(ax, df_sub, pad_ratio=0.10):
    
    for col, setter in [
        ("P_human", ax.set_xlim),
        ("P_pig", ax.set_ylim),
        ("P_chicken", ax.set_zlim)
    ]:
        vals = pd.to_numeric(df_sub[col], errors="coerce").dropna().values

        if len(vals) == 0:
            setter(0, 1)
            continue

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        if np.isclose(vmin, vmax):
            pad = 0.05
        else:
            pad = max(0.02, (vmax - vmin) * pad_ratio)

        lower = max(0.0, vmin - pad)
        upper = min(1.0, vmax + pad)

        if np.isclose(lower, upper):
            lower = max(0.0, lower - 0.05)
            upper = min(1.0, upper + 0.05)

        setter(lower, upper)


def save_legend_only(handles, filename_prefix, title=None, ncol=1, figsize=(3.8, 3.6)):
   
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    legend = ax.legend(
        handles=handles,
        title=title,
        loc="center",
        frameon=False,
        ncol=ncol,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        handlelength=1.3,
        handletextpad=0.7,
        columnspacing=1.2,
        labelspacing=0.9,
        borderaxespad=0.0
    )

    if title is not None:
        legend.get_title().set_fontweight("bold")

    out_png = os.path.join(out_dir, f"{filename_prefix}.png")
    out_pdf = os.path.join(out_dir, f"{filename_prefix}.pdf")

    plt.savefig(out_png, dpi=800, bbox_inches="tight", transparent=True)
    plt.savefig(out_pdf, dpi=800, bbox_inches="tight", transparent=True)
    plt.close()

    print("✅ 已保存 legend：", out_png)
    print("✅ 已保存 legend：", out_pdf)


df = pd.read_csv(input_csv)

print("列名：", list(df.columns))
print(df.head())

required_cols = ["P_human", "P_pig", "P_chicken", "Actual_Host", "Clade"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(
        f"CSV 缺少必要列: {missing_cols}\n"
        f"当前列名为: {list(df.columns)}"
    )

df["Actual_Host_clean"] = df["Actual_Host"].apply(clean_host_name)

for col in ["P_human", "P_pig", "P_chicken"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["P_human", "P_pig", "P_chicken"]).copy()

clade_order = sorted(
    df["Clade"].dropna().unique(),
    key=lambda x: int(str(x).split("_")[-1]) if str(x).split("_")[-1].isdigit() else 999
)

print("检测到 Clade：", clade_order)


marker_list = ["o", "^", "s", "D", "P", "X", "v", "<", ">"]

marker_map = {}
for i, clade in enumerate(clade_order):
    marker_map[clade] = marker_list[i % len(marker_list)]



host_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label=h.capitalize(),
        markerfacecolor=get_host_color(h),
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=17,
        linestyle="None"
    )
    for h in HOST_ORDER
]

save_legend_only(
    handles=host_handles,
    filename_prefix="Legend_ActualHost_3x1_bigfont",
    title="Actual host",
    ncol=1,
    figsize=(4.0, 4.0)
)

clade_handles = [
    Line2D(
        [0], [0],
        marker=marker_map.get(c, "o"),
        color="black",
        label=str(c).replace("_", " "),
        linestyle="None",
        markersize=17
    )
    for c in clade_order
]

save_legend_only(
    handles=clade_handles,
    filename_prefix="Legend_Clade_3x1_bigfont",
    title="Clade",
    ncol=1,
    figsize=(4.0, 4.0)
)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16

n_clades = len(clade_order)

fig = plt.figure(figsize=(8.8, 23.5))

for idx, clade in enumerate(clade_order, start=1):
    
    ax = fig.add_subplot(n_clades, 1, idx, projection="3d")

    df_c = df[df["Clade"] == clade].copy()

    for host in HOST_ORDER:
        sub = df_c[df_c["Actual_Host_clean"] == host].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub["P_human"],
            sub["P_pig"],
            sub["P_chicken"],
            s=48,
            alpha=0.82,
            color=get_host_color(host),
            edgecolor="black",
            linewidth=0.35
        )

    other_hosts = [
        h for h in df_c["Actual_Host_clean"].unique()
        if h not in HOST_ORDER
    ]

    for host in other_hosts:
        sub = df_c[df_c["Actual_Host_clean"] == host].copy()
        ax.scatter(
            sub["P_human"],
            sub["P_pig"],
            sub["P_chicken"],
            s=40,
            alpha=0.55,
            color=DEFAULT_COLOR,
            edgecolor="black",
            linewidth=0.25
        )

    ax.set_xlabel("P(human)", fontsize=AXIS_LABEL_SIZE, labelpad=13, fontweight="bold")
    ax.set_ylabel("P(pig)", fontsize=AXIS_LABEL_SIZE, labelpad=13, fontweight="bold")
    ax.set_zlabel("P(chicken)", fontsize=AXIS_LABEL_SIZE, labelpad=13, fontweight="bold")

    set_dynamic_limits(ax, df_c)

    ax.view_init(elev=23, azim=42)
    beautify_3d_axis(ax)

    ax.set_title(
        f"{str(clade).replace('_', ' ')}\n n = {len(df_c)}",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=16
    )

fig.suptitle(
    "3D Probability Space of Clade Samples",
    fontsize=SUPTITLE_SIZE,
    fontweight="bold",
    y=0.995
)

plt.subplots_adjust(
    left=0.10,
    right=0.96,
    top=0.955,
    bottom=0.025,
    hspace=0.03
)

out_png = os.path.join(out_dir, "Clade3D_3x1_NoLegend.png")
out_pdf = os.path.join(out_dir, "Clade3D_3x1_NoLegend.pdf")

plt.savefig(out_png, dpi=800, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, dpi=800, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ 已保存：", out_png)
print("✅ 已保存：", out_pdf)


fig = plt.figure(figsize=(11.5, 9.2))
ax = fig.add_subplot(111, projection="3d")

for clade in clade_order:
    df_c = df[df["Clade"] == clade].copy()
    marker = marker_map.get(clade, "o")

    for host in HOST_ORDER:
        sub = df_c[df_c["Actual_Host_clean"] == host].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub["P_human"],
            sub["P_pig"],
            sub["P_chicken"],
            s=52,
            alpha=0.78,
            marker=marker,
            color=get_host_color(host),
            edgecolor="black",
            linewidth=0.35
        )

ax.set_xlabel(
    "P(human)",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=14,
    fontweight="bold"
)

ax.set_ylabel(
    "P(pig)",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=14,
    fontweight="bold"
)

ax.set_zlabel(
    "P(chicken)",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=14,
    fontweight="bold"
)

set_dynamic_limits(ax, df)

ax.view_init(elev=24, azim=42)
beautify_3d_axis(ax)

ax.set_title(
    "All Clades in 3D Probability Space",
    fontsize=24,
    fontweight="bold",
    pad=20
)

plt.subplots_adjust(
    left=0.12,
    right=0.96,
    bottom=0.08,
    top=0.90
)

out_png = os.path.join(out_dir, "Clade3D_AllClades_OnePanel_NoLegend_fixed_margin.png")
out_pdf = os.path.join(out_dir, "Clade3D_AllClades_OnePanel_NoLegend_fixed_margin.pdf")

plt.savefig(
    out_png,
    dpi=800,
    facecolor="white",
    pad_inches=0.25
)

plt.savefig(
    out_pdf,
    dpi=800,
    facecolor="white",
    pad_inches=0.25
)

plt.close()

print("✅ 已保存：", out_png)
print("✅ 已保存：", out_pdf)