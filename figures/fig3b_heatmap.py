# -*- coding: utf-8 -*-
# PACKAGED COPY. Edit the absolute paths in the path block below before running.
"""
Fig 3b v2 —— 回应 Reviewer #2 次要意见 2

原稿问题（审稿人原话）：
  "图 3b 视觉上较为杂乱，目前的形式无法传达清晰的信息。
   按照基因功能类别对行和/或列排序，并增加标签或注释，有助于明确该热图想要展示的内容。"

原脚本 fig3b_heatmap.py 的问题：
  1. 列（基因）只按 Jaccard 层次聚类排序，**没有任何功能含义** —— 1800 列二值噪声，看不出结构
  2. 完全没有基因功能注释
  3. absent 用纯白 (#FFFFFF)，边界发虚，且色标写成 '0'/'1'，与图注的 Absent/Present 不一致
  4. 图例位置 (0.87, 0.55) 容易压到热图

v2 的做法：
  1. 列按 **COG 功能类别** 分块排序；同一 COG 块内部仍保留 Jaccard 聚类顺序
     → 既满足"按功能类别排序"，又不丢共现结构
  2. 热图上方加一条 **COG 类别色带** + 右侧 COG 图例（颜色与 Fig 2a 的饼图统一）
  3. 行（菌株）保留层次聚类，左侧保留 host 注释带
  4. absent 改浅灰、色标写 Absent/Present、图例全部移到图外
  5. 额外输出一张 **COG 汇总版**（菌株 × COG 类别，值为该类基因的存在比例），
     供判断哪种表达更适合放进手稿

输入：
  gene_presence_absence_cleaned.csv        0/1 矩阵（行=基因，列=样本）
  file_with_host_date_ip.csv               样本元数据（含 Host）
  eggNOG_Annotated_ValidCOG_Only.csv       基因 → COG 类别
输出：OUT_DIR 下的 PNG / PDF / CSV
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")

# =========================================================
# 0. 路径与参数
# =========================================================
PAV_CSV = r"D:\bio_project\代码\gene_presence_absence_cleaned.csv"
META_CSV = r"D:\bio_project\代码\file_with_host_date_ip.csv"
COG_CSV = r"D:\bio_project\egg\COG_only_annotated\eggNOG_Annotated_ValidCOG_Only.csv"
COG_COLOR_CSV = r"D:\bio_project\egg\COG_only_annotated\COG_Color_Map.csv"

OUT_DIR = r"D:\bio_project\代码\Fig3B_v3_barsright"
os.makedirs(OUT_DIR, exist_ok=True)

FREQ_LOW, FREQ_HIGH = 0.15, 0.95   # accessory 定义，与全文保持一致
DPI = 800
RANDOM_STATE = 42

HOST_LUT = {"human": "#E41A1C", "pig": "#377EB8", "chicken": "#FFD700"}

PRESENT_COLOR = "#2166AC"   # 存在：深蓝
ABSENT_COLOR = "#F0F0F0"    # 不存在：浅灰（原来是纯白，边界发虚）


def log(msg):
    print(msg, flush=True)


# =========================================================
# 1. 读取数据
# =========================================================
log("=" * 78)
log("[1/6] 读取 PAV 矩阵 ...")
t0 = time.time()

pa = pd.read_csv(PAV_CSV, index_col=0)
log(f"      原始 shape (gene x sample): {pa.shape}")
pa = pa.T
pa.index = pa.index.astype(str).str.replace(r"\.fna.*$", "", regex=True).str.strip()
log(f"      转置后 shape (sample x gene): {pa.shape}  用时 {time.time()-t0:.1f}s")

log("[2/6] 读取样本元数据 ...")
meta = pd.read_csv(META_CSV)
id_col = "ID" if "ID" in meta.columns else meta.columns[0]
meta[id_col] = meta[id_col].astype(str).str.replace(r"\.fna.*$", "", regex=True).str.strip()
meta = meta.set_index(id_col)

common = pa.index.intersection(meta.index)
pa = pa.loc[common]
meta = meta.loc[common]
log(f"      匹配样本数: {len(common)}")

host = meta["Host"].astype(str).str.strip().str.lower().fillna("unknown")
log("      宿主分布:\n" + host.value_counts().to_string())

# =========================================================
# 2. 选 accessory 基因（与全文阈值一致）
# =========================================================
log("[3/6] 筛选 accessory 基因 ...")
freq = pa.mean(axis=0)
keep = freq[(freq > FREQ_LOW) & (freq < FREQ_HIGH)].index
pa_sub = pa[keep]
log(f"      阈值 ({FREQ_LOW}, {FREQ_HIGH}) -> 保留 {pa_sub.shape[1]} 个基因 / 共 {pa.shape[1]}")

# =========================================================
# 3. COG 功能注释
# =========================================================
log("[4/6] 读取 COG 注释 ...")
cog = pd.read_csv(COG_CSV)
gene_col = "Gene" if "Gene" in cog.columns else cog.columns[0]
cog_col = "COG_category" if "COG_category" in cog.columns else cog.columns[1]
cog_map = dict(zip(cog[gene_col].astype(str), cog[cog_col].astype(str)))

NA_LABEL = "Not annotated"
col_cog = pd.Series([cog_map.get(str(g), NA_LABEL) for g in pa_sub.columns],
                    index=pa_sub.columns)
n_annot = int((col_cog != NA_LABEL).sum())
log(f"      有 COG 注释的基因: {n_annot} / {pa_sub.shape[1]} "
    f"({100*n_annot/pa_sub.shape[1]:.1f}%)")

# COG 颜色（优先与 Fig 2a 饼图统一）
if os.path.exists(COG_COLOR_CSV):
    cm = pd.read_csv(COG_COLOR_CSV)
    cog_colors = dict(zip(cm.iloc[:, 0].astype(str), cm.iloc[:, 1].astype(str)))
    log(f"      使用 Fig 2a 的 COG 配色 ({len(cog_colors)} 类)")
else:
    letters = sorted(set(col_cog) - {NA_LABEL})
    pal = sns.color_palette("tab20", n_colors=max(len(letters), 3)).as_hex()
    cog_colors = {l: c for l, c in zip(letters, pal)}
    log("      未找到 COG_Color_Map.csv，改用 tab20 配色")
cog_colors[NA_LABEL] = "#D9D9D9"

# =========================================================
# 4. 列排序：COG 分块 + 块内保留 Jaccard 聚类顺序
# =========================================================
log("[5/6] 计算列聚类（基因，Jaccard/average）...")
t1 = time.time()
col_link = linkage(pdist(pa_sub.T.values.astype(bool), metric="jaccard"), method="average")
col_cluster_order = leaves_list(col_link)
log(f"      完成，用时 {time.time()-t1:.1f}s")

rank = {g: i for i, g in enumerate(pa_sub.columns[col_cluster_order])}
# COG 类别按字母升序，未注释的放最后；同一 COG 块内保留聚类顺序
def cog_key(gene):
    c = col_cog[gene]
    return (1, "", rank[gene]) if c == NA_LABEL else (0, c, rank[gene])

order = sorted(pa_sub.columns, key=cog_key)
pa_ord = pa_sub[order]
col_cog_ord = col_cog[order]

# COG 色带
col_colors = col_cog_ord.map(cog_colors)
col_colors.name = "COG"

# COG 块边界（用于画分隔线）
blocks = col_cog_ord.groupby(col_cog_ord, sort=False).size()
log("      COG 分块（按列顺序）: " + ", ".join(f"{k}:{v}" for k, v in blocks.items()))

# =========================================================
# 5. 行聚类（菌株）—— 结果缓存，重跑不必再算
# =========================================================
ROW_LINK_CACHE = os.path.join(OUT_DIR, "row_linkage_jaccard.npz")
if os.path.exists(ROW_LINK_CACHE):
    log("[6/6] 载入已缓存的行聚类结果 ...")
    row_link = np.load(ROW_LINK_CACHE)["linkage"]
else:
    log("[6/6] 计算行聚类（菌株，Jaccard/average）—— 这一步较慢，会缓存 ...")
    t2 = time.time()
    row_link = linkage(pdist(pa_ord.values.astype(bool), metric="jaccard"), method="average")
    np.savez_compressed(ROW_LINK_CACHE, linkage=row_link)
    log(f"      完成，用时 {time.time()-t2:.1f}s")

row_ord = leaves_list(row_link)

# ⚠️ 重要：这里【不要】把数据按 row_ord 预先排序！
# seaborn.clustermap 在收到 row_linkage 时会自己执行 data.iloc[yind]（位置重排，
# 见 seaborn/matrix.py:1137 -> 1061）。若这里先排一次，等于排了两次，
# 热图行序就会与树状图不一致。
# 因此：行保持原始顺序交给 seaborn；row_colors 也必须与数据同序。
pa_ord = pa_ord          # 行 = 原始顺序；列 = 已按 COG 分块排序
host_ord = host          # 与 pa_ord.index 完全一致


# =========================================================
# 6. 画主图：基因级热图（列按 COG 分块）
#    v3：热图左侧只保留行聚类树；host 色带、Gene status 色标、两个图例
#        全部移到热图右侧（COG 列色带属于列注释，仍留在热图上方）
# =========================================================
log("\n绘制主图 v3（色带/色标/图例全部置于热图右侧）...")

cmap = ListedColormap([ABSENT_COLOR, PRESENT_COLOR])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

HOST_SHOW = {"human": "human", "pig": "swine", "chicken": "chicken"}   # 正文统一 swine

sns.set_theme(font_scale=1.0)
g = sns.clustermap(
    pa_ord.astype(float),
    row_linkage=row_link,
    col_cluster=False,
    row_cluster=True,
    cmap=cmap,
    norm=norm,
    row_colors=host_ord.map(HOST_LUT),
    col_colors=col_colors,
    figsize=(16, 11),
    xticklabels=False,
    yticklabels=False,
    dendrogram_ratio=(0.10, 0.02),
    cbar_pos=(0.90, 0.45, 0.010, 0.10),
)

fig = g.fig
HM = [0.048, 0.085, 0.590, 0.800]                 # 热图本体
g.ax_heatmap.set_position(HM)
g.ax_row_dendrogram.set_position([0.012, HM[1], 0.030, HM[3]])            # 行树：最左
g.ax_row_colors.set_position([HM[0] + HM[2] + 0.009, HM[1], 0.013, HM[3]])  # host 色带 → 右
g.ax_col_colors.set_position([HM[0], HM[1] + HM[3] + 0.007, HM[2], 0.013])  # COG 色带（列注释）→ 顶
g.cax.set_position([HM[0] + HM[2] + 0.032, HM[1] + HM[3] - 0.125, 0.014, 0.115])  # 色标 → 右

# COG 块分隔线
cog_seq = list(col_cog_ord.values)
boundaries = [i for i in range(1, len(cog_seq)) if cog_seq[i] != cog_seq[i - 1]]
for b in boundaries:
    g.ax_heatmap.axvline(b, color="white", linewidth=0.6, alpha=0.9)

g.ax_heatmap.set_xlabel(f"Accessory gene clusters, grouped by COG category "
                        f"(n = {pa_ord.shape[1]})", fontsize=13, labelpad=10)
g.ax_heatmap.set_ylabel(f"Isolates (n = {pa_ord.shape[0]})", fontsize=13, labelpad=10)

# 色标（热图右侧）
g.cax.set_yticks([0, 1])
g.cax.set_yticklabels(["Absent", "Present"], fontsize=10)
g.cax.set_title("Gene\nstatus", fontsize=10, pad=8)

# host 图例（右上）
host_handles = [Patch(facecolor=HOST_LUT[h], edgecolor="none", label=HOST_SHOW[h])
                for h in ["human", "pig", "chicken"] if h in set(host_ord)]
g.fig.legend(handles=host_handles, title="Host origin", loc="upper left",
             bbox_to_anchor=(0.672, 0.972), frameon=False, fontsize=11,
             title_fontsize=11)

# COG 图例（右侧）
cog_handles = [Patch(facecolor=cog_colors[c], edgecolor="none",
                     label=f"{c}  ({blocks[c]})")
               for c in blocks.index]
g.fig.legend(handles=cog_handles, title="COG category (n genes)",
             loc="center left", bbox_to_anchor=(0.672, 0.44),
             frameon=False, fontsize=9.5, title_fontsize=10.5, ncol=2,
             handlelength=1.2, columnspacing=1.0, labelspacing=0.45)

g.fig.suptitle("PAV-based hierarchical clustering of accessory genes,\n"
               "with columns grouped by COG functional category",
               y=0.985, fontsize=16, fontweight="bold")

# PNG 全分辨率；PDF 降 dpi（v2 的 800 dpi 矢量版达 198 MB，不能直接投稿）
# 热图/色带格子数以百万计，PDF 里必须栅格化，否则单文件上百 MB
for _ax in (g.ax_heatmap, g.ax_row_colors, g.ax_col_colors):
    for _art in list(_ax.get_children()) + list(getattr(_ax, "collections", [])):
        try:
            _art.set_rasterized(True)
        except Exception:
            pass

g.fig.savefig(os.path.join(OUT_DIR, "Fig3B_PAV_Heatmap_v3_barsright.png"),
              dpi=DPI, bbox_inches="tight", facecolor="white")
g.fig.savefig(os.path.join(OUT_DIR, "Fig3B_PAV_Heatmap_v3_barsright.pdf"),
              dpi=300, bbox_inches="tight", facecolor="white")
log("      ✅ PNG(%d dpi) + PDF(300 dpi)" % DPI)
plt.close(g.fig)

# 保存列顺序表（便于核查）
pd.DataFrame({
    "Column_Index": range(len(col_cog_ord)),
    "Gene": col_cog_ord.index,
    "COG_category": col_cog_ord.values,
}).to_csv(os.path.join(OUT_DIR, "Fig3B_v3_Column_Order.csv"), index=False)

# =========================================================
# 7. 备选版：COG 汇总热图（菌株 × COG 类别，值为该类基因存在比例）
# =========================================================
log("\n绘制备选图（COG 汇总，值 = 该类基因在该菌株中的存在比例）...")

annot_mask = col_cog_ord != NA_LABEL
pa_annot = pa_ord.loc[:, annot_mask.values]
cog_annot = col_cog_ord[annot_mask]

agg = pa_annot.T.groupby(cog_annot.values).mean().T   # 样本 × COG
agg = agg[sorted(agg.columns)]

agg_link = linkage(pdist(agg.values, metric="euclidean"), method="average")
agg_ord = agg.iloc[leaves_list(agg_link)]
host_agg = host.loc[agg_ord.index]

fig, ax = plt.subplots(figsize=(9.5, 11))
sns.heatmap(
    agg_ord, ax=ax, cmap="YlGnBu", vmin=0, vmax=1,
    yticklabels=False, xticklabels=True,
    cbar_kws={"label": "Fraction of genes present in this COG", "shrink": 0.45},
)
ax.set_title("PAV summary at COG-category level\n(only COG-annotated accessory genes)",
             fontsize=14, fontweight="bold", pad=14)

# 注意：先把热图缩窄，再依据其最终位置把 host 色带放到「右侧」，
# 否则色带会留在左边、或与色标重叠。
fig.tight_layout(rect=[0.02, 0.02, 0.90, 0.95])
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width - 0.10, pos.height])
pos = ax.get_position()

# host 色带 → 热图右侧
ax_host = fig.add_axes([pos.x1 + 0.010, pos.y0, 0.011, pos.height])
for i, h in enumerate(host_agg):
    ax_host.add_patch(plt.Rectangle((0, i), 1, 1, color=HOST_LUT.get(h, "#999999")))
ax_host.set_xlim(0, 1); ax_host.set_ylim(len(host_agg), 0)
ax_host.axis("off")
ax_host.set_title("Host", fontsize=10, pad=6)

# 色标再往右挪，避免压住 host 色带
cbar_ax = ax.collections[0].colorbar.ax
cbar_ax.set_position([pos.x1 + 0.048, pos.y0 + pos.height * 0.30, 0.014, pos.height * 0.40])

fig.legend(handles=[Patch(facecolor=HOST_LUT[h], edgecolor="none",
                           label=HOST_SHOW[h])
                    for h in ["human", "pig", "chicken"] if h in set(host_agg)],
           loc="upper right", bbox_to_anchor=(0.995, 0.965), frameon=False, fontsize=10)
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"Fig3B_v3_alt_COGlevel.{ext}")
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    log(f"      ✅ {p}")
plt.close(fig)

agg_ord.to_csv(os.path.join(OUT_DIR, "Fig3B_v3_COGlevel_Matrix.csv"))

# =========================================================
# 8. 各条色带 / 图例「单独导出」，便于在排版软件里自由摆放
#    统一放在 OUT_DIR 下，文件名前缀 Bar_ / Legend_
# =========================================================
log("\n单独导出各色带与图例 ...")

import matplotlib as mpl
from matplotlib.colors import to_rgb

yind = leaves_list(row_link)
host_sorted = host_ord.iloc[yind]
pd.DataFrame({"Row_Index": range(len(host_sorted)),
              "Sample": host_sorted.index,
              "Host": host_sorted.values}).to_csv(
    os.path.join(OUT_DIR, "Fig3B_v3_Row_Order.csv"), index=False)


def export_strip(colors, vertical, fname, thickness_in, length_in, dpi=800):
    """把一条色带导出成独立图片；vertical=True 为竖条（行注释），False 为横条（列注释）。"""
    arr = np.array([to_rgb(c) for c in colors])
    if vertical:
        img, figsize = arr.reshape(-1, 1, 3), (thickness_in, length_in)
    else:
        img, figsize = arr.reshape(1, -1, 3), (length_in, thickness_in)
    f = plt.figure(figsize=figsize)
    a = f.add_axes([0, 0, 1, 1])
    a.imshow(img, aspect="auto", interpolation="nearest")
    a.axis("off")
    for ext in ("png", "pdf"):
        f.savefig(os.path.join(OUT_DIR, fname + "." + ext), dpi=dpi, facecolor="white")
    plt.close(f)
    log("      ✅ %s.png / .pdf" % fname)


def export_legend(handles, title, fname, figsize, ncol=1, transparent=True, **kw):
    f = plt.figure(figsize=figsize)
    f.legend(handles=handles, title=title, loc="center", frameon=False,
             ncol=ncol, **kw)
    for ext in ("png", "pdf"):
        f.savefig(os.path.join(OUT_DIR, fname + "." + ext), dpi=600,
                  bbox_inches="tight", transparent=transparent)
    plt.close(f)
    log("      ✅ %s.png / .pdf" % fname)


# 行注释：host 色带（顺序与热图行顺序一致）
export_strip([HOST_LUT.get(h, "#999999") for h in host_sorted.values],
             True, "Bar_HostOrigin", 0.30, 8.80)
# 列注释：COG 类别色带（顺序与热图列顺序一致）
export_strip([cog_colors[c] for c in col_cog_ord.values],
             False, "Bar_COGcategory", 0.30, 9.44)
# 存在/缺失色标
_f = plt.figure(figsize=(1.1, 2.4))
_cax = _f.add_axes([0.34, 0.18, 0.22, 0.62])
_sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); _sm.set_array([])
_cb = _f.colorbar(_sm, cax=_cax, ticks=[0, 1])
_cb.ax.set_yticklabels(["Absent", "Present"], fontsize=11)
_cb.ax.set_title("Gene\nstatus", fontsize=11, pad=8)
for ext in ("png", "pdf"):
    _f.savefig(os.path.join(OUT_DIR, "Bar_GeneStatus_colorbar." + ext),
               dpi=600, bbox_inches="tight", facecolor="white")
plt.close(_f)
log("      ✅ Bar_GeneStatus_colorbar.png / .pdf")

# 两个图例也单独导出（透明底，便于叠加）
export_legend(host_handles, "Host origin", "Legend_HostOrigin", (2.4, 1.1),
              fontsize=11, title_fontsize=11)
export_legend(cog_handles, "COG category (n genes)", "Legend_COGcategory", (5.0, 4.4),
              ncol=2, fontsize=9.5, title_fontsize=10.5,
              handlelength=1.2, columnspacing=1.0, labelspacing=0.45)

log("\n🎉 完成。输出目录：" + OUT_DIR)
