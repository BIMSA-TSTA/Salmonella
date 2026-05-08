import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings
warnings.filterwarnings('ignore')

pa_data = pd.read_csv("gene_presence_absence_cleaned.csv", index_col=0).T
pa_data.index = pa_data.index.str.replace(r'\.fna.*$', '', regex=True).str.strip()

meta_df = pd.read_csv("file_with_host_date_ip.csv")
id_col = 'ID' if 'ID' in meta_df.columns else meta_df.columns[0]
meta_df[id_col] = meta_df[id_col].astype(str).str.replace(r'\.fna.*$', '', regex=True).str.strip()
meta_df = meta_df.set_index(id_col)

common_ids = pa_data.index.intersection(meta_df.index)
pa_data = pa_data.loc[common_ids]
meta_df = meta_df.loc[common_ids]

gene_frequencies = pa_data.mean(axis=0)
accessory_genes = gene_frequencies[(gene_frequencies > 0.15) & (gene_frequencies < 0.95)].index
pa_subset = pa_data[accessory_genes]

if len(pa_subset.columns) > 2000:
    pa_subset = pa_subset.sample(n=2000, axis=1, random_state=42)
    print("基因数超过 2000，已随机抽取 2000 个基因用于绘图。")
host_labels = meta_df['Host'].fillna('Unknown')

lut = {
    'human': "#E41A1C",
    'pig': "#377EB8",
    'chicken': "#FFD700",
    'Unknown': '#D3D3D3'
}

row_colors = host_labels.map(lut)



col_dist = pdist(pa_subset.T, metric='jaccard')
col_linkage = linkage(col_dist, method='average')

col_order = leaves_list(col_linkage)

pa_subset_clustered = pa_subset.iloc[:, col_order]


binary_cmap = ListedColormap(["#FFFFFF", "#2166AC"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)


sns.set_theme(font_scale=0.8)

g = sns.clustermap(
    pa_subset_clustered,
    method='average',
    metric='jaccard',
    cmap=binary_cmap,
    norm=norm,
    row_colors=row_colors,
    figsize=(14, 10),
    xticklabels=False,
    yticklabels=False,
    cbar_pos=(0.02, 0.8, 0.03, 0.18),
    dendrogram_ratio=(0.2, 0.01),  
    col_cluster=False,              
    row_cluster=True               
)


g.fig.subplots_adjust(right=0.78, top=0.93, bottom=0.06)

g.cax.set_position([0.82, 0.25, 0.025, 0.40])

g.cax.set_yticks([0, 1])
g.cax.set_yticklabels(['0', '1'])
g.cax.set_ylabel("Gene Presence", fontsize=11)


legend_handles = [
    Patch(facecolor=lut['human'], edgecolor='none', label='human'),
    Patch(facecolor=lut['pig'], edgecolor='none', label='pig'),
    Patch(facecolor=lut['chicken'], edgecolor='none', label='chicken')
]

g.fig.legend(
    handles=legend_handles,
    title="Host Origin",
    loc="center left",
    bbox_to_anchor=(0.87, 0.55),
    frameon=False
)


g.fig.suptitle(
    'PAV-based Hierarchical Clustering (Accessory Genes)',
    y=0.98,
    fontsize=16,
    fontweight='bold'
)

plt.savefig("Fig3B_PAV_Heatmap.pdf", dpi=800, bbox_inches='tight')
plt.savefig("Fig3B_PAV_Heatmap.png", dpi=800, bbox_inches='tight')
print("✅ Fig 3B 绘制完成！已保存为 PDF 和 PNG。")