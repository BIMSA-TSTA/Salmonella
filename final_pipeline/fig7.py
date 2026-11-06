# -*- coding: utf-8 -*-
"""
Final Figure 7 (single composite, panels a-d) for the revised manuscript.

  a  per-class performance of the three one-vs-rest classifiers on the independent
     validation cohort (A-set, n = 223), six metrics with 95% bootstrap CIs
  b  ROC and precision-recall curves for the same three tasks
  c  confusion matrices of the integrated three-class model: 5-fold (development-set
     out-of-fold, n = 6,210) and Independent validation (n = 223)
  d  integrated three-class model on the independent cohort across six evaluation
     dimensions, with 95% bootstrap CIs

"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             matthews_corrcoef, f1_score, accuracy_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.environ.get("FIG7_DATA_DIR", os.path.join(_HERE, "..", "expected_outputs"))
SEED, NBOOT, THR = 20251009, 2000, 0.5
PRED = os.environ.get("FIG7_PRED", os.path.join(_DATA, "Fig7_Aset_Independent_Integrated_ThreeClass_Predictions.csv"))
B_METRICS = os.environ.get("FIG7_B_METRICS", os.path.join(_DATA, "Fig7_Integrated_ThreeClass_Metrics_BsetCV_vs_Aset.csv"))
OUT = os.environ.get("FIG7_OUT", os.path.join(_HERE, "fig7_out"))
HOSTS = ["human", "pig", "chicken"]
DISP = {"human": "human", "pig": "swine", "chicken": "chicken"}
LAB3 = [DISP[h] for h in HOSTS]
HOST_COLORS = {"human": "#E41A1C", "pig": "#377EB8", "chicken": "#FFD700"}
ASET_COLOR = "#F4A259"
BSET_COLOR = "#7AA6DC"
CMAP_CM = LinearSegmentedColormap.from_list("lightblue", sns.light_palette("#2C7FB8", as_cmap=False))
FIG_W, FIG_H = 7.4, 4.80
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="ticks")
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222", "ytick.color": "#222222",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "grid.color": "#BEBEBE", "grid.linestyle": "--", "grid.linewidth": 0.6,
    "savefig.facecolor": "white",
})


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.9); s.set_color("#444444")


# ================================================================== data
df = pd.read_csv(PRED)
y = df["True_Host"].to_numpy()
y_am = df["Predicted_Host"].to_numpy()
P = {h: df["P_" + h].to_numpy(float) for h in HOSTS}
PB = {h: (P[h] >= THR).astype(int) for h in HOSTS}

rows = []
for h in HOSTS:
    yb = (y == h).astype(int); pb = PB[h]
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    rows.append(dict(host=h, n_pos=int(yb.sum()), prevalence=float(yb.mean()),
                     TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                     ROC_AUC=roc_auc_score(yb, P[h]), PR_AUC=average_precision_score(yb, P[h]),
                     Accuracy=accuracy_score(yb, pb), Precision=precision_score(yb, pb, zero_division=0),
                     Recall=recall_score(yb, pb, zero_division=0),
                     Specificity=tn / (tn + fp),
                     Macro_F1=(f1_score(yb, pb, zero_division=0) + f1_score(1 - yb, 1 - pb, zero_division=0)) / 2,
                     MCC=matthews_corrcoef(yb, pb), Brier=float(np.mean((P[h] - yb) ** 2))))
res = pd.DataFrame(rows).set_index("host")

rng = np.random.default_rng(SEED)
idx = {h: np.flatnonzero(y == h) for h in HOSTS}
KEYS = ["ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "Specificity", "Macro_F1", "MCC", "Brier"]
boot = {k: {h: [] for h in HOSTS} for k in KEYS}
for _ in range(NBOOT):
    pick = np.concatenate([rng.choice(idx[h], len(idx[h]), replace=True) for h in HOSTS])
    yy = y[pick]
    for h in HOSTS:
        yb = (yy == h).astype(int); pb = (P[h][pick] >= THR).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
        v = dict(ROC_AUC=roc_auc_score(yb, P[h][pick]),
                 PR_AUC=average_precision_score(yb, P[h][pick]),
                 Accuracy=accuracy_score(yb, pb), Precision=precision_score(yb, pb, zero_division=0),
                 Recall=recall_score(yb, pb, zero_division=0), Specificity=tn / (tn + fp),
                 Macro_F1=(f1_score(yb, pb, zero_division=0) + f1_score(1 - yb, 1 - pb, zero_division=0)) / 2,
                 MCC=matthews_corrcoef(yb, pb), Brier=float(np.mean((P[h][pick] - yb) ** 2)))
        for k, val in v.items():
            boot[k][h].append(val)
for k in KEYS:
    for h in HOSTS:
        a = np.array(boot[k][h], float)
        res.loc[h, k + "_lo"], res.loc[h, k + "_hi"] = np.percentile(a, [2.5, 97.5])
res.reset_index().assign(host=lambda d: [DISP[h] for h in d["host"]]).round(6).to_csv(
    os.path.join(OUT, "Figure7a_数据_逐类指标含95CI.csv"), index=False, encoding="utf-8-sig")

LAB = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision",
       "Macro Recall / Balanced Acc", "MCC"]


def mset(yt, yp):
    return {"Accuracy": accuracy_score(yt, yp),
            "Macro F1": f1_score(yt, yp, labels=HOSTS, average="macro", zero_division=0),
            "Weighted F1": f1_score(yt, yp, labels=HOSTS, average="weighted", zero_division=0),
            "Macro Precision": precision_score(yt, yp, labels=HOSTS, average="macro", zero_division=0),
            "Macro Recall / Balanced Acc": recall_score(yt, yp, labels=HOSTS, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(yt, yp)}


A = mset(y, y_am)
_b = pd.read_csv(B_METRICS)
_b = _b[_b.Dataset == "B-set 5-fold CV"].set_index("Metric")["Value"]
B = {k: float(_b[k]) for k in LAB}
bo = {k: [] for k in LAB}
for _ in range(NBOOT):
    pick = np.concatenate([rng.choice(idx[h], len(idx[h]), replace=True) for h in HOSTS])
    m = mset(y[pick], y_am[pick])
    for k in LAB:
        bo[k].append(m[k])
lo = [np.percentile(bo[k], 2.5) for k in LAB]; hi = [np.percentile(bo[k], 97.5) for k in LAB]
pd.DataFrame({"Metric": LAB, "Development_set_5fold_CV": [B[k] for k in LAB],
              "Independent_Aset": [A[k] for k in LAB],
              "Aset_CI_low": lo, "Aset_CI_high": hi}).round(6).to_csv(
    os.path.join(OUT, "Figure7d_数据_集成模型_B集CV与A集.csv"), index=False, encoding="utf-8-sig")

CM_B = np.array([[3230, 120, 275], [334, 476, 55], [390, 71, 1259]])   # n = 6,210
CM_A = np.array([[68, 0, 3], [34, 78, 6], [12, 16, 6]])                # n = 223

MET = [("ROC_AUC", "ROC-AUC"), ("PR_AUC", "PR-AUC"), ("Recall", "Recall"),
       ("Accuracy", "Accuracy"), ("Macro_F1", "Macro F1"), ("MCC", "MCC")]

# ================================================================== figure
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(2, 2, figure=fig, left=0.084, right=0.988, top=0.958, bottom=0.102,
              hspace=0.22, wspace=0.12)


LETTERS = []


def letter(ax, txt):
    """bold panel letter (no parentheses), sitting on the same line as the panel title"""
    p = ax.get_position()
    x, y = p.x0 - 0.016, p.y1 + 2.0 / (FIG_H * 72.0)
    LETTERS.append((txt, x, y, p))
    fig.text(x, y, txt, ha="right", va="bottom",
             fontsize=8.5, fontweight="bold", color="#111111")


# ---- a: 2 x 3 small multiples, shared y axis
gsa = gs[0, 0].subgridspec(2, 3, hspace=0.22, wspace=0.22)
a_axes = []
for i, (k, lab) in enumerate(MET):
    r, c = i // 3, i % 3
    ax = fig.add_subplot(gsa[r, c]); a_axes.append(ax)
    v = [res.loc[h, k] for h in HOSTS]
    l_ = [res.loc[h, k + "_lo"] for h in HOSTS]
    h_ = [res.loc[h, k + "_hi"] for h in HOSTS]
    ax.bar(range(3), v, color=[HOST_COLORS[h] for h in HOSTS], width=0.68, alpha=0.92,
           edgecolor="#555555", linewidth=0.6,
           yerr=[np.array(v) - np.array(l_), np.array(h_) - np.array(v)],
           capsize=1.7, error_kw=dict(lw=0.8, ecolor="black"))
    for i2, val in enumerate(v):
        ax.text(i2, min(h_[i2], 1.0) + 0.03, "%.2f" % val, ha="center", va="bottom",
                fontsize=5.2, color="#222222")
    if k == "PR_AUC":
        for i2, h in enumerate(HOSTS):
            ax.hlines(res.loc[h, "prevalence"], i2 - 0.34, i2 + 0.34,
                      color=HOST_COLORS[h], ls="--", lw=1.0)
    ax.set_ylim(0, 1.16); ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(3))
    ax.set_xticklabels(LAB3 if r == 1 else [], fontsize=5.2)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"] if c == 0 else [], fontsize=5.4)
    ax.set_title(lab, fontsize=6.6, fontweight="bold", loc="left", pad=2.2)
    style(ax)
letter(a_axes[0], "a")

_apos = [ax.get_position() for ax in a_axes]
fig.text(min(p.x0 for p in _apos) - 0.042,
         (min(p.y0 for p in _apos) + max(p.y1 for p in _apos)) / 2.0, "Score",
         rotation=90, ha="center", va="center", fontsize=6.2, color="#222222")

# ---- b: ROC + precision-recall
gsb = gs[0, 1].subgridspec(1, 2, wspace=0.30)
a1 = fig.add_subplot(gsb[0, 0]); a2 = fig.add_subplot(gsb[0, 1])
for h in HOSTS:
    yb = (y == h).astype(int)
    fpr, tpr, _ = roc_curve(yb, P[h]); a1.plot(fpr, tpr, color=HOST_COLORS[h], lw=1.5)
    pr, rc, _ = precision_recall_curve(yb, P[h]); a2.plot(rc, pr, color=HOST_COLORS[h], lw=1.5)
    a2.axhline(res.loc[h, "prevalence"], color=HOST_COLORS[h], ls="--", lw=0.8, alpha=0.85)
a1.plot([0, 1], [0, 1], color="#999999", ls=":", lw=0.9)
a1.set_xlabel("False-positive rate", fontsize=6.2)
a1.set_ylabel("True-positive rate", fontsize=6.2)
a1.set_title("ROC curves", fontsize=6.6, fontweight="bold", loc="left", pad=2.2)
a2.set_xlabel("Recall (sensitivity)", fontsize=6.2)
a2.set_ylabel("Precision", fontsize=6.2)
a2.set_title("Precision-recall curves", fontsize=6.6, fontweight="bold", loc="left", pad=2.2)
for ax in (a1, a2):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.tick_params(labelsize=5.4)
    style(ax)
letter(a1, "b")

# ---- c: two confusion matrices
gsc = gs[1, 0].subgridspec(1, 2, wspace=0.38)
c_axes = []
for j, (CM, title) in enumerate(zip((CM_B, CM_A), ("5-fold", "Independent validation"))):
    ax = fig.add_subplot(gsc[0, j]); c_axes.append(ax)
    rp = CM / CM.sum(axis=1, keepdims=True)
    ax.imshow(rp, cmap=CMAP_CM, vmin=0, vmax=1)
    for i in range(3):
        for k2 in range(3):
            ax.text(k2, i, "%d\n(%.0f%%)" % (CM[i, k2], 100 * rp[i, k2]), ha="center", va="center",
                    fontsize=5.6, color="white" if rp[i, k2] > 0.62 else "#222222",
                    fontweight="bold" if i == k2 else "normal")
    for e in np.arange(-0.5, 3, 1.0):
        ax.axhline(e, color="white", lw=1.6, zorder=3)
        ax.axvline(e, color="white", lw=1.6, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(LAB3, fontsize=5.6)
    ax.set_yticks(range(3)); ax.set_yticklabels(LAB3, fontsize=5.6)
    ax.set_xlabel("Predicted host", fontsize=6.2)
    if j == 0:                       # only the left matrix carries the y label
        ax.set_ylabel("True host", fontsize=6.2, labelpad=1.5)
    ax.set_title(title, fontsize=6.4, fontweight="bold", loc="left", pad=2.2)
    for i in range(3):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                               edgecolor="#C0392B", lw=1.1, zorder=4))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.9); s.set_color("#444444")
letter(c_axes[0], "c")

# ---- d: integrated three-class model, six dimensions
axd = fig.add_subplot(gs[1, 1])
xs = np.arange(len(LAB))
va = [A[k] for k in LAB]
axd.bar(xs, va, 0.60, color=ASET_COLOR, alpha=0.95, edgecolor="#555555", linewidth=0.6,
        yerr=[np.array(va) - np.array(lo), np.array(hi) - np.array(va)],
        capsize=1.8, error_kw=dict(lw=0.8, ecolor="black"))
for x, v, h in zip(xs, va, hi):
    axd.text(x, min(h, 1.0) + 0.02, "%.3f" % v, ha="center", va="bottom", fontsize=5.2, color="#222222")
axd.set_xticks(xs)
axd.set_xticklabels(["Accuracy", "Macro\nF1", "Weighted\nF1", "Macro\nprecision",
                     "Balanced\naccuracy", "MCC"], fontsize=5.2)
axd.set_ylim(0, 1.05)
axd.set_yticks([0, 0.5, 1.0]); axd.set_yticklabels(["0", "0.5", "1"], fontsize=5.4)
axd.set_ylabel("Score", fontsize=6.2)
axd.tick_params(labelsize=5.4)
style(axd)
letter(axd, "d")

# ---- shared legend
lh = [Patch(facecolor=HOST_COLORS[h], edgecolor="#555555", linewidth=0.5, alpha=0.92,
            label="%s (AUC %.2f, PR-AUC %.2f)" % (DISP[h], res.loc[h, "ROC_AUC"], res.loc[h, "PR_AUC"]))
      for h in HOSTS]
lh.append(Patch(facecolor=ASET_COLOR, edgecolor="#555555", linewidth=0.5,
                label="independent validation (panel d)"))
fig.legend(handles=lh, loc="lower center", ncol=4, frameon=False, fontsize=5.9,
           bbox_to_anchor=(0.5, 0.008), handlelength=1.1, handleheight=0.9,
           handletextpad=0.35, columnspacing=1.3)

# ================================================================== audit + save
if os.environ.get("FIG7_DEBUG"):
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.canvas.get_width_height()
    items = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((s.replace("\n", "/"), bb, t.axes))
    print("--- canvas %dx%d  text artists %d ---" % (fw, fh, len(items)))
    for s, bb, _ in items:
        if bb.x0 < -1 or bb.x1 > fw + 1 or bb.y0 < -1 or bb.y1 > fh + 1:
            print("   OUTSIDE  %-26r x[%7.1f %7.1f] y[%7.1f %7.1f]" % (s[:26], bb.x0, bb.x1, bb.y0, bb.y1))
    nov = 0
    for i in range(len(items)):
        for k in range(i + 1, len(items)):
            if items[i][2] is items[k][2]:
                continue
            a, b = items[i][1], items[k][1]
            if a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1:
                print("   OVERLAP  %-24r x[%6.1f %6.1f] y[%6.1f %6.1f] <-> %-24r x[%6.1f %6.1f] y[%6.1f %6.1f]"
                      % (items[i][0][:24], a.x0, a.x1, a.y0, a.y1,
                         items[k][0][:24], b.x0, b.x1, b.y0, b.y1))
                nov += 1
    print("   overlaps:", nov)
    for nm, lx, ly, p in LETTERS:
        me = [bb for s, bb, ax in items if s == nm and ax is None]
        if not me:
            continue
        a = me[0]
        best, nb = 1e9, ""
        for s, bb, ax in items:
            if bb is a or (ax is None and s == nm):
                continue
            gx = max(a.x0 - bb.x1, bb.x0 - a.x1, 0.0)
            gy = max(a.y0 - bb.y1, bb.y0 - a.y1, 0.0)
            d = np.hypot(gx, gy)
            if d < best:
                best, nb = d, s
        print("   letter %-2s  nearest other text %-22r  clearance %.1f px (%.2f mm)"
              % (nm, nb[:22], best, best / fw * 178.0))
    for nm, lx, ly, p in LETTERS:
        print("   letter %s  fig(%.3f, %.3f)  axes x0=%.3f y1=%.3f" % (nm, lx, ly, p.x0, p.y1))
    fig.savefig(os.path.join(OUT, "_dbg_nontight.png"), dpi=600)
    px2cm = 17.8 / fw
    b0 = c_axes[0].get_window_extent(r); b1 = c_axes[1].get_window_extent(r)
    tl = min(t.get_window_extent(r).x0 for t in c_axes[1].get_yticklabels() if t.get_text())
    print("   (c) matrices side %.2f cm each, gap between boxes %.2f cm, tick-label clearance %.2f cm"
          % (b0.width * px2cm, (b1.x0 - b0.x1) * px2cm, (tl - b0.x1) * px2cm))

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "Figure7_v2." + ext), dpi=600, bbox_inches="tight")
plt.close(fig)
print("written:", os.path.join(OUT, "Figure7_v2.png"))
print("panel a metrics:", ", ".join(l for _, l in MET))
