
#
import os
import re
import sys
import joblib
import warnings
import gzip
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import networkx as nx

from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.utils import resample
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("===========================================", flush=True)
print(f"当前工作目录: {BASE_DIR}", flush=True)
print("===========================================", flush=True)


def log(msg):
    print(msg, flush=True)

class NelderMeadEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators
        self.weights_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        y_arr = np.asarray(y)
        unique_classes, class_counts = np.unique(y_arr, return_counts=True)

        if len(unique_classes) < 2:
            raise ValueError("NelderMeadEnsemble 训练数据中只有一个类别，无法训练。")

        min_class_count = int(class_counts.min())
        n_splits = min(3, min_class_count)

        if n_splits < 2:
            self.weights_ = np.ones(len(self.estimators), dtype=float)
            self.weights_ = self.weights_ / self.weights_.sum()
            return self

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        oof_preds = np.zeros(
            (len(y_arr), len(self.classes_), len(self.estimators))
        )

        for i, (name, model) in enumerate(self.estimators):
            oof_preds[:, :, i] = cross_val_predict(
                model,
                X,
                y_arr,
                cv=cv,
                method="predict_proba",
                n_jobs=1
            )

        def objective(w):
            w = np.clip(w, 0, 1)

            if np.sum(w) == 0:
                w = np.ones(len(w))

            w = w / np.sum(w)

            blended = np.average(oof_preds, axis=2, weights=w)
            preds = self.classes_[np.argmax(blended, axis=1)]

            return -f1_score(y_arr, preds, average="macro")

        initial_weights = np.ones(len(self.estimators)) / len(self.estimators)

        res = minimize(
            objective,
            initial_weights,
            method="Nelder-Mead",
            options={"maxiter": 200}
        )

        self.weights_ = np.clip(res.x, 0, 1)

        if np.sum(self.weights_) == 0:
            self.weights_ = np.ones(len(self.estimators), dtype=float)

        self.weights_ = self.weights_ / np.sum(self.weights_)

        return self

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for _, m in self.estimators])
        return np.average(probs, axis=0, weights=self.weights_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

def clean_sample_id(sample_id):
    return re.sub(r"\.fna.*$", "", str(sample_id)).strip()


def sanitize_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")


def clean_gene_name(raw_name):
    """Panaroo 簇名 -> 图与正文中使用的标签。

    修订（Reviewer #1, comment 3）：命名簇去掉末尾的 " cluster" 后缀，直接以基因名
    呈现（如 "tuf1_2 cluster" -> "tuf1_2"）；未获命名的簇保留 Panaroo 的 group 编号。
    """
    if "~~~" in str(raw_name):
        name = f"{str(raw_name).split('~~~')[0]} cluster"
    else:
        name = str(raw_name)
    if name.endswith(" cluster"):
        name = name[: -len(" cluster")]
    return name


def is_tree_like_model(model):
    return model.__class__.__name__ in [
        "LGBMClassifier",
        "XGBClassifier",
        "RandomForestClassifier",
        "XGBRFClassifier"
    ]


def is_rf_model(name, model):
    name_upper = str(name).upper()
    return (
        name_upper in [
            "RF",
            "RANDOMFOREST",
            "RANDOM_FOREST",
            "RANDOM_FOREST_CLASSIFIER"
        ]
        or model.__class__.__name__ in [
            "RandomForestClassifier",
            "ExtraTreesClassifier"
        ]
    )


def is_svm_model(name, model):
    return name.upper() == "SVM" or model.__class__.__name__ == "SVC"


def is_mlp_model(name, model):
    return name.upper() == "MLP" or model.__class__.__name__ == "MLPClassifier"


def savefig_png_pdf(fig, out_prefix, dpi=600):
    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    log(f"✅ 已保存: {png_path}")
    log(f"✅ 已保存: {pdf_path}")


def save_matrix_csv(matrix, row_names, col_names, out_path):
    df = pd.DataFrame(matrix, index=row_names, columns=col_names)
    df.to_csv(out_path)
    return df


def safe_extract_binary_interaction_values(shap_interaction_output):

    if isinstance(shap_interaction_output, list):
        if len(shap_interaction_output) >= 2:
            return np.asarray(shap_interaction_output[1], dtype=float)
        return np.asarray(shap_interaction_output[-1], dtype=float)

    vals = np.asarray(shap_interaction_output, dtype=float)

    if vals.ndim == 4:
        if vals.shape[-1] >= 2:
            return vals[:, :, :, 1]
        return vals[:, :, :, 0]

    return vals


def get_flat_path(filename, required=True):
    path = os.path.join(BASE_DIR, filename)
    if required and not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    return path


def get_host_model_path(host_name):
    return get_flat_path(f"{host_name}_Ensemble__Nelder_Mead.joblib", required=True)


def get_host_top100_joblib_path(host_name):
    return get_flat_path(
        f"{host_name}_Bset_NelderMead_TreeOnly_SHAP_Top100.joblib",
        required=True
    )


def get_host_fig5_weight_path(host_name):
    return get_flat_path(
        f"{host_name}_Final_TreeOnly_SHAP_Weights_After_Failures.csv",
        required=True
    )


def load_fig5_final_tree_model_names_flat(target_host):
    final_weight_path = get_host_fig5_weight_path(target_host)

    df_w = pd.read_csv(final_weight_path)

    if "Model" not in df_w.columns:
        raise ValueError(
            f"{final_weight_path} 中没有 Model 列，当前列名为: {list(df_w.columns)}"
        )

    model_names = df_w["Model"].astype(str).tolist()

    if len(model_names) == 0:
        raise ValueError(f"{target_host} 的 Fig5 最终 Tree-only 模型列表为空。")

    return model_names, df_w


def select_tree_models_from_fig5(
    nelder_model,
    fig5_allowed_model_names,
    max_models=3
):
   
    if not hasattr(nelder_model, "estimators"):
        raise ValueError("当前模型没有 estimators，可能不是 NelderMeadEnsemble。")

    model_lookup = {str(name): model for name, model in nelder_model.estimators}

    selected_names = []
    skipped_records = []

    for name in fig5_allowed_model_names:
        name = str(name)

        if name not in model_lookup:
            skipped_records.append({
                "Model": name,
                "Class": "NA",
                "Reason": "Model name from Fig5 not found in original NelderMead estimators"
            })
            continue

        model = model_lookup[name]
        model_class = model.__class__.__name__

        if is_svm_model(name, model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Reason": "Skip SVM"
            })
            continue

        if is_mlp_model(name, model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Reason": "Skip MLP"
            })
            continue

        if not is_tree_like_model(model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Reason": "Skip non-tree model"
            })
            continue

        selected_names.append(name)

        if len(selected_names) >= max_models:
            break

    if len(selected_names) == 0:
        raise ValueError("没有可用于 Fig6 二阶 SHAP 的树模型。")

    if len(selected_names) < max_models:
        log(f"⚠️ 当前只找到 {len(selected_names)} 个可用树模型，少于预期 {max_models} 个。")

    skipped_df = pd.DataFrame(skipped_records)

    return selected_names, skipped_df


def extract_tree_template_components_from_saved_nelder(
    nelder_model,
    allowed_model_names
):
    if not hasattr(nelder_model, "estimators"):
        raise ValueError("当前模型没有 estimators，可能不是 NelderMeadEnsemble。")

    model_lookup = {str(name): model for name, model in nelder_model.estimators}

    tree_templates = []
    skipped_records = []

    for name in allowed_model_names:
        name = str(name)

        if name not in model_lookup:
            skipped_records.append({
                "Model": name,
                "Class": "NA",
                "Reason": "Selected model not found in original NelderMead estimators"
            })
            continue

        model = model_lookup[name]
        model_class = model.__class__.__name__

        if is_svm_model(name, model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Reason": "Skip SVM"
            })
            continue

        if is_mlp_model(name, model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Reason": "Skip MLP"
            })
            continue

        if not is_tree_like_model(model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Reason": "Skip non-tree model"
            })
            continue

        tree_templates.append((name, model))

    if len(tree_templates) == 0:
        raise ValueError("没有可用的树模型模板。")

    return tree_templates, pd.DataFrame(skipped_records)


def refit_top100_tree_nelder_from_templates(
    tree_templates,
    X_fit_df,
    y_fit,
    random_state=42
):
    fitted_estimators = []

    for name, template_model in tree_templates:
        log(f"   -> 重新训练 Top100 子模型: {name} [{template_model.__class__.__name__}]")
        m = clone(template_model)

        if hasattr(m, "set_params"):
            params = m.get_params()
            if "random_state" in params:
                try:
                    m.set_params(random_state=random_state)
                except Exception:
                    pass

        m.fit(X_fit_df, y_fit)
        fitted_estimators.append((name, m))

    top100_nelder = NelderMeadEnsemble(estimators=fitted_estimators)
    top100_nelder.fit(X_fit_df, y_fit)

    return top100_nelder


def get_tree_components_from_refit_nelder(nelder_model, skip_zero_weight=True):
    if not hasattr(nelder_model, "estimators"):
        raise ValueError("当前模型没有 estimators。")

    if not hasattr(nelder_model, "weights_") or nelder_model.weights_ is None:
        raise ValueError("当前 NelderMead 模型没有 weights_。")

    names = [name for name, _ in nelder_model.estimators]
    models = [model for _, model in nelder_model.estimators]
    weights = np.asarray(nelder_model.weights_, dtype=float)

    kept_names = []
    kept_models = []
    kept_weights = []
    skipped_records = []

    for name, model, weight in zip(names, models, weights):
        model_class = model.__class__.__name__

        if not is_tree_like_model(model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Weight": weight,
                "Reason": "Skip non-tree"
            })
            continue

        if skip_zero_weight and weight <= 0:
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Weight": weight,
                "Reason": "Skip zero weight"
            })
            continue

        kept_names.append(name)
        kept_models.append(model)
        kept_weights.append(weight)

    kept_weights = np.asarray(kept_weights, dtype=float)

    if len(kept_weights) == 0:
        raise ValueError("没有剩余树模型可用于加权。")

    kept_weights = kept_weights / kept_weights.sum()

    used_df = pd.DataFrame({
        "Model": kept_names,
        "Class": [m.__class__.__name__ for m in kept_models],
        "Normalized_Top100_TreeOnly_Weight": kept_weights
    })

    skipped_df = pd.DataFrame(skipped_records)

    return kept_names, kept_models, kept_weights, used_df, skipped_df


def make_tree_explainers(names, models):
    explainers = []

    for name, model in zip(names, models):
        log(f"   -> 创建 TreeExplainer: {name} [{model.__class__.__name__}]")
        # PATCH: explain every tree component on the probability scale so that
        # the permutation null is on the same scale as the observed statistic.
        try:
            explainers.append(
                shap.TreeExplainer(model, model_output="probability")
            )
        except Exception:
            # Random forest is natively on the probability scale; for the
            # gradient-boosted models, never silently fall back to log-odds.
            if model.__class__.__name__ in (
                "LGBMClassifier",
                "XGBClassifier",
                "XGBRFClassifier",
            ):
                raise
            explainers.append(shap.TreeExplainer(model))

    return explainers


def compute_tree_interaction_with_explainer(
    model_name,
    model,
    explainer,
    X_top100_df
):
    if not is_tree_like_model(model):
        raise ValueError(f"{model_name} 不是树模型。")

    raw_interactions = explainer.shap_interaction_values(X_top100_df)

    interactions = safe_extract_binary_interaction_values(raw_interactions)

    if interactions.ndim != 3:
        raise ValueError(f"{model_name} 的 interaction 维度异常: {interactions.shape}")

    if (
        interactions.shape[1] != X_top100_df.shape[1]
        or interactions.shape[2] != X_top100_df.shape[1]
    ):
        raise ValueError(
            f"{model_name} interaction shape 与 Top100 特征数不一致: "
            f"{interactions.shape}, X={X_top100_df.shape}"
        )

    if not np.isfinite(interactions).all():
        raise ValueError(f"{model_name} 的 interaction 中存在 NaN 或 Inf。")

    mean_signed = np.mean(interactions, axis=0)
    mean_abs = np.mean(np.abs(interactions), axis=0)

    main_effects = np.mean(
        np.abs(np.diagonal(interactions, axis1=1, axis2=2)),
        axis=0
    )

    np.fill_diagonal(mean_signed, 0)
    np.fill_diagonal(mean_abs, 0)

    return mean_signed, mean_abs, main_effects


def compute_weighted_interactions_from_fixed_components(
    names,
    models,
    weights,
    explainers,
    X_top100_df,
    context_label="main"
):
    signed_list = []
    abs_list = []
    main_effect_list = []
    interaction_names = []
    interaction_weights = []
    skipped_records = []

    for name, model, weight, explainer in zip(names, models, weights, explainers):
        try:
            log(
                f"      -> [{context_label}] 计算 {name} "
                f"[{model.__class__.__name__}] 二阶 SHAP interaction | n={len(X_top100_df)}"
            )

            mean_signed, mean_abs, main_effects = compute_tree_interaction_with_explainer(
                model_name=name,
                model=model,
                explainer=explainer,
                X_top100_df=X_top100_df
            )

            signed_list.append(mean_signed)
            abs_list.append(mean_abs)
            main_effect_list.append(main_effects)
            interaction_names.append(name)
            interaction_weights.append(weight)

        except Exception as e:
            skipped_records.append({
                "Model": name,
                "Class": model.__class__.__name__,
                "Weight": weight,
                "Reason": f"Interaction failed: {e}"
            })

    if len(signed_list) == 0:
        raise ValueError(
            f"{context_label}: 没有任何树模型成功计算 Top100 二阶 SHAP interaction。"
        )

    interaction_weights = np.asarray(interaction_weights, dtype=float)
    interaction_weights = interaction_weights / interaction_weights.sum()

    weighted_signed = np.average(
        np.stack(signed_list, axis=0),
        axis=0,
        weights=interaction_weights
    )

    weighted_abs = np.average(
        np.stack(abs_list, axis=0),
        axis=0,
        weights=interaction_weights
    )

    weighted_main_effects = np.average(
        np.stack(main_effect_list, axis=0),
        axis=0,
        weights=interaction_weights
    )

    interaction_weight_df = pd.DataFrame({
        "Model": interaction_names,
        "Final_Top100_Interaction_Weight": interaction_weights
    })

    interaction_skipped_df = pd.DataFrame(skipped_records)

    return (
        weighted_signed,
        weighted_abs,
        weighted_main_effects,
        interaction_weight_df,
        interaction_skipped_df
    )


def predict_proba_top100_tree_nelder(top100_nelder_model, X_top100_df):
    return top100_nelder_model.predict_proba(X_top100_df)[:, 1]


def draw_signed_interaction_heatmap(
    matrix,
    labels,
    title,
    out_prefix,
    figsize=(10, 8),
    color_quantile=0.90
):
    values = np.asarray(matrix, dtype=float)
    abs_values = np.abs(values[np.isfinite(values)])

    if len(abs_values) == 0:
        vmax = 1e-6
    else:
        vmax = np.quantile(abs_values, color_quantile)

        if vmax <= 0 or not np.isfinite(vmax):
            vmax = np.max(abs_values)

        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1e-6

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        values,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdBu_r",
        norm=norm,
        linewidths=0.25,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Signed SHAP interaction"}
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=10
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=10
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=10
    )

    plt.tight_layout()
    savefig_png_pdf(fig, out_prefix, dpi=600)
    plt.close(fig)




def append_csv_gz(df, out_path):
    """
    追加保存 csv.gz。
    pandas 后续可以直接用 pd.read_csv('xxx.csv.gz') 读取。
    """
    write_header = not os.path.exists(out_path)

    with gzip.open(out_path, "at", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, header=write_header)


def make_bootstrap_pair_rank_df(
    abs_matrix,
    signed_matrix,
    selected_features,
    clean_features,
    bootstrap_id
):
    """

    """
    abs_matrix = np.asarray(abs_matrix, dtype=float)
    signed_matrix = np.asarray(signed_matrix, dtype=float)

    if abs_matrix.ndim != 2 or signed_matrix.ndim != 2:
        raise ValueError(
            f"interaction matrix 必须是二维矩阵，当前 abs={abs_matrix.shape}, signed={signed_matrix.shape}"
        )

    if abs_matrix.shape != signed_matrix.shape:
        raise ValueError(
            f"abs_matrix 和 signed_matrix 形状不一致: abs={abs_matrix.shape}, signed={signed_matrix.shape}"
        )

    upper_i, upper_j = np.triu_indices_from(abs_matrix, k=1)

    abs_vals = abs_matrix[upper_i, upper_j]
    signed_vals = signed_matrix[upper_i, upper_j]

    finite_mask = np.isfinite(abs_vals) & np.isfinite(signed_vals)
    upper_i = upper_i[finite_mask]
    upper_j = upper_j[finite_mask]
    abs_vals = abs_vals[finite_mask]
    signed_vals = signed_vals[finite_mask]

    order = np.argsort(abs_vals)[::-1]

    upper_i = upper_i[order]
    upper_j = upper_j[order]
    abs_vals = abs_vals[order]
    signed_vals = signed_vals[order]

    raw_arr = np.asarray(selected_features, dtype=object)
    clean_arr = np.asarray(clean_features, dtype=object)

    gene_a_raw = raw_arr[upper_i]
    gene_b_raw = raw_arr[upper_j]
    gene_a_clean = clean_arr[upper_i]
    gene_b_clean = clean_arr[upper_j]

    n_pairs = len(abs_vals)
    ranks = np.arange(1, n_pairs + 1)

    df_rank = pd.DataFrame({
        "Bootstrap": int(bootstrap_id),
        "Rank": ranks,
        "Rank_Percent": ranks / n_pairs * 100.0 if n_pairs > 0 else np.nan,
        "Gene_A_Raw": gene_a_raw,
        "Gene_B_Raw": gene_b_raw,
        "Gene_A": gene_a_clean,
        "Gene_B": gene_b_clean,
        "Display_Name": [
            f"{a} & {b}" for a, b in zip(gene_a_clean, gene_b_clean)
        ],
        "Signed_Interaction": signed_vals,
        "Abs_Interaction": abs_vals,
        "Sign": np.where(signed_vals >= 0, 1, -1)
    })

    return df_rank


def update_default_topk_counters_from_rank_df(
    pair_rank_df,
    top_k,
    pair_frequency,
    pair_signs
):
    
    top_k = int(top_k)

    if top_k <= 0:
        return

    sub = pair_rank_df[pair_rank_df["Rank"] <= top_k].copy()

    for row in sub[["Gene_A_Raw", "Gene_B_Raw", "Sign"]].itertuples(index=False):
        pair = (str(row.Gene_A_Raw), str(row.Gene_B_Raw))
        sign = int(row.Sign)

        pair_frequency[pair] = pair_frequency.get(pair, 0) + 1

        if pair not in pair_signs:
            pair_signs[pair] = []

        pair_signs[pair].append(sign)


def build_stability_df_from_counters(
    pair_frequency,
    pair_signs,
    valid_bootstrap_count,
    selected_features,
    clean_features,
    mean_signed_interactions_main,
    mean_abs_interactions_main,
    main_names,
    top_k
):
   
    if valid_bootstrap_count <= 0:
        return pd.DataFrame()

    feature_to_idx = {f: i for i, f in enumerate(selected_features)}
    clean_lookup = {f: c for f, c in zip(selected_features, clean_features)}

    records = []

    for pair, freq in pair_frequency.items():
        gene_a_raw, gene_b_raw = pair

        if gene_a_raw not in feature_to_idx or gene_b_raw not in feature_to_idx:
            continue

        signs = np.asarray(pair_signs.get(pair, []), dtype=int)

        if len(signs) == 0:
            continue

        pos_count = int(np.sum(signs > 0))
        neg_count = int(np.sum(signs < 0))

        major_sign = 1 if pos_count >= neg_count else -1
        direction_consistency = max(pos_count, neg_count) / len(signs) * 100.0

        idx_a = feature_to_idx[gene_a_raw]
        idx_b = feature_to_idx[gene_b_raw]

        gene_a_clean = clean_lookup[gene_a_raw]
        gene_b_clean = clean_lookup[gene_b_raw]

        records.append({
            "Bootstrap_TopK": int(top_k),
            "Pair": pair,
            "Gene_A_Raw": gene_a_raw,
            "Gene_B_Raw": gene_b_raw,
            "Gene_A": gene_a_clean,
            "Gene_B": gene_b_clean,
            "Display_Name": gene_a_clean + " & " + gene_b_clean,
            "Selection_Count": int(freq),
            "Valid_Bootstrap_Count": int(valid_bootstrap_count),
            "Stability(%)": freq / valid_bootstrap_count * 100.0,
            "Positive_Count": pos_count,
            "Negative_Count": neg_count,
            "Major_Sign": int(major_sign),
            "Direction_Consistency(%)": direction_consistency,
            "Main_Signed_Interaction": float(mean_signed_interactions_main[idx_a, idx_b]),
            "Main_Abs_Interaction": float(mean_abs_interactions_main[idx_a, idx_b]),
            "Fig6_Models": ";".join(main_names)
        })

    df = pd.DataFrame(records)

    if len(df) == 0:
        return df

    df = df.sort_values(
        by=[
            "Stability(%)",
            "Direction_Consistency(%)",
            "Main_Abs_Interaction"
        ],
        ascending=False
    ).reset_index(drop=True)

    return df



TARGET_HOSTS = ["human", "pig", "chicken"]

TOP_N_GENES = 100

N_PERMUTATIONS = 1000

SAVE_FULL_PERMUTATION_PAIR_VALUES = True

FULL_PAIR_RANKING_MAX_RANK = None

OPTIMIZE_NELDER_WEIGHTS_IN_PERMUTATION = False

FAST_WEIGHT_MODE = "equal"

INTERACTION_SAMPLE_SIZE = None

PERMUTATION_INTERACTION_SAMPLE_SIZE = None

# permutation 阈值分位数
NULL_QUANTILES = [0.90, 0.95, 0.99]


# =========================================================
# 5. permutation 专用辅助函数
# =========================================================
def make_pair_value_df_from_matrices(
    abs_matrix,
    signed_matrix,
    selected_features,
    clean_features,
    iteration_id,
    iteration_col_name="Permutation"
):
    
    abs_matrix = np.asarray(abs_matrix, dtype=float)
    signed_matrix = np.asarray(signed_matrix, dtype=float)

    if abs_matrix.ndim != 2 or signed_matrix.ndim != 2:
        raise ValueError(
            f"interaction matrix 必须是二维矩阵，当前 abs={abs_matrix.shape}, signed={signed_matrix.shape}"
        )

    if abs_matrix.shape != signed_matrix.shape:
        raise ValueError(
            f"abs_matrix 和 signed_matrix 形状不一致: abs={abs_matrix.shape}, signed={signed_matrix.shape}"
        )

    upper_i, upper_j = np.triu_indices_from(abs_matrix, k=1)

    abs_vals = abs_matrix[upper_i, upper_j]
    signed_vals = signed_matrix[upper_i, upper_j]

    finite_mask = np.isfinite(abs_vals) & np.isfinite(signed_vals)
    upper_i = upper_i[finite_mask]
    upper_j = upper_j[finite_mask]
    abs_vals = abs_vals[finite_mask]
    signed_vals = signed_vals[finite_mask]

    order = np.argsort(abs_vals)[::-1]

    raw_arr = np.asarray(selected_features, dtype=object)
    clean_arr = np.asarray(clean_features, dtype=object)

    n_pairs = len(abs_vals)

    # Pair_ID 是按照原始上三角顺序来的，便于不同 permutation 对齐
    pair_id = np.arange(len(np.triu_indices_from(abs_matrix, k=1)[0]))[finite_mask]

    df = pd.DataFrame({
        iteration_col_name: int(iteration_id),
        "Pair_ID": pair_id,
        "Gene_A_Raw": raw_arr[upper_i],
        "Gene_B_Raw": raw_arr[upper_j],
        "Gene_A": clean_arr[upper_i],
        "Gene_B": clean_arr[upper_j],
        "Signed_Interaction": signed_vals,
        "Abs_Interaction": abs_vals,
        "Sign": np.where(signed_vals >= 0, 1, -1)
    })

    df["Display_Name"] = df["Gene_A"].astype(str) + " & " + df["Gene_B"].astype(str)

    # 按 Abs_Interaction 从大到小给 rank
    rank_series = pd.Series(np.empty(n_pairs, dtype=int), index=df.index)
    rank_series.iloc[order] = np.arange(1, n_pairs + 1)
    df["Rank"] = rank_series.values
    df["Rank_Percent"] = df["Rank"] / n_pairs * 100.0 if n_pairs > 0 else np.nan

    # 保存时按 Rank 排序，阅读方便
    df = df.sort_values("Rank", ascending=True).reset_index(drop=True)

    return df


def make_pair_map(selected_features, clean_features):
    """
    生成固定 Pair_ID 对照表。
    """
    n = len(selected_features)
    upper_i, upper_j = np.triu_indices(n, k=1)

    raw_arr = np.asarray(selected_features, dtype=object)
    clean_arr = np.asarray(clean_features, dtype=object)

    df = pd.DataFrame({
        "Pair_ID": np.arange(len(upper_i), dtype=int),
        "Gene_A_Raw": raw_arr[upper_i],
        "Gene_B_Raw": raw_arr[upper_j],
        "Gene_A": clean_arr[upper_i],
        "Gene_B": clean_arr[upper_j]
    })

    df["Display_Name"] = df["Gene_A"].astype(str) + " & " + df["Gene_B"].astype(str)

    return df


def extract_abs_vector_by_pair_id(abs_matrix):

    abs_matrix = np.asarray(abs_matrix, dtype=float)
    upper_i, upper_j = np.triu_indices_from(abs_matrix, k=1)
    return abs_matrix[upper_i, upper_j]


def fit_top100_tree_model_for_permutation(
    tree_templates,
    X_fit_df,
    y_fit,
    random_state,
    optimize_nelder_weights=True,
    fast_weight_mode="equal",
    saved_nelder_model=None,
    allowed_model_names=None
):

    fitted_estimators = []

    for name, template_model in tree_templates:
        log(f"      -> 训练 permutation 子模型: {name} [{template_model.__class__.__name__}]")
        m = clone(template_model)

        if hasattr(m, "set_params"):
            params = m.get_params()
            if "random_state" in params:
                try:
                    m.set_params(random_state=random_state)
                except Exception:
                    pass

        m.fit(X_fit_df, y_fit)
        fitted_estimators.append((name, m))

    perm_nelder = NelderMeadEnsemble(estimators=fitted_estimators)

    if optimize_nelder_weights:
        perm_nelder.fit(X_fit_df, y_fit)
    else:
        # 手动设置 classes_ 和 weights_
        perm_nelder.classes_ = np.unique(np.asarray(y_fit))

        if fast_weight_mode == "template" and saved_nelder_model is not None:
            try:
                saved_names = [str(n) for n, _ in saved_nelder_model.estimators]
                saved_weights = np.asarray(saved_nelder_model.weights_, dtype=float)
                selected_weights = []

                for name, _ in fitted_estimators:
                    if str(name) in saved_names:
                        idx = saved_names.index(str(name))
                        selected_weights.append(saved_weights[idx])
                    else:
                        selected_weights.append(0.0)

                selected_weights = np.asarray(selected_weights, dtype=float)

                if selected_weights.sum() <= 0:
                    selected_weights = np.ones(len(fitted_estimators), dtype=float)

                perm_nelder.weights_ = selected_weights / selected_weights.sum()

            except Exception:
                perm_nelder.weights_ = np.ones(len(fitted_estimators), dtype=float)
                perm_nelder.weights_ = perm_nelder.weights_ / perm_nelder.weights_.sum()
        else:
            perm_nelder.weights_ = np.ones(len(fitted_estimators), dtype=float)
            perm_nelder.weights_ = perm_nelder.weights_ / perm_nelder.weights_.sum()

    return perm_nelder


def summarize_permutation_thresholds(
    pair_map_df,
    null_abs_matrix,
    observed_df=None,
    quantiles=(0.90, 0.95, 0.99)
):
    
    null_abs_matrix = np.asarray(null_abs_matrix, dtype=float)

    if null_abs_matrix.ndim != 2:
        raise ValueError(f"null_abs_matrix 必须是二维，当前 shape={null_abs_matrix.shape}")

    out = pair_map_df.copy()

    out["Null_N"] = np.sum(np.isfinite(null_abs_matrix), axis=0)
    out["Null_Mean_Abs"] = np.nanmean(null_abs_matrix, axis=0)
    out["Null_Median_Abs"] = np.nanmedian(null_abs_matrix, axis=0)
    out["Null_Std_Abs"] = np.nanstd(null_abs_matrix, axis=0)

    for q in quantiles:
        q_name = f"Null_Q{int(q * 100)}_Abs"
        out[q_name] = np.nanquantile(null_abs_matrix, q, axis=0)

    if observed_df is not None and len(observed_df) > 0:
        obs_small = observed_df[
            [
                "Pair_ID",
                "Signed_Interaction",
                "Abs_Interaction",
                "Sign",
                "Rank",
                "Rank_Percent"
            ]
        ].rename(columns={
            "Signed_Interaction": "Observed_Signed_Interaction",
            "Abs_Interaction": "Observed_Abs_Interaction",
            "Sign": "Observed_Sign",
            "Rank": "Observed_Rank",
            "Rank_Percent": "Observed_Rank_Percent"
        })

        out = out.merge(obs_small, on="Pair_ID", how="left")

        for q in quantiles:
            q_label = int(q * 100)
            q_col = f"Null_Q{q_label}_Abs"
            out[f"Observed_Exceeds_Q{q_label}"] = out["Observed_Abs_Interaction"] > out[q_col]

        # 经验 P 值：null >= observed 的比例，使用 +1 修正
        # p = (count(null >= observed) + 1) / (N_perm + 1)
        obs_vec = out["Observed_Abs_Interaction"].values.astype(float)
        p_vals = []
        for j in range(null_abs_matrix.shape[1]):
            valid_null = null_abs_matrix[:, j]
            valid_null = valid_null[np.isfinite(valid_null)]
            if len(valid_null) == 0 or not np.isfinite(obs_vec[j]):
                p_vals.append(np.nan)
            else:
                p_vals.append((np.sum(valid_null >= obs_vec[j]) + 1) / (len(valid_null) + 1))
        out["Permutation_Empirical_P"] = p_vals

    return out



log("📦 正在读取基因组存在/缺失矩阵与元数据...")

pa_path = get_flat_path("gene_presence_absence_cleaned.csv", required=True)
meta_path = get_flat_path("file_with_host_date_ip.csv", required=True)

pa_data = pd.read_csv(pa_path, index_col=0)
X_all = pa_data.T
X_all.index = X_all.index.to_series().apply(clean_sample_id)

meta_df = pd.read_csv(meta_path)
id_col = "ID" if "ID" in meta_df.columns else meta_df.columns[0]
meta_df[id_col] = meta_df[id_col].astype(str).apply(clean_sample_id)
meta_df = meta_df.set_index(id_col)

common_ids = X_all.index.intersection(meta_df.index)
X_all = X_all.loc[common_ids]
meta_df = meta_df.loc[common_ids]

meta_df["TIME_PERIOD"] = meta_df["DATE"].astype(str).str.strip()
meta_df["IP_CLEAN"] = meta_df["IP"].astype(str).str.strip()

counts = meta_df.groupby(["TIME_PERIOD", "IP_CLEAN"]).size()
valid_schemes = counts[counts > 200].index.tolist()

if len(valid_schemes) == 0:
    raise ValueError("没有找到 counts > 200 的 valid_schemes。请检查 DATE / IP 列。")

SCHEME_IDX = 0
if SCHEME_IDX >= len(valid_schemes):
    raise ValueError(f"SCHEME_IDX={SCHEME_IDX} 超出 valid_schemes 范围。")

final_scheme = valid_schemes[SCHEME_IDX]
time_p, ip_p = final_scheme

is_a = (meta_df["TIME_PERIOD"] == time_p) & (meta_df["IP_CLEAN"] == ip_p)

X_A_full = X_all.loc[is_a]
X_B_full = X_all.loc[~is_a]

host_col = "Host" if "Host" in meta_df.columns else meta_df.columns[1]

log(f"✅ Scheme {SCHEME_IDX}: {time_p} | {ip_p}")
log(f"✅ A集: {len(X_A_full)} 样本")
log(f"✅ B集: {len(X_B_full)} 样本")
log("📌 当前 permutation null test 只使用 B集。")


out_dir = os.path.join(
    BASE_DIR,
    "Fig6_Bset_Top100_PermutationNull_chicken"
)
os.makedirs(out_dir, exist_ok=True)

weight_dir = os.path.join(out_dir, "Weights")
os.makedirs(weight_dir, exist_ok=True)

table_dir = os.path.join(out_dir, "Tables")
os.makedirs(table_dir, exist_ok=True)

observed_dir = os.path.join(out_dir, "Observed_Interactions")
os.makedirs(observed_dir, exist_ok=True)

permutation_values_dir = os.path.join(out_dir, "Permutation_AllPair_Values")
os.makedirs(permutation_values_dir, exist_ok=True)

threshold_dir = os.path.join(out_dir, "Permutation_Thresholds")
os.makedirs(threshold_dir, exist_ok=True)

model_dir = os.path.join(out_dir, "Observed_MainModel")
os.makedirs(model_dir, exist_ok=True)

pairmap_dir = os.path.join(out_dir, "Pair_Maps")
os.makedirs(pairmap_dir, exist_ok=True)

all_host_summary_records = []


for target_host in TARGET_HOSTS:
    host_start_time = time.time()

    log(f"\n{'=' * 80}")
    log(f"🎯 正在进行 permutation null test: 【{target_host.upper()}】")
    log(f"{'=' * 80}")

    y_B = (
        meta_df.loc[X_B_full.index, host_col]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(target_host.lower())
        .astype(int)
    )

    if y_B.nunique() < 2:
        log(f"⚠️ {target_host} 的 B集标签只有一个类别，跳过。")
        continue

    # -----------------------------------------------------
    # Step 1: 读取 Top100 genes
    # -----------------------------------------------------
    shap_top_joblib_path = get_host_top100_joblib_path(target_host)

    selected_features = list(joblib.load(shap_top_joblib_path))
    selected_features = [f for f in selected_features if f in X_B_full.columns]

    if len(selected_features) == 0:
        log(f"⚠️ {target_host} selected_features 为空，跳过。")
        continue

    selected_features = selected_features[:TOP_N_GENES]
    clean_features = [clean_gene_name(f) for f in selected_features]
    X_B_top100 = X_B_full[selected_features].copy()

    used_top100_df = pd.DataFrame({
        "Rank": np.arange(1, len(selected_features) + 1),
        "Feature": selected_features,
        "Clean_Name": clean_features
    })

    used_top100_df.to_csv(
        os.path.join(table_dir, f"{target_host}_Used_Top100_Features.csv"),
        index=False
    )

    log(f"📥 已加载 {target_host} 的 B集 Top{len(selected_features)} genes。")
    log(f"✅ X_B_top100 shape = {X_B_top100.shape}")

    pair_map_df = make_pair_map(selected_features, clean_features)
    pair_map_path = os.path.join(pairmap_dir, f"{target_host}_Pair_Map_Top{len(selected_features)}.csv")
    pair_map_df.to_csv(pair_map_path, index=False)
    log(f"✅ Pair map 已保存: {pair_map_path}")

    
    try:
        fig5_allowed_tree_names, fig5_final_weight_df = load_fig5_final_tree_model_names_flat(
            target_host=target_host
        )

        fig5_final_weight_df.to_csv(
            os.path.join(weight_dir, f"{target_host}_Fig5_Final_TreeOnly_Models_Original.csv"),
            index=False
        )

        log("📌 Fig5 实际参与 Tree-only SHAP 的模型：")
        for m in fig5_allowed_tree_names:
            log(f"   - {m}")

    except Exception as e:
        log(f"❌ 读取 Fig5 最终 Tree-only 模型列表失败: {e}")
        continue

    # -----------------------------------------------------
    # Step 3: 读取原始保存的 NelderMead 模型
    # -----------------------------------------------------
    model_path = get_host_model_path(target_host)
    log(f"✅ 找到原始模型文件: {model_path}")

    bundle = joblib.load(model_path)
    saved_nelder_model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    
    try:
        allowed_tree_names_for_perm, perm_model_selection_skipped_df = (
            select_tree_models_from_fig5(
                nelder_model=saved_nelder_model,
                fig5_allowed_model_names=fig5_allowed_tree_names,
                max_models=3
            )
        )

        perm_model_selection_df = pd.DataFrame({
            "Selected_Model_For_Permutation": allowed_tree_names_for_perm,
            "Rule": "Tree models from Fig5 final Tree-only SHAP"
        })

        perm_model_selection_df.to_csv(
            os.path.join(weight_dir, f"{target_host}_Permutation_Selected_TreeModels.csv"),
            index=False
        )

        if len(perm_model_selection_skipped_df) > 0:
            perm_model_selection_skipped_df.to_csv(
                os.path.join(weight_dir, f"{target_host}_Permutation_ModelSelection_Skipped_Models.csv"),
                index=False
            )

        log("📌 Permutation 二阶 SHAP 将使用以下树模型：")
        for m in allowed_tree_names_for_perm:
            log(f"   - {m}")

    except Exception as e:
        log(f"❌ 选择树模型失败: {e}")
        continue

    
    try:
        tree_templates, template_skipped_df = extract_tree_template_components_from_saved_nelder(
            nelder_model=saved_nelder_model,
            allowed_model_names=allowed_tree_names_for_perm
        )
    except Exception as e:
        log(f"❌ 提取树模型模板失败: {e}")
        continue

    template_skipped_df.to_csv(
        os.path.join(weight_dir, f"{target_host}_Template_Skipped_Models.csv"),
        index=False
    )

    log("✅ 从原始 NelderMead 中提取到的树模型模板：")
    for name, model in tree_templates:
        log(f"   - {name}: {model.__class__.__name__}")

    
    log("🖥️ 正在使用真实 y_B 训练 observed 主模型...")

    try:
        observed_nelder = refit_top100_tree_nelder_from_templates(
            tree_templates=tree_templates,
            X_fit_df=X_B_top100,
            y_fit=y_B,
            random_state=42
        )

        (
            observed_names,
            observed_models,
            observed_weights,
            observed_used_df,
            observed_skipped_df
        ) = get_tree_components_from_refit_nelder(
            observed_nelder,
            skip_zero_weight=True
        )

    except Exception as e:
        log(f"❌ {target_host} observed 主模型训练/权重提取失败: {e}")
        continue

    observed_used_df.to_csv(
        os.path.join(weight_dir, f"{target_host}_Observed_NelderMead_TreeOnly_Weights.csv"),
        index=False
    )

    log("✅ observed 主模型树模型权重：")
    log(observed_used_df.to_string(index=False))

    joblib.dump(
        {
            "target_host": target_host,
            "selected_features": selected_features,
            "clean_features": clean_features,
            "observed_nelder": observed_nelder,
            "observed_weights": observed_used_df,
            "interaction_models": observed_names,
        },
        os.path.join(model_dir, f"{target_host}_Observed_MainModel.joblib"),
        compress=3
    )

    
    if INTERACTION_SAMPLE_SIZE is not None and len(X_B_top100) > INTERACTION_SAMPLE_SIZE:
        X_observed_interaction = X_B_top100.sample(
            n=INTERACTION_SAMPLE_SIZE,
            random_state=42
        )
        log(f"⚠️ observed interaction 仅使用 {len(X_observed_interaction)} 个 B集样本。")
    else:
        X_observed_interaction = X_B_top100

    log("⚡ 正在计算 observed 主模型 Top100 加权二阶 SHAP interaction...")

    try:
        observed_explainers = make_tree_explainers(observed_names, observed_models)

        (
            observed_signed,
            observed_abs,
            observed_main_effects,
            observed_interaction_weight_df,
            observed_interaction_skipped_df
        ) = compute_weighted_interactions_from_fixed_components(
            names=observed_names,
            models=observed_models,
            weights=observed_weights,
            explainers=observed_explainers,
            X_top100_df=X_observed_interaction,
            context_label=f"{target_host}_observed"
        )

    except Exception as e:
        log(f"❌ {target_host} observed interaction 计算失败: {e}")
        continue

    observed_pair_df = make_pair_value_df_from_matrices(
        abs_matrix=observed_abs,
        signed_matrix=observed_signed,
        selected_features=selected_features,
        clean_features=clean_features,
        iteration_id=0,
        iteration_col_name="Observed"
    )

    observed_pair_path = os.path.join(
        observed_dir,
        f"{target_host}_Observed_AllPair_Interactions.csv"
    )
    observed_pair_df.to_csv(observed_pair_path, index=False)
    log(f"✅ observed pair interaction 已保存: {observed_pair_path}")

   
    log(f"🔀 开始 {N_PERMUTATIONS} 次 label permutation null test...")

    permutation_values_path = os.path.join(
        permutation_values_dir,
        f"{target_host}_Permutation_AllPair_Values.csv.gz"
    )

    if os.path.exists(permutation_values_path):
        os.remove(permutation_values_path)

    permutation_summary_records = []
    null_abs_vectors = []

    y_B_arr = np.asarray(y_B, dtype=int)

    for p in range(N_PERMUTATIONS):
        perm_start = time.time()
        perm_id = p + 1

        log(f"\n   ▶ Permutation {perm_id}/{N_PERMUTATIONS} 开始...")

        rng = np.random.default_rng(seed=10000 + p)
        y_perm = rng.permutation(y_B_arr)

        # 防御性检查：理论上 permutation 后类别数不变
        if len(np.unique(y_perm)) < 2:
            log(f"   ⚠️ Permutation {perm_id}: 打乱后只有一个类别，跳过。")
            permutation_summary_records.append({
                "Permutation": perm_id,
                "Status": "Skipped: only one class after permutation",
                "Elapsed_Seconds": time.time() - perm_start
            })
            continue

        try:
            perm_nelder = fit_top100_tree_model_for_permutation(
                tree_templates=tree_templates,
                X_fit_df=X_B_top100,
                y_fit=y_perm,
                random_state=10000 + p,
                optimize_nelder_weights=OPTIMIZE_NELDER_WEIGHTS_IN_PERMUTATION,
                fast_weight_mode=FAST_WEIGHT_MODE,
                saved_nelder_model=saved_nelder_model,
                allowed_model_names=allowed_tree_names_for_perm
            )

            (
                perm_names,
                perm_models,
                perm_weights,
                perm_used_df,
                perm_skipped_df
            ) = get_tree_components_from_refit_nelder(
                perm_nelder,
                skip_zero_weight=True
            )

            if PERMUTATION_INTERACTION_SAMPLE_SIZE is not None and len(X_B_top100) > PERMUTATION_INTERACTION_SAMPLE_SIZE:
                X_perm_interaction = X_B_top100.sample(
                    n=PERMUTATION_INTERACTION_SAMPLE_SIZE,
                    random_state=10000 + p
                )
                n_interaction_samples = len(X_perm_interaction)
            else:
                X_perm_interaction = X_B_top100
                n_interaction_samples = len(X_perm_interaction)

            perm_explainers = make_tree_explainers(perm_names, perm_models)

            (
                perm_signed,
                perm_abs,
                perm_main_effects,
                perm_interaction_weight_df,
                perm_interaction_skipped_df
            ) = compute_weighted_interactions_from_fixed_components(
                names=perm_names,
                models=perm_models,
                weights=perm_weights,
                explainers=perm_explainers,
                X_top100_df=X_perm_interaction,
                context_label=f"{target_host}_permutation_{perm_id}"
            )

            # 以固定 Pair_ID 顺序保存 null abs 向量，用于最后算每个 pair 的 Q90/Q95/Q99
            null_abs_vectors.append(extract_abs_vector_by_pair_id(perm_abs))

            # 保存本次 permutation 每个 pair 的具体 interaction 值
            perm_pair_df = make_pair_value_df_from_matrices(
                abs_matrix=perm_abs,
                signed_matrix=perm_signed,
                selected_features=selected_features,
                clean_features=clean_features,
                iteration_id=perm_id,
                iteration_col_name="Permutation"
            )

            if SAVE_FULL_PERMUTATION_PAIR_VALUES:
                if FULL_PAIR_RANKING_MAX_RANK is not None:
                    save_perm_df = perm_pair_df[
                        perm_pair_df["Rank"] <= FULL_PAIR_RANKING_MAX_RANK
                    ].copy()
                else:
                    save_perm_df = perm_pair_df

                append_csv_gz(save_perm_df, permutation_values_path)

            elapsed = time.time() - perm_start

            permutation_summary_records.append({
                "Permutation": perm_id,
                "Status": "OK",
                "Elapsed_Seconds": elapsed,
                "N_B_samples": len(X_B_top100),
                "N_interaction_samples": n_interaction_samples,
                "N_pairs": len(perm_pair_df),
                "N_models_for_interaction": len(perm_interaction_weight_df),
                "Models": ";".join(perm_interaction_weight_df["Model"].astype(str).tolist()),
                "Weights": ";".join([str(x) for x in np.asarray(perm_weights, dtype=float)]),
                "Optimize_Nelder_Weights": OPTIMIZE_NELDER_WEIGHTS_IN_PERMUTATION,
                "Saved_Full_Permutation_Values": SAVE_FULL_PERMUTATION_PAIR_VALUES,
                "Permutation_Values_File": permutation_values_path
            })

            log(
                f"   ✅ Permutation {perm_id}/{N_PERMUTATIONS} 完成 | "
                f"time = {elapsed:.2f} sec | pairs = {len(perm_pair_df)}"
            )

        except Exception as e:
            elapsed = time.time() - perm_start
            log(
                f"   ❌ Permutation {perm_id}/{N_PERMUTATIONS} 失败 | "
                f"time = {elapsed:.2f} sec | 原因: {e}"
            )

            permutation_summary_records.append({
                "Permutation": perm_id,
                "Status": f"Failed: {e}",
                "Elapsed_Seconds": elapsed,
                "Optimize_Nelder_Weights": OPTIMIZE_NELDER_WEIGHTS_IN_PERMUTATION
            })

            continue

    
    permutation_summary_df = pd.DataFrame(permutation_summary_records)

    summary_path = os.path.join(
        table_dir,
        f"{target_host}_Permutation_Run_Summary.csv"
    )

    permutation_summary_df.to_csv(summary_path, index=False)
    log(f"✅ permutation summary 已保存: {summary_path}")

    if len(null_abs_vectors) == 0:
        log(f"⚠️ {target_host}: 没有成功的 permutation，无法计算 null threshold。")
        all_host_summary_records.append({
            "Host": target_host,
            "Status": "No successful permutation",
            "N_Successful_Permutations": 0
        })
        continue

    null_abs_matrix = np.vstack(null_abs_vectors)

    
    null_matrix_path = os.path.join(
        threshold_dir,
        f"{target_host}_Permutation_Null_AbsInteraction_Matrix.npy"
    )
    np.save(null_matrix_path, null_abs_matrix)
    log(f"✅ null abs interaction matrix 已保存: {null_matrix_path}")

    threshold_df = summarize_permutation_thresholds(
        pair_map_df=pair_map_df,
        null_abs_matrix=null_abs_matrix,
        observed_df=observed_pair_df,
        quantiles=NULL_QUANTILES
    )

    threshold_path = os.path.join(
        threshold_dir,
        f"{target_host}_Permutation_Q90_Q95_Q99_Thresholds_ByPair.csv"
    )

    threshold_df.to_csv(threshold_path, index=False)

    log(f"✅ permutation Q90/Q95/Q99 threshold 已保存: {threshold_path}")

    host_elapsed = time.time() - host_start_time

    success_count = int((permutation_summary_df["Status"] == "OK").sum()) if "Status" in permutation_summary_df.columns else 0

    all_host_summary_records.append({
        "Host": target_host,
        "Status": "OK",
        "N_Top100": len(selected_features),
        "N_Pairs": len(pair_map_df),
        "N_Permutations_Planned": N_PERMUTATIONS,
        "N_Successful_Permutations": success_count,
        "Optimize_Nelder_Weights": OPTIMIZE_NELDER_WEIGHTS_IN_PERMUTATION,
        "Selected_Models": ";".join(allowed_tree_names_for_perm),
        "Observed_File": observed_pair_path,
        "Permutation_Values_File": permutation_values_path,
        "Threshold_File": threshold_path,
        "Null_Matrix_NPY": null_matrix_path,
        "Elapsed_Seconds": host_elapsed
    })

    log(
        f"✅ {target_host} permutation null test 完成 | "
        f"successful={success_count}/{N_PERMUTATIONS} | "
        f"time={host_elapsed:.2f} sec"
    )


summary_df = pd.DataFrame(all_host_summary_records)

summary_path = os.path.join(
    table_dir,
    "Fig6_All_Hosts_Permutation_Null_Run_Summary.csv"
)

summary_df.to_csv(summary_path, index=False)
