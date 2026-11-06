
import os

import re

import sys

import gzip

import shutil

import joblib

import warnings


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

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier


warnings.filterwarnings("ignore")


try:

    sys.stdout.reconfigure(line_buffering=True)

    sys.stderr.reconfigure(line_buffering=True)

except Exception:

    pass








SCRIPT_VERSION = "HSS_BOOTSTRAP1000_STABILITY99_DIRECTION80_V1"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)




TARGET_HOSTS = ["human", "pig", "chicken"]




ALL_VALID_HOSTS = ["human", "pig", "chicken"]


TOP_N_GENES = 100

N_BOOTSTRAPS = 1000


DEFAULT_TOP_K_FOR_CURRENT_FIGURE = 50


NETWORK_STABILITY_THRESHOLD = 99

DIRECTION_CONSISTENCY_THRESHOLD = 80


SAVE_FULL_BOOTSTRAP_PAIR_RANKINGS = True

FULL_PAIR_RANKING_MAX_RANK = None


INTERACTION_SAMPLE_SIZE = None

BOOTSTRAP_INTERACTION_SAMPLE_SIZE = None


HEATMAP_COLOR_QUANTILE = 0.90

MIN_FREE_GB = 5


HOST_COLORS = {

    "human": "#E41A1C",

    "pig": "#377EB8",

    "chicken": "#FFD700"

}


def log(msg):

    print(msg, flush=True)


log("===========================================")

log(f"脚本版本: {SCRIPT_VERSION}")

log(f"当前工作目录: {BASE_DIR}")

log("当前脚本只跑: chicken")

log("数据中保留 human / pig / chicken，用于 chicken vs rest")

log("B-set Top100 二阶 SHAP interaction + Bootstrap")

log(f"N_BOOTSTRAPS = {N_BOOTSTRAPS}")

log("主模型只重新训练一次")

log("Bootstrap 不重新训练模型")

log("二阶 SHAP + bootstrap 阶段强制筛选")

log("使用 Fig5 中可用的树模型")

log("所有输入文件均从当前目录读取")

log("===========================================")








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


            return -f1_score(y_arr, preds, average="macro", zero_division=0)


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


def clean_host_name(x):

    return str(x).strip().lower()


def is_tree_like_model(model):

    return model.__class__.__name__ in [

        "LGBMClassifier",

        "XGBClassifier",

        "RandomForestClassifier",

        "XGBRFClassifier"

    ]

def is_svm_model(name, model):

    return name.upper() == "SVM" or model.__class__.__name__ == "SVC"


def is_mlp_model(name, model):

    return name.upper() == "MLP" or model.__class__.__name__ == "MLPClassifier"


def get_flat_path(filename, required=True):

    path = os.path.join(BASE_DIR, filename)

    if required and not os.path.exists(path):

        raise FileNotFoundError(f"找不到文件: {path}")

    return path


def get_host_model_path(host_name):

    return get_flat_path(f"{host_name}_Ensemble_Nelder_Mead.joblib", required=True)


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


def check_free_space(path, min_free_gb=5):

    usage = shutil.disk_usage(path)

    free_gb = usage.free / (1024 ** 3)

    log(f"当前输出磁盘剩余空间: {free_gb:.2f} GB")


    if free_gb < min_free_gb:

        raise RuntimeError(

            f"磁盘剩余空间不足: {free_gb:.2f} GB < {min_free_gb} GB。"

        )


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


    log(f"已保存: {png_path}")

    log(f"已保存: {pdf_path}")


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

        log(f"只找到 {len(selected_names)} 个树模型，少于预期 {max_models} 个。")


    return selected_names, pd.DataFrame(skipped_records)


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
        # component-level interaction values are directly comparable.
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

            f"{context_label}: 没有任何树 树模型成功计算 Top100 二阶 SHAP interaction。"

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


    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)


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


def save_one_bootstrap_backup(df, backup_dir, target_host, bootstrap_id):

    os.makedirs(backup_dir, exist_ok=True)


    out_path = os.path.join(

        backup_dir,

        f"{target_host}_bootstrap_{bootstrap_id:04d}_pair_ranking.csv"

    )


    tmp_path = out_path + ".tmp"

    df.to_csv(tmp_path, index=False, encoding="utf-8")

    os.replace(tmp_path, out_path)


    return out_path


def append_csv_gz_safely(df, out_path, bootstrap_id):

    write_header = not os.path.exists(out_path)


    tmp_gz_path = out_path + f".bootstrap_{bootstrap_id:04d}.tmp.gz"


    with gzip.open(tmp_gz_path, "wt", encoding="utf-8", newline="") as f:

        df.to_csv(f, index=False, header=write_header)


    with open(out_path, "ab") as fout:

        with open(tmp_gz_path, "rb") as fin:

            shutil.copyfileobj(fin, fout)


    os.remove(tmp_gz_path)


def validate_gzip_file(gz_path):

    if not os.path.exists(gz_path):

        raise FileNotFoundError(f"找不到 gzip 文件: {gz_path}")


    total_bytes = 0


    with gzip.open(gz_path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:

                break

            total_bytes += len(chunk)


    return total_bytes


def make_bootstrap_pair_rank_df(

    abs_matrix,

    signed_matrix,

    selected_features,

    clean_features,

    bootstrap_id

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








required_files = [

    "gene_presence_absence_cleaned.csv",

    "file_with_host_date_ip.csv",

]


for host in TARGET_HOSTS:

    required_files.extend([

        f"{host}_Ensemble_Nelder_Mead.joblib",

        f"{host}_Bset_NelderMead_TreeOnly_SHAP_Top100.joblib",

        f"{host}_Final_TreeOnly_SHAP_Weights_After_Failures.csv",

    ])


log("检查当前目录输入文件...")

for fn in required_files:

    p = get_flat_path(fn, required=True)

    log(f"OK: {fn}: {p}")








log("正在读取基因存在/缺失矩阵与元数据...")


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

X_all = X_all.loc[common_ids].copy()

meta_df = meta_df.loc[common_ids].copy()


host_col = "Host" if "Host" in meta_df.columns else meta_df.columns[1]

meta_df["Host_Clean"] = meta_df[host_col].apply(clean_host_name)










valid_host_mask = meta_df["Host_Clean"].isin(ALL_VALID_HOSTS)

X_all = X_all.loc[valid_host_mask].copy()

meta_df = meta_df.loc[valid_host_mask].copy()


log("保留 human / pig / chicken 后样本数:")

log(str(len(X_all)))

log(meta_df["Host_Clean"].value_counts().reindex(ALL_VALID_HOSTS).fillna(0).astype(int).to_string())


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


X_A_full = X_all.loc[is_a].copy()

X_B_full = X_all.loc[~is_a].copy()


log(f"Scheme {SCHEME_IDX}: {time_p} | {ip_p}")

log(f"A集: {len(X_A_full)} 样本")

log(f"B集: {len(X_B_full)} 样本")

log("当前 Fig6 interaction / bootstrap 只使用 B集。")








out_dir = os.path.join(

    BASE_DIR,

    "Fig6_chicken_bootstrap1000"

)


os.makedirs(out_dir, exist_ok=True)

check_free_space(out_dir, min_free_gb=MIN_FREE_GB)


global_summary_records = []








for target_host in TARGET_HOSTS:

    log(f"\n{'=' * 90}")

    log(f"正在解析宿主: {target_host.upper()}")

    log(f"{'=' * 90}")


    host_out_dir = os.path.join(out_dir, target_host)

    weight_dir = os.path.join(host_out_dir, "Weights")

    matrix_dir = os.path.join(host_out_dir, "Raw_Interaction_Matrices")

    table_dir = os.path.join(host_out_dir, "Tables")

    figure_dir = os.path.join(host_out_dir, "Figures")

    plotdata_dir = os.path.join(host_out_dir, "Plot_Data")

    model_dir = os.path.join(host_out_dir, "Refit_Top100_Models")

    ranking_dir = os.path.join(host_out_dir, "Bootstrap_AllPair_Rankings")

    ranking_backup_dir = os.path.join(host_out_dir, "Bootstrap_Rankings_ByBootstrap")


    for d in [

        host_out_dir,

        weight_dir,

        matrix_dir,

        table_dir,

        figure_dir,

        plotdata_dir,

        model_dir,

        ranking_dir,

        ranking_backup_dir,

    ]:

        os.makedirs(d, exist_ok=True)


    y_B = (

        meta_df.loc[X_B_full.index, "Host_Clean"]

        .fillna("")

        .astype(str)

        .str.lower()

        .eq(target_host.lower())

        .astype(int)

    )


    log("B集二分类标签分布:")

    label_counts = y_B.value_counts().rename(index={0: "rest", 1: target_host})

    log(label_counts.to_string())


    if y_B.nunique() < 2:

        log(f"{target_host} 的 B集标签只有一个类别，跳过。")


        global_summary_records.append({

            "Host": target_host,

            "Status": "Skipped: only one class in B set",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


        continue








    shap_top_joblib_path = get_host_top100_joblib_path(target_host)

    selected_features = list(joblib.load(shap_top_joblib_path))


    selected_features = [

        f for f in selected_features

        if f in X_B_full.columns

    ]


    if len(selected_features) == 0:

        log(f"{target_host} selected_features 为空，跳过。")


        global_summary_records.append({

            "Host": target_host,

            "Status": "Skipped: no selected features found in X_B",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


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

        os.path.join(table_dir, f"Fig6_{target_host}_Used_Top100_Features.csv"),

        index=False

    )


    log(f"已加载 {target_host} 的 B集 Top{len(selected_features)} genes。")

    log(f"X_B_top100 shape = {X_B_top100.shape}")








    try:

        fig5_allowed_tree_names, fig5_final_weight_df = load_fig5_final_tree_model_names_flat(

            target_host=target_host

        )


        fig5_final_weight_df.to_csv(

            os.path.join(

                weight_dir,

                f"{target_host}_Fig5_Final_TreeOnly_Models_Original.csv"

            ),

            index=False

        )


        log("Fig5 实际参与 Tree-only SHAP 的模型:")

        for m in fig5_allowed_tree_names:

            log(f"   - {m}")


    except Exception as e:

        log(f"读取 Fig5 最终 Tree-only 模型列表失败: {e}")


        global_summary_records.append({

            "Host": target_host,

            "Status": f"Failed: load Fig5 tree model names: {e}",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


        continue








    try:

        model_path = get_host_model_path(target_host)

        log(f"找到原始模型文件: {model_path}")


        bundle = joblib.load(model_path)

        saved_nelder_model = (

            bundle["model"]

            if isinstance(bundle, dict) and "model" in bundle

            else bundle

        )


    except Exception as e:

        log(f"读取原始 NelderMead 模型失败: {e}")


        global_summary_records.append({

            "Host": target_host,

            "Status": f"Failed: load saved NelderMead model: {e}",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


        continue








    try:

        selected_tree_names_for_fig6, model_selection_skipped_df = (

            select_tree_models_from_fig5(

                nelder_model=saved_nelder_model,

                fig5_allowed_model_names=fig5_allowed_tree_names,

                max_models=3

            )

        )


        model_selection_df = pd.DataFrame({

            "Selected_Model_For_Fig6": selected_tree_names_for_fig6,

            "Rule": "Tree models from Fig5 final Tree-only SHAP",

        })


        model_selection_df.to_csv(

            os.path.join(

                weight_dir,

                f"{target_host}_Fig6_Selected_TreeModels.csv"

            ),

            index=False

        )


        if len(model_selection_skipped_df) > 0:

            model_selection_skipped_df.to_csv(

                os.path.join(

                    weight_dir,

                    f"{target_host}_Fig6_ModelSelection_Skipped_Models.csv"

                ),

                index=False

            )


        log("Fig6 二阶 SHAP + bootstrap 将使用树模型:")

        for m in selected_tree_names_for_fig6:

            log(f"   - {m}")


    except Exception as e:

        log(f"Fig6 选择树模型失败: {e}")


        global_summary_records.append({

            "Host": target_host,

            "Status": f"Failed: select tree models: {e}",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


        continue








    try:

        tree_templates, template_skipped_df = extract_tree_template_components_from_saved_nelder(

            nelder_model=saved_nelder_model,

            allowed_model_names=selected_tree_names_for_fig6

        )


        template_skipped_df.to_csv(

            os.path.join(weight_dir, f"{target_host}_Template_Skipped_Models.csv"),

            index=False

        )


        log("从原始 NelderMead 中提取到的 Fig6 树 树模型模板:")

        for name, model in tree_templates:

            log(f"   - {name}: {model.__class__.__name__}")


    except Exception as e:

        log(f"提取 Fig6 树模型模板失败: {e}")


        global_summary_records.append({

            "Host": target_host,

            "Status": f"Failed: extract tree templates: {e}",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


        continue








    log("正在使用 B集 Top100 重新训练 Fig6 树 Tree-only NelderMead 主模型...")

    log("注意：这个主模型只训练一次，Bootstrap 阶段不会再重新训练。")


    try:

        top100_main_nelder = refit_top100_tree_nelder_from_templates(

            tree_templates=tree_templates,

            X_fit_df=X_B_top100,

            y_fit=y_B,

            random_state=42

        )


        (

            main_names,

            main_models,

            main_weights,

            main_used_df,

            main_skipped_df

        ) = get_tree_components_from_refit_nelder(

            top100_main_nelder,

            skip_zero_weight=True

        )


        main_used_df.to_csv(

            os.path.join(

                weight_dir,

                f"{target_host}_Fig6_Top100_Main_Tree_NelderMead_TreeOnly_Weights.csv"

            ),

            index=False

        )


        if len(main_skipped_df) > 0:

            main_skipped_df.to_csv(

                os.path.join(

                    weight_dir,

                    f"{target_host}_Fig6_Top100_Main_Tree_NelderMead_TreeOnly_Skipped.csv"

                ),

                index=False

            )


        log("Fig6 Top100 树 主模型 NelderMead 权重:")

        log(main_used_df.to_string(index=False))


    except Exception as e:

        log(f"{target_host} Top100 主模型重训练或权重提取失败: {e}")


        global_summary_records.append({

            "Host": target_host,

            "Status": f"Failed: refit Top100 main model: {e}",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


        continue


    log("正在创建树 树模型 TreeExplainer...")

    main_explainers = make_tree_explainers(main_names, main_models)


    joblib.dump(

        {

            "target_host": target_host,

            "selected_features": selected_features,

            "clean_features": clean_features,

            "selected_tree_names_for_fig6": selected_tree_names_for_fig6,

            "tree_models_selected_for_fig6": True,

            "top100_main_nelder": top100_main_nelder,

            "fig6_main_weights": main_used_df,

            "bootstrap_mode": "sample_only_no_model_refit",

            "interaction_models": main_names,

            "n_bootstraps": N_BOOTSTRAPS,

            "full_ranking_format": "csv.gz",

            "per_bootstrap_backup": True,

            "script_version": SCRIPT_VERSION,

        },

        os.path.join(

            model_dir,

            f"{target_host}_Fig6_Bset_Top100_MainModel_TreeModels_NoBootstrapRefit.joblib"

        ),

        compress=3

    )








    if INTERACTION_SAMPLE_SIZE is not None and len(X_B_top100) > INTERACTION_SAMPLE_SIZE:

        X_interaction_main = X_B_top100.sample(

            n=INTERACTION_SAMPLE_SIZE,

            random_state=42

        )

        log(f"主 interaction 仅使用 {len(X_interaction_main)} 个 B集样本。")

    else:

        X_interaction_main = X_B_top100


    log("正在计算主模型 Top100 加权二阶 SHAP interaction...")


    try:

        (

            mean_signed_interactions_main,

            mean_abs_interactions_main,

            main_effects,

            interaction_weight_df,

            interaction_skipped_df

        ) = compute_weighted_interactions_from_fixed_components(

            names=main_names,

            models=main_models,

            weights=main_weights,

            explainers=main_explainers,

            X_top100_df=X_interaction_main,

            context_label=f"{target_host}_main"

        )


    except Exception as e:

        log(f"{target_host} 主 interaction 计算失败: {e}")


        global_summary_records.append({

            "Host": target_host,

            "Status": f"Failed: main SHAP interaction: {e}",

            "N_BOOTSTRAPS": N_BOOTSTRAPS

        })


        continue


    interaction_weight_df.to_csv(

        os.path.join(

            weight_dir,

            f"{target_host}_Main_Top100_Tree_Interaction_Weights.csv"

        ),

        index=False

    )


    if len(interaction_skipped_df) > 0:

        interaction_skipped_df.to_csv(

            os.path.join(

                weight_dir,

                f"{target_host}_Main_Top100_Tree_Interaction_Skipped.csv"

            ),

            index=False

        )


    log("主 interaction 使用的最终树 模型权重:")

    log(interaction_weight_df.to_string(index=False))


    save_matrix_csv(

        mean_signed_interactions_main,

        clean_features,

        clean_features,

        os.path.join(

            matrix_dir,

            f"{target_host}_Main_Top100_Tree_Signed_Interaction_Matrix_CleanNames.csv"

        )

    )


    save_matrix_csv(

        mean_abs_interactions_main,

        clean_features,

        clean_features,

        os.path.join(

            matrix_dir,

            f"{target_host}_Main_Top100_Tree_Abs_Interaction_Matrix_CleanNames.csv"

        )

    )


    save_matrix_csv(

        mean_signed_interactions_main,

        selected_features,

        selected_features,

        os.path.join(

            matrix_dir,

            f"{target_host}_Main_Top100_Tree_Signed_Interaction_Matrix_RawFeatureNames.csv"

        )

    )


    save_matrix_csv(

        mean_abs_interactions_main,

        selected_features,

        selected_features,

        os.path.join(

            matrix_dir,

            f"{target_host}_Main_Top100_Tree_Abs_Interaction_Matrix_RawFeatureNames.csv"

        )

    )


    main_effect_df = pd.DataFrame({

        "Feature": selected_features,

        "Clean_Name": clean_features,

        "Main_Effect_Abs_Diagonal": main_effects

    }).sort_values(

        by="Main_Effect_Abs_Diagonal",

        ascending=False

    ).reset_index(drop=True)


    main_effect_df.to_csv(

        os.path.join(table_dir, f"{target_host}_Main_Top100_Tree_MainEffects.csv"),

        index=False

    )








    log("绘制 Fig6A: Top15 main-effect genes signed interaction heatmap...")


    top15_idx = (

        np.argsort(main_effects)[::-1][:15]

        if len(main_effects) >= 15

        else np.argsort(main_effects)[::-1]

    )


    interaction_subset_signed = mean_signed_interactions_main[np.ix_(top15_idx, top15_idx)]

    top15_names = [clean_features[i] for i in top15_idx]


    fig6a_plot_df = pd.DataFrame(

        interaction_subset_signed,

        index=top15_names,

        columns=top15_names

    )


    fig6a_plot_df.to_csv(

        os.path.join(

            plotdata_dir,

            f"Fig6A_{target_host}_Top15_MainEffect_Tree_SignedInteraction_HeatmapData.csv"

        )

    )


    draw_signed_interaction_heatmap(

        matrix=interaction_subset_signed,

        labels=top15_names,

        title=(

            f"Fig.6A: B-set Top100 Signed SHAP Interaction\n"

            f"{target_host.capitalize()} | tree models"

        ),

        out_prefix=os.path.join(

            figure_dir,

            f"Fig6A_{target_host}_Bset_Top100_Tree_Signed_Heatmap"

        ),

        figsize=(10, 8),

        color_quantile=HEATMAP_COLOR_QUANTILE

    )








    log(

        f"正在执行 {N_BOOTSTRAPS} 次 Bootstrap："

        f"只重抽样 B集样本，不重新训练模型，使用树模型"

    )


    pair_frequency = {}

    pair_signs = {}


    valid_bootstrap_count = 0

    bootstrap_summary_records = []


    n_samples_B = len(X_B_top100)


    full_ranking_path = os.path.join(

        ranking_dir,

        f"{target_host}_Bootstrap_AllPair_Rankings_1000.csv.gz"

    )


    if os.path.exists(full_ranking_path):

        os.remove(full_ranking_path)


    for b in range(N_BOOTSTRAPS):

        bootstrap_id = b + 1


        log(f"   {target_host} Bootstrap {bootstrap_id}/{N_BOOTSTRAPS} 开始...")


        check_free_space(out_dir, min_free_gb=MIN_FREE_GB)


        boot_indices = resample(

            np.arange(n_samples_B),

            n_samples=n_samples_B,

            replace=True,

            random_state=b

        )


        X_boot_top100 = X_B_top100.iloc[boot_indices].copy()


        if (

            BOOTSTRAP_INTERACTION_SAMPLE_SIZE is not None

            and len(X_boot_top100) > BOOTSTRAP_INTERACTION_SAMPLE_SIZE

        ):

            X_boot_for_interaction = X_boot_top100.sample(

                n=BOOTSTRAP_INTERACTION_SAMPLE_SIZE,

                random_state=b

            )

        else:

            X_boot_for_interaction = X_boot_top100


        try:

            (

                signed_boot_mean,

                boot_mean_abs,

                boot_main_effects,

                boot_interaction_weight_df,

                boot_interaction_skipped_df

            ) = compute_weighted_interactions_from_fixed_components(

                names=main_names,

                models=main_models,

                weights=main_weights,

                explainers=main_explainers,

                X_top100_df=X_boot_for_interaction,

                context_label=f"{target_host}_bootstrap_{bootstrap_id}"

            )


            pair_rank_df = make_bootstrap_pair_rank_df(

                abs_matrix=boot_mean_abs,

                signed_matrix=signed_boot_mean,

                selected_features=selected_features,

                clean_features=clean_features,

                bootstrap_id=bootstrap_id

            )


            if SAVE_FULL_BOOTSTRAP_PAIR_RANKINGS:

                if FULL_PAIR_RANKING_MAX_RANK is not None:

                    save_rank_df = pair_rank_df[

                        pair_rank_df["Rank"] <= FULL_PAIR_RANKING_MAX_RANK

                    ].copy()

                else:

                    save_rank_df = pair_rank_df


                backup_file = save_one_bootstrap_backup(

                    df=save_rank_df,

                    backup_dir=ranking_backup_dir,

                    target_host=target_host,

                    bootstrap_id=bootstrap_id

                )


                append_csv_gz_safely(

                    df=save_rank_df,

                    out_path=full_ranking_path,

                    bootstrap_id=bootstrap_id

                )

            else:

                backup_file = ""


            valid_bootstrap_count += 1


            update_default_topk_counters_from_rank_df(

                pair_rank_df=pair_rank_df,

                top_k=DEFAULT_TOP_K_FOR_CURRENT_FIGURE,

                pair_frequency=pair_frequency,

                pair_signs=pair_signs

            )


            bootstrap_summary_records.append({

                "Bootstrap": bootstrap_id,

                "Status": "OK",

                "Host": target_host,

                "Mode": "Sample-only bootstrap; no model refit; tree models",

                "N_original_B_samples": n_samples_B,

                "N_resampled_rows": len(X_boot_top100),

                "N_samples_for_interaction": len(X_boot_for_interaction),

                "N_models_for_interaction": len(boot_interaction_weight_df),

                "Models": ";".join(boot_interaction_weight_df["Model"].astype(str).tolist()),

                "N_All_Pairs_Ranked": len(pair_rank_df),

                "Default_TopK_For_Current_Figure": DEFAULT_TOP_K_FOR_CURRENT_FIGURE,

                "Saved_Full_Ranking": SAVE_FULL_BOOTSTRAP_PAIR_RANKINGS,

                "Full_Ranking_File": full_ranking_path,

                "Full_Ranking_Format": "csv.gz",

                "Backup_File": backup_file

            })


            log(

                f"   {target_host} Bootstrap {bootstrap_id}/{N_BOOTSTRAPS} 完成。"

                f" 本次共排序 {len(pair_rank_df)} 个 gene pairs。"

            )


        except Exception as e:

            log(f"   {target_host} Bootstrap {bootstrap_id}/{N_BOOTSTRAPS} 失败，跳过。原因: {e}")


            bootstrap_summary_records.append({

                "Bootstrap": bootstrap_id,

                "Status": f"Failed: {e}",

                "Host": target_host,

                "Mode": "Sample-only bootstrap; no model refit; tree models",

            })


            continue


    bootstrap_summary_df = pd.DataFrame(bootstrap_summary_records)


    bootstrap_summary_df.to_csv(

        os.path.join(

            table_dir,

            f"{target_host}_Bootstrap1000_Summary_SampleOnly_NoRefit.csv"

        ),

        index=False

    )


    log(f"{target_host} 有效 Bootstrap 次数: {valid_bootstrap_count}/{N_BOOTSTRAPS}")


    if SAVE_FULL_BOOTSTRAP_PAIR_RANKINGS:

        log(f"完整 bootstrap pair ranking 总表已保存: {full_ranking_path}")

        log(f"每次 bootstrap 单独 ranking 备份已保存: {ranking_backup_dir}")


        try:

            log("正在验证总 csv.gz 是否完整...")

            total_bytes = validate_gzip_file(full_ranking_path)

            log(f"gzip 验证通过，解压后字节数约: {total_bytes}")

        except Exception as e:

            log(f"gzip 验证失败: {e}")

            log("不用慌：每次 bootstrap 的单独 CSV 备份已经保存，可以用备份重建总表。")








    log(

        f"正在汇总默认 TopK={DEFAULT_TOP_K_FOR_CURRENT_FIGURE} "

        f"下的稳定性与方向一致性..."

    )


    df_stability = build_stability_df_from_counters(

        pair_frequency=pair_frequency,

        pair_signs=pair_signs,

        valid_bootstrap_count=valid_bootstrap_count,

        selected_features=selected_features,

        clean_features=clean_features,

        mean_signed_interactions_main=mean_signed_interactions_main,

        mean_abs_interactions_main=mean_abs_interactions_main,

        main_names=main_names,

        top_k=DEFAULT_TOP_K_FOR_CURRENT_FIGURE

    )


    stability_csv = os.path.join(

        table_dir,

        f"Fig6_{target_host}_TopK{DEFAULT_TOP_K_FOR_CURRENT_FIGURE}_Bootstrap1000_All_Pair_Stability_Direction_SampleOnly_NoRefit.csv"

    )


    df_stability.to_csv(stability_csv, index=False)


    old_style_stability_csv = os.path.join(

        table_dir,

        f"Fig6_{target_host}_Top100_All_Pair_Stability_Direction_SampleOnly_NoRefit.csv"

    )


    df_stability.to_csv(old_style_stability_csv, index=False)


    log(f"默认 TopK stability 表已保存: {stability_csv}")

    log(f"兼容旧命名 stability 表已保存: {old_style_stability_csv}")


    if len(df_stability) == 0:

        log(f"{target_host} 未产生可统计互作对，跳过后续图。")


        global_summary_records.append({

            "Host": target_host,

            "Status": "No pair generated",

            "Bootstrap_Mode": "Sample-only; no model refit",

            "N_BOOTSTRAPS": N_BOOTSTRAPS,

            "N_Top100": len(selected_features),

            "Default_TopK_For_Current_Figure": DEFAULT_TOP_K_FOR_CURRENT_FIGURE,

            "N_Fig6_TreeModels": len(selected_tree_names_for_fig6),

            "Fig6_TreeModels": ";".join(selected_tree_names_for_fig6),

            "N_Valid_Bootstrap": valid_bootstrap_count,

            "Full_Ranking_File": full_ranking_path,

            "Full_Ranking_Format": "csv.gz"

        })


        continue








    log("绘制 Fig6B: Top15 稳定互作条形图...")


    df_stability_plot = df_stability.head(15).copy()


    df_stability_plot.to_csv(

        os.path.join(

            plotdata_dir,

            f"Fig6B_{target_host}_Top15_Bootstrap1000_Stability_BarplotData.csv"

        ),

        index=False

    )


    sns.set_theme(style="ticks", font_scale=1.1)


    fig, ax = plt.subplots(figsize=(11, 7))


    sns.barplot(

        data=df_stability_plot,

        x="Stability(%)",

        y="Display_Name",

        palette="mako",

        ax=ax

    )


    sns.despine(ax=ax)


    ax.axvline(

        x=NETWORK_STABILITY_THRESHOLD,

        color="#e74c3c",

        linestyle="--",

        linewidth=2,

        label=f"{NETWORK_STABILITY_THRESHOLD}% threshold"

    )


    for container in ax.containers:

        ax.bar_label(container, fmt="%.1f%%", padding=5, fontsize=10)


    ax.set_title(

        f"Fig.6B: B-set Top100 Bootstrap1000 Interaction Stability\n"

        f"{target_host.capitalize()} | tree models",

        fontweight="bold",

        fontsize=15

    )


    ax.set_xlabel("Selection Frequency (%)", fontweight="bold", fontsize=13)

    ax.set_ylabel("")

    ax.legend(loc="lower right", frameon=False)


    plt.tight_layout()


    savefig_png_pdf(

        fig,

        os.path.join(

            figure_dir,

            f"Fig6B_{target_host}_Top100_Bootstrap1000_Stability_"

        ),

        dpi=600

    )


    plt.close(fig)








    log("绘制 Fig6C: 方向一致性互作网络...")


    robust_pairs = df_stability[

        (df_stability["Stability(%)"] >= NETWORK_STABILITY_THRESHOLD) &

        (df_stability["Direction_Consistency(%)"] >= DIRECTION_CONSISTENCY_THRESHOLD)

    ].copy()


    robust_csv = os.path.join(

        table_dir,

        f"Fig6C_{target_host}_Robust_Pairs_For_Network_Bootstrap1000_SampleOnly_NoRefit.csv"

    )


    robust_pairs.to_csv(robust_csv, index=False)


    if len(robust_pairs) == 0:

        log(

            f"{target_host} 无满足 "

            f"Stability>={NETWORK_STABILITY_THRESHOLD}% 且 "

            f"DirectionConsistency>={DIRECTION_CONSISTENCY_THRESHOLD}% 的互作，跳过 Fig6C/D。"

        )


        global_summary_records.append({

            "Host": target_host,

            "Status": "No robust pair",

            "Bootstrap_Mode": "Sample-only; no model refit",

            "N_BOOTSTRAPS": N_BOOTSTRAPS,

            "N_Top100": len(selected_features),

            "N_Fig6_TreeModels": len(selected_tree_names_for_fig6),

            "Fig6_TreeModels": ";".join(selected_tree_names_for_fig6),

            "N_Valid_Bootstrap": valid_bootstrap_count,

            "N_All_Pairs_Record": len(df_stability),

            "N_Robust_Pairs": 0,

            "Full_Ranking_File": full_ranking_path,

            "Full_Ranking_Format": "csv.gz"

        })


        continue


    G = nx.Graph()


    for _, row in robust_pairs.iterrows():

        edge_color = "#d73027" if row["Major_Sign"] > 0 else "#4575b4"


        G.add_edge(

            row["Gene_A"],

            row["Gene_B"],

            weight=row["Stability(%)"] / 100.0,

            edge_color=edge_color,

            major_sign=row["Major_Sign"],

            direction_consistency=row["Direction_Consistency(%)"],

            stability=row["Stability(%)"],

            main_abs_interaction=row["Main_Abs_Interaction"]

        )


    fig, ax = plt.subplots(figsize=(11, 9))


    try:

        pos = nx.kamada_kawai_layout(G)

    except Exception:

        pos = nx.spring_layout(G, seed=42)


    degrees = dict(G.degree())


    node_sizes = [v * 650 + 450 for v in degrees.values()]

    edge_colors = [G[u][v]["edge_color"] for u, v in G.edges()]

    edge_widths = [max(1.5, G[u][v]["weight"] * 4.5) for u, v in G.edges()]


    nx.draw_networkx_nodes(

        G,

        pos,

        node_size=node_sizes,

        node_color=list(degrees.values()),

        cmap=plt.cm.YlGnBu,

        edgecolors="#2c3e50",

        linewidths=1.0,

        ax=ax

    )


    nx.draw_networkx_edges(

        G,

        pos,

        width=edge_widths,

        edge_color=edge_colors,

        connectionstyle="arc3,rad=0.12",

        alpha=0.85,

        ax=ax

    )


    nx.draw_networkx_labels(

        G,

        pos,

        font_size=9,

        font_weight="bold",

        bbox=dict(

            boxstyle="round,pad=0.25",

            fc="white",

            ec="none",

            alpha=0.88

        ),

        ax=ax

    )


    legend_elements = [

        Line2D([0], [0], color="#d73027", lw=3, label="Positive interaction"),

        Line2D([0], [0], color="#4575b4", lw=3, label="Negative interaction")

    ]


    ax.legend(

        handles=legend_elements,

        loc="upper right",

        frameon=False,

        fontsize=11

    )


    ax.set_title(

        f"Fig.6C: Direction-consistent Interaction Network\n"

        f"{target_host.capitalize()} | tree models",

        fontweight="bold",

        fontsize=15

    )


    ax.axis("off")

    plt.tight_layout()


    savefig_png_pdf(

        fig,

        os.path.join(

            figure_dir,

            f"Fig6C_{target_host}_Direction_Consistent_Network_Bootstrap1000_"

        ),

        dpi=600

    )


    plt.close(fig)








    log("绘制 Fig6D: Top robust pair 四组合状态机制图...")


    top_raw_pair = robust_pairs.iloc[0]["Pair"]

    geneA_raw, geneB_raw = top_raw_pair


    geneA_clean = clean_gene_name(geneA_raw)

    geneB_clean = clean_gene_name(geneB_raw)


    probs = predict_proba_top100_tree_nelder(

        top100_nelder_model=top100_main_nelder,

        X_top100_df=X_B_top100

    )


    gA_vals = X_B_top100[geneA_raw].values.astype(int)

    gB_vals = X_B_top100[geneB_raw].values.astype(int)


    combinations = [f"{vA}{vB}" for vA, vB in zip(gA_vals, gB_vals)]


    df_comb = pd.DataFrame({

        "SampleID": X_B_top100.index,

        "Prediction_Probability": probs,

        "State": combinations,

        "GeneA_Raw": geneA_raw,

        "GeneB_Raw": geneB_raw,

        "GeneA_Clean": geneA_clean,

        "GeneB_Clean": geneB_clean,

        "Actual_Host": meta_df.loc[X_B_top100.index, "Host_Clean"].astype(str).values,

        "Fig6_Models": ";".join(main_names),

        "N_BOOTSTRAPS": N_BOOTSTRAPS

    })


    state_mapping = {

        "00": "Neither",

        "10": f"Only {geneA_clean}",

        "01": f"Only {geneB_clean}",

        "11": "Both Present"

    }


    df_comb["State_Label"] = df_comb["State"].map(state_mapping)


    order_labels = [

        state_mapping["00"],

        state_mapping["10"],

        state_mapping["01"],

        state_mapping["11"]

    ]


    fig6d_data_csv = os.path.join(

        plotdata_dir,

        f"Fig6D_{target_host}_TopRobustPair_Bootstrap1000_ViolinBox_Data.csv"

    )


    df_comb.to_csv(fig6d_data_csv, index=False)


    df_comb_summary = df_comb.groupby("State_Label")["Prediction_Probability"].agg(

        N="count",

        Mean="mean",

        Median="median",

        Std="std",

        Min="min",

        Max="max"

    ).reset_index()


    df_comb_summary.to_csv(

        os.path.join(

            table_dir,

            f"Fig6D_{target_host}_TopRobustPair_Bootstrap1000_ViolinBox_Summary.csv"

        ),

        index=False

    )


    fig, ax = plt.subplots(figsize=(10, 6.5))


    sns.violinplot(

        data=df_comb,

        x="State_Label",

        y="Prediction_Probability",

        order=order_labels,

        palette=["#e0e0e0", "#a8ddb5", "#7bccc4", "#0868ac"],

        inner=None,

        linewidth=0,

        alpha=0.55,

        ax=ax

    )


    sns.boxplot(

        data=df_comb,

        x="State_Label",

        y="Prediction_Probability",

        order=order_labels,

        color="white",

        width=0.22,

        boxprops={"zorder": 2, "alpha": 0.85},

        showfliers=False,

        ax=ax

    )


    sns.stripplot(

        data=df_comb,

        x="State_Label",

        y="Prediction_Probability",

        order=order_labels,

        color="black",

        alpha=0.15,

        size=2,

        jitter=True,

        zorder=3,

        ax=ax

    )


    ax.set_title(

        f"Fig.6D: Epistatic Mechanism of Top Robust Pair\n"

        f"{target_host.capitalize()} | {geneA_clean} × {geneB_clean}",

        fontweight="bold",

        fontsize=15

    )


    ax.set_xlabel("Gene Combination", fontweight="bold", fontsize=13)


    ax.set_ylabel(

        f"Predicted Probability ({target_host.capitalize()})",

        fontweight="bold",

        fontsize=13

    )


    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.tick_params(axis="x", labelrotation=15)


    plt.tight_layout()


    savefig_png_pdf(

        fig,

        os.path.join(

            figure_dir,

            f"Fig6D_{target_host}_TopRobustPair_Bootstrap1000_ViolinBox_"

        ),

        dpi=600

    )


    plt.close(fig)


    global_summary_records.append({

        "Host": target_host,

        "Status": "OK",

        "Bootstrap_Mode": "Sample-only; no model refit",

        "N_BOOTSTRAPS": N_BOOTSTRAPS,

        "N_Top100": len(selected_features),

        "N_Fig6_TreeModels": len(selected_tree_names_for_fig6),

        "Fig6_TreeModels": ";".join(selected_tree_names_for_fig6),

        "N_Valid_Bootstrap": valid_bootstrap_count,

        "N_All_Pairs_Record": len(df_stability),

        "N_Robust_Pairs": len(robust_pairs),

        "Top_Robust_Pair": f"{geneA_clean} & {geneB_clean}",

        "Top_Robust_Pair_Stability(%)": robust_pairs.iloc[0]["Stability(%)"],

        "Top_Robust_Pair_Direction_Consistency(%)": robust_pairs.iloc[0]["Direction_Consistency(%)"],

        "Top_Robust_Pair_Major_Sign": robust_pairs.iloc[0]["Major_Sign"],

        "Full_Ranking_File": full_ranking_path,

        "Full_Ranking_Format": "csv.gz",

        "Per_Bootstrap_Backup_Dir": ranking_backup_dir

    })


    log(f"{target_host} 完成。")








summary_df = pd.DataFrame(global_summary_records)


summary_dir = os.path.join(out_dir, "Summary")

os.makedirs(summary_dir, exist_ok=True)


run_tag = "_".join(TARGET_HOSTS)


summary_path = os.path.join(

    summary_dir,

    f"Fig6_{run_tag}_Bootstrap1000_Run_Summary_SampleOnly_NoRefit_gz_backup.csv"

)


summary_df.to_csv(summary_path, index=False)


log("\n全部完成。")

log(f"Chicken Fig6 bootstrap1000 结果已保存到: {out_dir}")

log(f"总汇总表: {summary_path}")


log("\n本代码保存内容包括:")

log("1. chicken 的 Fig5 实际参与 Tree-only SHAP 的原始模型列表")

log("2. chicken 的 Fig6 实际使用的树模型列表")

log("3. chicken 的 Fig6 Top100 主模型重训后的树 模型权重")

log("4. chicken 主模型 Top100 signed / abs 二阶 SHAP interaction 矩阵")

log("5. chicken 1000 次 sample-only bootstrap summary")

log("6. chicken 每次 bootstrap 的完整 gene-pair ranking 总表 csv.gz")

log("7. chicken 每次 bootstrap 单独 ranking 备份 CSV")

log("8. chicken 默认 TopK 下全部 pair 的稳定性和方向一致性表")

log("9. chicken robust pairs 网络输入表")

log("10. chicken Fig6A / Fig6B / Fig6C / Fig6D 的 png + pdf")

log("11. chicken Fig6D 的作图数据和 summary")

log("12. chicken 重训后的主模型 joblib")
