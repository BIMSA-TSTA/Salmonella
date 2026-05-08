import pandas as pd
import numpy as np
import os
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')


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

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_preds = np.zeros((len(y_arr), len(self.classes_), len(self.estimators)))

        for i, (name, model) in enumerate(self.estimators):
            oof_preds[:, :, i] = cross_val_predict(
                model,
                X,
                y_arr,
                cv=cv,
                method='predict_proba',
                n_jobs=1
            )

        def objective(w):
            w = np.clip(w, 0, 1)
            if np.sum(w) == 0:
                w = np.ones(len(w))
            w = w / np.sum(w)

            blended = np.average(oof_preds, axis=2, weights=w)
            preds = self.classes_[np.argmax(blended, axis=1)]
            return -f1_score(y_arr, preds, average='macro')

        initial_weights = np.ones(len(self.estimators)) / len(self.estimators)

        res = minimize(
            objective,
            initial_weights,
            method='Nelder-Mead',
            options={'maxiter': 200}
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
    return re.sub(r'\.fna.*$', '', str(sample_id)).strip()


def sanitize_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")


def clean_gene_name(raw_name):
    if '~~~' in str(raw_name):
        first_part = str(raw_name).split('~~~')[0]
        return f"{first_part} cluster"
    return str(raw_name)


def safe_extract_binary_shap_values(shap_output):

    if isinstance(shap_output, list):
        if len(shap_output) >= 2:
            return np.asarray(shap_output[1], dtype=float)
        return np.asarray(shap_output[-1], dtype=float)

    if hasattr(shap_output, "values"):
        vals = np.asarray(shap_output.values, dtype=float)
    else:
        vals = np.asarray(shap_output, dtype=float)

    
    if vals.ndim == 3:
        if vals.shape[2] >= 2:
            return vals[:, :, 1]
        return vals[:, :, 0]

    return vals


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


def resolve_model_path(base_dir, host_name, candidate_model_names):
    host_dir = os.path.join(base_dir, sanitize_name(host_name))

    for mname in candidate_model_names:
        path = os.path.join(host_dir, f"{mname}.joblib")
        if os.path.exists(path):
            return path, mname

    return None, None


def get_nelder_mead_tree_components(nelder_model, skip_zero_weight=True):
    
    if not hasattr(nelder_model, "estimators"):
        raise ValueError("当前模型没有 estimators，可能不是 NelderMeadEnsemble。")

    if not hasattr(nelder_model, "weights_") or nelder_model.weights_ is None:
        raise ValueError("当前 NelderMeadEnsemble 没有 weights_，可能没有正确 fit。")

    names = [name for name, _ in nelder_model.estimators]
    fitted_models = [model for _, model in nelder_model.estimators]
    weights = np.asarray(nelder_model.weights_, dtype=float)

    if len(names) != len(fitted_models) or len(weights) != len(fitted_models):
        raise ValueError("子模型名字、模型对象、Nelder-Mead 权重三者长度不一致。")

    kept_names = []
    kept_models = []
    kept_weights = []
    skipped_records = []

    for name, model, weight in zip(names, fitted_models, weights):
        model_class = model.__class__.__name__

        if is_svm_model(name, model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Original_NelderMead_Weight": weight,
                "Reason": "Skip SVM"
            })
            continue

        if is_mlp_model(name, model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Original_NelderMead_Weight": weight,
                "Reason": "Skip MLP because Kernel SHAP was numerically unstable"
            })
            continue

        if not is_tree_like_model(model):
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Original_NelderMead_Weight": weight,
                "Reason": "Skip non-tree model"
            })
            continue

        if skip_zero_weight and weight <= 0:
            skipped_records.append({
                "Model": name,
                "Class": model_class,
                "Original_NelderMead_Weight": weight,
                "Reason": "Skip zero weight"
            })
            continue

        kept_names.append(name)
        kept_models.append(model)
        kept_weights.append(weight)

    kept_weights = np.asarray(kept_weights, dtype=float)

    if len(kept_weights) == 0:
        raise ValueError("跳过 SVM、MLP、非树模型和零权重模型后，没有剩余树模型可用于 SHAP。")

    if np.sum(kept_weights) <= 0:
        raise ValueError("剩余树模型权重之和 <= 0，无法归一化。")

    normalized_weights = kept_weights / np.sum(kept_weights)

    used_weight_df = pd.DataFrame({
        "Model": kept_names,
        "Class": [m.__class__.__name__ for m in kept_models],
        "Original_NelderMead_Weight": kept_weights,
        "Normalized_TreeOnly_Weight": normalized_weights
    })

    skipped_weight_df = pd.DataFrame(skipped_records)

    return kept_names, kept_models, kept_weights, normalized_weights, used_weight_df, skipped_weight_df


def compute_tree_model_shap(model_name, model, X_explain_df):
    
    print(f"   -> 正在计算树模型 SHAP: {model_name} [{model.__class__.__name__}]")

    try:
        explainer = shap.TreeExplainer(model, model_output="probability")
        shap_raw = explainer.shap_values(X_explain_df)
    except Exception:
        explainer = shap.TreeExplainer(model)
        shap_raw = explainer.shap_values(X_explain_df)

    shap_values = safe_extract_binary_shap_values(shap_raw)
    shap_values = np.asarray(shap_values, dtype=float)

    if shap_values.shape != X_explain_df.shape:
        raise ValueError(
            f"{model_name} 的 SHAP shape 异常: {shap_values.shape}，"
            f"期望: {X_explain_df.shape}"
        )

    if not np.isfinite(shap_values).all():
        raise ValueError(f"{model_name} 的 SHAP 中存在 NaN 或 Inf。")

    print(f"      SHAP min = {np.min(shap_values):.6g}")
    print(f"      SHAP max = {np.max(shap_values):.6g}")
    print(f"      mean(|SHAP|) = {np.mean(np.abs(shap_values)):.6g}")
    print(f"      p99(|SHAP|) = {np.percentile(np.abs(shap_values), 99):.6g}")

    return shap_values


def predict_proba_nelder_tree_only(nelder_model, X_full_df):
    
    (
        submodel_names,
        submodels,
        original_weights,
        normalized_weights,
        used_weight_df,
        skipped_weight_df
    ) = get_nelder_mead_tree_components(
        nelder_model=nelder_model,
        skip_zero_weight=True
    )

    prob_list = []
    for sub_name, sub_model in zip(submodel_names, submodels):
        prob = sub_model.predict_proba(X_full_df)[:, 1]
        prob_list.append(prob)

    prob_arr = np.vstack(prob_list)
    final_prob = np.average(prob_arr, axis=0, weights=normalized_weights)

    return final_prob, used_weight_df, skipped_weight_df

SCHEME_IDX = 0
TARGET_HOSTS = ['human', 'pig', 'chicken']
TOP_N_GENES = 100
TOP_N_UNION = 15

MODEL_NAME_CANDIDATES = [
    'Ensemble__Nelder_Mead'
]


SKIP_ZERO_WEIGHT_MODELS = True

EXPLAIN_SAMPLE_SIZE = None

pa_data = pd.read_csv("gene_presence_absence_cleaned.csv", index_col=0)
X_all = pa_data.T
X_all.index = X_all.index.to_series().apply(clean_sample_id)

meta_df = pd.read_csv("file_with_host_date_ip.csv")
id_col = 'ID' if 'ID' in meta_df.columns else meta_df.columns[0]
meta_df[id_col] = meta_df[id_col].astype(str).apply(clean_sample_id)
meta_df = meta_df.set_index(id_col)

common_ids = X_all.index.intersection(meta_df.index)
X_all = X_all.loc[common_ids]
meta_df = meta_df.loc[common_ids]

meta_df['TIME_PERIOD'] = meta_df['DATE'].astype(str).str.strip()
meta_df['IP_CLEAN'] = meta_df['IP'].astype(str).str.strip()

counts = meta_df.groupby(['TIME_PERIOD', 'IP_CLEAN']).size()
valid_schemes = counts[counts > 200].index.tolist()

if SCHEME_IDX >= len(valid_schemes):
    raise ValueError(f"SCHEME_IDX={SCHEME_IDX} 超出 valid_schemes 范围。")

final_scheme = valid_schemes[SCHEME_IDX]
time_p, ip_p = final_scheme

is_a = (meta_df['TIME_PERIOD'] == time_p) & (meta_df['IP_CLEAN'] == ip_p)

X_A_df = X_all.loc[is_a]
X_B_df = X_all.loc[~is_a]

out_dir = os.path.join(
    "Ultimate_Binary_Results",
    f"Scheme_{SCHEME_IDX}",
    "Fig5_SHAP_Bset_NelderMead_TreeOnly"
)
os.makedirs(out_dir, exist_ok=True)

topgene_dir = os.path.join(out_dir, "Top100_From_B_SHAP")
os.makedirs(topgene_dir, exist_ok=True)

weight_dir = os.path.join(out_dir, "NelderMead_TreeOnly_Weights_Used")
os.makedirs(weight_dir, exist_ok=True)

raw_dir = os.path.join(out_dir, "Raw_Matrices_For_Replot")
os.makedirs(raw_dir, exist_ok=True)

raw_shap_dir = os.path.join(raw_dir, "SHAP_Matrices")
os.makedirs(raw_shap_dir, exist_ok=True)

raw_x_dir = os.path.join(raw_dir, "X_Explain_Matrices")
os.makedirs(raw_x_dir, exist_ok=True)

raw_summary_dir = os.path.join(raw_dir, "Feature_Summaries")
os.makedirs(raw_summary_dir, exist_ok=True)

raw_pred_dir = os.path.join(raw_dir, "Prediction_Matrices")
os.makedirs(raw_pred_dir, exist_ok=True)

raw_plotdata_dir = os.path.join(raw_dir, "Plot_Data")
os.makedirs(raw_plotdata_dir, exist_ok=True)

raw_bundle_dir = os.path.join(raw_dir, "Joblib_Bundles")
os.makedirs(raw_bundle_dir, exist_ok=True)

fig_d_dir = os.path.join(out_dir, "Fig5_D_CrossHost_Top15_Union_Heatmap")
os.makedirs(fig_d_dir, exist_ok=True)

fig_e_dir = os.path.join(out_dir, "Fig5_E_Top1_Gene_Effect")
os.makedirs(fig_e_dir, exist_ok=True)

saved_model_base_dir = os.path.join(
    "Ultimate_Binary_Results",
    f"Scheme_{SCHEME_IDX}",
    "Saved_Models"
)



host_feature_summary_map = {}
host_top15_map = {}
global_clean_name_map = {}


for target_host in TARGET_HOSTS:
    print(f"\n{'=' * 70}")
    print(f"🎯 正在处理宿主: 【{target_host.upper()}】")
    print(f"{'=' * 70}")

    model_path, resolved_model_name = resolve_model_path(
        base_dir=saved_model_base_dir,
        host_name=target_host,
        candidate_model_names=MODEL_NAME_CANDIDATES
    )

    if model_path is None:
        print(f"⚠️ 未找到 {target_host} 的 Nelder_Mead 模型文件，跳过该宿主。")
        print(f"   期望路径类似: {saved_model_base_dir}/{target_host}/Ensemble__Nelder_Mead.joblib")
        continue

    print(f"✅ 找到模型文件: {model_path}")

    bundle = joblib.load(model_path)

    current_model = bundle["model"]
    current_genes = list(bundle["feature_names"])

    print(f"✅ 成功加载模型 [{resolved_model_name}]")
    print(f"   模型类型: {current_model.__class__.__name__}")
    print(f"   特征数: {len(current_genes)}")

    # 检查特征是否存在
    missing_genes = [g for g in current_genes if g not in X_B_df.columns]
    if len(missing_genes) > 0:
        raise ValueError(
            f"{target_host} 模型中有 {len(missing_genes)} 个特征不在 X_B_df 中，"
            f"例如: {missing_genes[:5]}"
        )

    cleaned_genes = [clean_gene_name(g) for g in current_genes]
    global_clean_name_map.update(dict(zip(current_genes, cleaned_genes)))

    X_B_model_full = X_B_df[current_genes].copy()

    # actual label
    host_col = 'Host' if 'Host' in meta_df.columns else meta_df.columns[1]
    y_B_target = (
        meta_df.loc[X_B_model_full.index, host_col]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(target_host.lower())
        .astype(int)
    )


    X_explain_full = X_B_model_full.copy()

    if EXPLAIN_SAMPLE_SIZE is not None and len(X_explain_full) > EXPLAIN_SAMPLE_SIZE:
        X_explain_for_shap = X_explain_full.sample(
            n=EXPLAIN_SAMPLE_SIZE,
            random_state=42
        )
        print(f"⚠️ B集样本较多，当前仅解释 {len(X_explain_for_shap)} 个 B集样本用于 SHAP。")
    else:
        X_explain_for_shap = X_explain_full

    print(f"✅ SHAP解释对象: B集")
    print(f"✅ SHAP解释样本数: {len(X_explain_for_shap)}")

    try:
        (
            submodel_names,
            submodels,
            original_weights,
            normalized_weights,
            used_weight_df,
            skipped_weight_df
        ) = get_nelder_mead_tree_components(
            nelder_model=current_model,
            skip_zero_weight=SKIP_ZERO_WEIGHT_MODELS
        )

    except Exception as e:
        print(f"❌ 提取 Nelder-Mead 树模型子模型失败: {e}")
        continue

    print("\n✅ 参与 Tree-only SHAP 的 Nelder-Mead 子模型权重如下：")
    for n, m, ow, nw in zip(submodel_names, submodels, original_weights, normalized_weights):
        print(
            f"   - {n}: {m.__class__.__name__} | "
            f"original_weight = {ow:.6f} | "
            f"tree_only_weight = {nw:.6f}"
        )

    if len(skipped_weight_df) > 0:
        print("\n⚠️ 以下模型不参与 SHAP 加权：")
        for _, row in skipped_weight_df.iterrows():
            print(
                f"   - {row['Model']}: {row['Class']} | "
                f"weight = {row['Original_NelderMead_Weight']:.6f} | "
                f"reason = {row['Reason']}"
            )

    used_weight_df.to_csv(
        os.path.join(weight_dir, f"{target_host}_NelderMead_TreeOnly_Weights_Used.csv"),
        index=False
    )

    if len(skipped_weight_df) > 0:
        skipped_weight_df.to_csv(
            os.path.join(weight_dir, f"{target_host}_NelderMead_TreeOnly_Weights_Skipped.csv"),
            index=False
        )


    shap_list = []
    valid_names = []
    valid_weights = []

    for sub_name, sub_model, sub_weight in zip(submodel_names, submodels, normalized_weights):
        try:
            sub_shap = compute_tree_model_shap(
                model_name=sub_name,
                model=sub_model,
                X_explain_df=X_explain_for_shap
            )

            shap_list.append(sub_shap)
            valid_names.append(sub_name)
            valid_weights.append(sub_weight)

            print(f"   ✅ {sub_name} SHAP 完成，shape = {sub_shap.shape}")

        except Exception as e:
            print(f"⚠️ {sub_name} 的 SHAP 计算失败，已跳过。原因: {e}")

    if len(shap_list) == 0:
        print("❌ 没有任何树模型成功产出 SHAP，跳过该宿主。")
        continue

    valid_weights = np.asarray(valid_weights, dtype=float)
    valid_weights = valid_weights / valid_weights.sum()

    print("\n✅ 最终参与 Tree-only SHAP 加权平均的模型：")
    for n, w in zip(valid_names, valid_weights):
        print(f"   - {n}: final_tree_shap_weight = {w:.6f}")

    final_weight_df = pd.DataFrame({
        "Model": valid_names,
        "Final_TreeOnly_SHAP_Weight": valid_weights
    })
    final_weight_df.to_csv(
        os.path.join(weight_dir, f"{target_host}_Final_TreeOnly_SHAP_Weights_After_Failures.csv"),
        index=False
    )

    
    shap_values = np.average(
        np.stack(shap_list, axis=0),
        axis=0,
        weights=valid_weights
    )

    if not np.isfinite(shap_values).all():
        print(f"❌ {target_host} 最终加权 SHAP 中存在 NaN/Inf，跳过。")
        continue

    print(f"\n✅ B-set Nelder-Mead Tree-only 加权 SHAP 完成，shape = {shap_values.shape}")
    print(f"   Final SHAP min = {np.min(shap_values):.6g}")
    print(f"   Final SHAP max = {np.max(shap_values):.6g}")
    print(f"   Final mean(|SHAP|) = {np.mean(np.abs(shap_values)):.6g}")
    print(f"   Final p99(|SHAP|) = {np.percentile(np.abs(shap_values), 99):.6g}")

    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_signed_shap = np.mean(shap_values, axis=0)

    shap_df = pd.DataFrame(
        shap_values,
        index=X_explain_for_shap.index,
        columns=current_genes
    )

    feature_summary_df = pd.DataFrame({
        "Feature": current_genes,
        "Clean_Name": cleaned_genes,
        "MeanAbsSHAP_Bset_TreeOnly": mean_abs_shap,
        "MeanSignedSHAP_Bset_TreeOnly": mean_signed_shap
    }).sort_values(
        by="MeanAbsSHAP_Bset_TreeOnly",
        ascending=False
    ).reset_index(drop=True)

    shap_df.to_csv(
        os.path.join(raw_shap_dir, f"{target_host}_Bset_TreeOnly_SHAP_Matrix.csv.gz"),
        compression='gzip'
    )

    X_explain_for_shap.to_csv(
        os.path.join(raw_x_dir, f"{target_host}_Bset_TreeOnly_X_Explain.csv.gz"),
        compression='gzip'
    )

    feature_summary_df.to_csv(
        os.path.join(raw_summary_dir, f"{target_host}_Bset_TreeOnly_Feature_Summary.csv"),
        index=False
    )

    joblib.dump(
        {
            "target_host": target_host,
            "sample_ids": list(X_explain_for_shap.index),
            "feature_names_raw": list(current_genes),
            "feature_names_clean": list(cleaned_genes),
            "shap_values": shap_values,
            "X_explain_for_shap": X_explain_for_shap,
            "mean_abs_shap": mean_abs_shap,
            "mean_signed_shap": mean_signed_shap,
            "valid_model_names": valid_names,
            "valid_model_weights": valid_weights
        },
        os.path.join(raw_bundle_dir, f"{target_host}_Bset_TreeOnly_SHAP_Bundle.joblib"),
        compress=3
    )

    print(f"💾 {target_host} 原始 SHAP 矩阵和特征汇总已保存。")


    top100_idx = np.argsort(mean_abs_shap)[::-1][:TOP_N_GENES]
    top100_raw = [current_genes[i] for i in top100_idx]
    top100_clean = [cleaned_genes[i] for i in top100_idx]
    top100_scores = mean_abs_shap[top100_idx]

    df_top100 = pd.DataFrame({
        "Rank": np.arange(1, len(top100_raw) + 1),
        "Feature": top100_raw,
        "Clean_Name": top100_clean,
        "MeanAbsSHAP_Bset_TreeOnly": top100_scores
    })

    csv_path = os.path.join(
        topgene_dir,
        f"{target_host}_Bset_NelderMead_TreeOnly_SHAP_Top{TOP_N_GENES}.csv"
    )
    joblib_path = os.path.join(
        topgene_dir,
        f"{target_host}_Bset_NelderMead_TreeOnly_SHAP_Top{TOP_N_GENES}.joblib"
    )

    df_top100.to_csv(csv_path, index=False)
    joblib.dump(top100_raw, joblib_path)

    print(f"\n✅ {target_host} 的 B集 Tree-only SHAP Top{TOP_N_GENES} 基因已保存:")
    print(f"   CSV: {csv_path}")
    print(f"   JOBLIB: {joblib_path}")

    # 保存 Top15
    df_top15 = df_top100.head(TOP_N_UNION).copy()
    df_top15.to_csv(
        os.path.join(topgene_dir, f"{target_host}_Bset_NelderMead_TreeOnly_SHAP_Top{TOP_N_UNION}.csv"),
        index=False
    )

    host_feature_summary_map[target_host] = feature_summary_df.copy()
    host_top15_map[target_host] = df_top15["Feature"].tolist()

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_explain_for_shap,
        feature_names=cleaned_genes,
        show=False,
        max_display=15
    )

    plt.title(
        f"Part B: Top SHAP Genes Driving '{target_host.capitalize()}' Prediction\n"
        f"(B-set Nelder-Mead Tree-only SHAP; SVM and MLP Excluded)",
        fontsize=14,
        fontweight='bold',
        y=1.05
    )

    plt.tight_layout()

    fig_b_path = os.path.join(
        out_dir,
        f"Fig5_B_{target_host}_Bset_NelderMead_TreeOnly_SHAP_Summary.png"
    )

    plt.savefig(
        fig_b_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


    top_indices = np.argsort(mean_abs_shap)[::-1][:10]

    top_genes_raw = [current_genes[i] for i in top_indices]
    top_genes_clean = [cleaned_genes[i] for i in top_indices]

    plot_data = []
    host_series_lower = meta_df['Host'].fillna('').astype(str).str.lower()

    for raw_g, clean_g in zip(top_genes_raw, top_genes_clean):
        for host_name in TARGET_HOSTS:
            host_samples = meta_df[host_series_lower == host_name].index

            # 只统计 B集中的该宿主样本
            valid_samples = X_B_df.index.intersection(host_samples)

            if len(valid_samples) > 0:
                prevalence = X_B_df.loc[valid_samples, raw_g].mean() * 100
            else:
                prevalence = np.nan

            plot_data.append({
                "Gene": clean_g,
                "Raw_Feature": raw_g,
                "Host": host_name.capitalize(),
                "Bset_N": len(valid_samples),
                "Prevalence (%)": prevalence
            })

    df_plot_c = pd.DataFrame(plot_data)

    fig_c_data_path = os.path.join(
        raw_plotdata_dir,
        f"Fig5_C_{target_host}_Bset_NelderMead_TreeOnly_Top10_Genes_Data.csv"
    )
    df_plot_c.to_csv(fig_c_data_path, index=False)

    plt.figure(figsize=(14, 6))

    sns.barplot(
        data=df_plot_c,
        x="Gene",
        y="Prevalence (%)",
        hue="Host",
        palette={
            "Human": "tab:blue",
            "Pig": "tab:red",
            "Chicken": "tab:green"
        }
    )

    plt.title(
        f"Part C: B-set Prevalence of Top 10 Key Genes\n"
        f"(Selected by {target_host.capitalize()} B-set Nelder-Mead Tree-only SHAP)",
        fontsize=16,
        fontweight='bold'
    )

    plt.xlabel("Top Features (Genes)", fontsize=12)
    plt.ylabel("Gene Prevalence Rate in B set (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.legend(title="Actual Host", loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fig_c_path = os.path.join(
        out_dir,
        f"Fig5_C_{target_host}_Bset_NelderMead_TreeOnly_Gene_Prevalence.png"
    )

    plt.savefig(
        fig_c_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print(f"📊 {target_host} Fig5_C Tree-only Top10 检出率图已保存: {fig_c_path}")
    print(f"📄 {target_host} Fig5_C 数据已保存: {fig_c_data_path}")

    pred_probs, pred_used_weight_df, pred_skipped_weight_df = predict_proba_nelder_tree_only(
        current_model,
        X_B_model_full
    )

    pred_used_weight_df.to_csv(
        os.path.join(weight_dir, f"{target_host}_TreeOnly_Prediction_Weights_Used.csv"),
        index=False
    )

    if len(pred_skipped_weight_df) > 0:
        pred_skipped_weight_df.to_csv(
            os.path.join(weight_dir, f"{target_host}_TreeOnly_Prediction_Weights_Skipped.csv"),
            index=False
        )

    pred_df = pd.DataFrame({
        "SampleID": X_B_model_full.index,
        "Predicted_Probability": pred_probs,
        "Actual_Label": y_B_target.values
    })

    pred_df.to_csv(
        os.path.join(raw_pred_dir, f"{target_host}_Bset_TreeOnly_Prediction_Probability.csv"),
        index=False
    )

    top1_raw = df_top100.iloc[0]["Feature"]
    top1_clean = df_top100.iloc[0]["Clean_Name"]

    top1_state = X_B_model_full[top1_raw].astype(int)

    df_plot_e = pred_df.copy()
    df_plot_e["Top1_Gene_Raw"] = top1_raw
    df_plot_e["Top1_Gene_Clean"] = top1_clean
    df_plot_e["Gene_State_Code"] = top1_state.values
    df_plot_e["Gene_State_Label"] = np.where(top1_state.values == 1, "Present", "Absent")

    df_plot_e.to_csv(
        os.path.join(raw_plotdata_dir, f"Fig5_E_{target_host}_Top1_Gene_Effect_Data.csv"),
        index=False
    )

    if df_plot_e["Gene_State_Label"].nunique() >= 2:
        plt.figure(figsize=(8, 6))

        sns.violinplot(
            data=df_plot_e,
            x="Gene_State_Label",
            y="Predicted_Probability",
            order=["Absent", "Present"],
            palette=["#d9d9d9", "#67a9cf"],
            inner=None,
            linewidth=0,
            alpha=0.5
        )

        sns.boxplot(
            data=df_plot_e,
            x="Gene_State_Label",
            y="Predicted_Probability",
            order=["Absent", "Present"],
            color="white",
            width=0.2,
            boxprops={'zorder': 2, 'alpha': 0.85},
            showfliers=False
        )

        sns.stripplot(
            data=df_plot_e,
            x="Gene_State_Label",
            y="Predicted_Probability",
            order=["Absent", "Present"],
            color="black",
            alpha=0.15,
            size=2,
            jitter=True,
            zorder=3
        )

        plt.title(
            f"Fig.5E: Effect of Top1 SHAP Gene on Prediction\n"
            f"{target_host.capitalize()} | {top1_clean}",
            fontsize=14,
            fontweight='bold'
        )
        plt.xlabel(f"{top1_clean} Presence/Absence", fontsize=12, fontweight='bold')
        plt.ylabel(
            f"Predicted Probability ({target_host.capitalize()})",
            fontsize=12,
            fontweight='bold'
        )
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        fig_e_path = os.path.join(
            fig_e_dir,
            f"Fig5_E_{target_host}_Top1_Gene_Presence_Effect.png"
        )

        plt.savefig(
            fig_e_path,
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        print(f"📊 {target_host} Fig5_E Top1基因影响图已保存: {fig_e_path}")
    else:
        print(f"⚠️ {target_host} 的 Top1 基因在B集中只有一个状态，跳过 Fig5_E 作图。")



available_hosts = [h for h in TARGET_HOSTS if h in host_feature_summary_map and h in host_top15_map]

if len(available_hosts) == 0:
    print("⚠️ 没有可用宿主结果，无法构建 Fig5_D。")
else:
    union_features = []
    for h in TARGET_HOSTS:
        if h in host_top15_map:
            for g in host_top15_map[h]:
                if g not in union_features:
                    union_features.append(g)

    union_clean = [global_clean_name_map.get(g, clean_gene_name(g)) for g in union_features]

  
    heatmap_raw = pd.DataFrame(
        index=[h.capitalize() for h in TARGET_HOSTS],
        columns=union_features,
        dtype=float
    )

    availability_raw = pd.DataFrame(
        index=[h.capitalize() for h in TARGET_HOSTS],
        columns=union_features,
        dtype=float
    )

    membership_df = pd.DataFrame({
        "Feature": union_features,
        "Clean_Name": union_clean
    })

    for h in TARGET_HOSTS:
        membership_df[f"Selected_in_{h.capitalize()}_Top15"] = [
            1 if (h in host_top15_map and g in host_top15_map[h]) else 0
            for g in union_features
        ]

    for h in TARGET_HOSTS:
        if h in host_feature_summary_map:
            temp_df = host_feature_summary_map[h].set_index("Feature")
            for g in union_features:
                if g in temp_df.index:
                    heatmap_raw.loc[h.capitalize(), g] = temp_df.loc[g, "MeanSignedSHAP_Bset_TreeOnly"]
                    availability_raw.loc[h.capitalize(), g] = 1
                else:
                    heatmap_raw.loc[h.capitalize(), g] = np.nan
                    availability_raw.loc[h.capitalize(), g] = 0
        else:
            heatmap_raw.loc[h.capitalize(), :] = np.nan
            availability_raw.loc[h.capitalize(), :] = 0

    heatmap_plot = heatmap_raw.fillna(0.0).copy()
    heatmap_plot.columns = union_clean

    heatmap_raw.to_csv(
        os.path.join(fig_d_dir, "Fig5_D_Top15_Union_SignedSHAP_Heatmap_RawFeatureNames.csv")
    )
    heatmap_plot.to_csv(
        os.path.join(fig_d_dir, "Fig5_D_Top15_Union_SignedSHAP_Heatmap_CleanNames.csv")
    )
    availability_raw.to_csv(
        os.path.join(fig_d_dir, "Fig5_D_Top15_Union_Gene_Availability_Matrix.csv")
    )
    membership_df.to_csv(
        os.path.join(fig_d_dir, "Fig5_D_Top15_Union_Gene_Membership.csv"),
        index=False
    )

    vmax = np.nanmax(np.abs(heatmap_plot.values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6

    fig_width = max(14, len(union_features) * 0.55)
    plt.figure(figsize=(fig_width, 4.8))

    sns.heatmap(
        heatmap_plot,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean signed SHAP"}
    )

    plt.title(
        "Fig.5D: Signed SHAP Heatmap of Union of Top15 Genes Across Hosts\n"
        "(B-set Nelder-Mead Tree-only SHAP)",
        fontsize=15,
        fontweight='bold'
    )
    plt.xlabel("Union of Host-specific Top15 SHAP Genes", fontsize=12, fontweight='bold')
    plt.ylabel("Host", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    fig_d_path = os.path.join(
        fig_d_dir,
        "Fig5_D_Top15_Union_SignedSHAP_Heatmap.png"
    )
    plt.savefig(
        fig_d_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


