import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTEN
from boruta import BorutaPy
import warnings
warnings.filterwarnings('ignore')

METRIC_ORDER = ["Accuracy", "Macro F1", "MCC", "PR-AUC", "ROC-AUC"]


def get_positive_class_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if np.ndim(proba) == 1:
            return np.asarray(proba).ravel()
        if hasattr(model, "classes_"):
            classes_list = list(model.classes_)
            pos_idx = classes_list.index(1) if 1 in classes_list else -1
            return proba[:, pos_idx]
        return proba[:, -1]

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        score = np.asarray(score).ravel()
        if np.max(score) > np.min(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
        else:
            score = np.zeros_like(score, dtype=float)
        return score

    return np.asarray(model.predict(X)).astype(float)


def compute_binary_metrics(y_true, y_pred, y_score):
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average='macro'),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    try:
        result["PR-AUC"] = average_precision_score(y_true, y_score)
    except Exception:
        result["PR-AUC"] = np.nan

    try:
        result["ROC-AUC"] = roc_auc_score(y_true, y_score)
    except Exception:
        result["ROC-AUC"] = np.nan

    return result


def append_metric_records(record_list, target_host, model_name, model_type, metrics_dict, fold=None):
    for metric_name in METRIC_ORDER:
        row = {
            "Target_Host": target_host,
            "Model": model_name,
            "Model_Type": model_type,
            "Metric": metric_name,
            "Score": metrics_dict.get(metric_name, np.nan)
        }
        if fold is not None:
            row["Fold"] = fold
        record_list.append(row)


def plot_cv_multi_metric_boxplots(df_cv, scheme_idx, display_scheme, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    legend_handles, legend_labels = None, None

    for i, metric in enumerate(METRIC_ORDER):
        ax = axes[i]
        sub_df = df_cv[df_cv["Metric"] == metric].copy()

        if sub_df.empty:
            ax.axis("off")
            continue

        sns.boxplot(
            data=sub_df,
            x="Model",
            y="Score",
            hue="Target_Host",
            palette="Set2",
            ax=ax
        )

        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=45)

        if metric == "MCC":
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.set_ylim(-0.05, 1.05)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[5].axis("off")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Target Host",
            bbox_to_anchor=(0.98, 0.5),
            loc='center left'
        )

    fig.suptitle(
        f"Plot 1: 5-Fold CV Stability (Multi-Metric) [Scheme {scheme_idx}: {display_scheme}]",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


class NelderMeadEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators
        self.weights_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_preds = np.zeros((X.shape[0], len(self.classes_), len(self.estimators)))

        for i, (name, model) in enumerate(self.estimators):
            oof_preds[:, :, i] = cross_val_predict(
                model, X, y, cv=cv, method='predict_proba', n_jobs=5
            )

        def objective(w):
            w = np.clip(w, 0, 1)
            if np.sum(w) == 0:
                w = np.ones(len(w))
            w /= np.sum(w)
            blended = np.average(oof_preds, axis=2, weights=w)
            preds = self.classes_[np.argmax(blended, axis=1)]
            return -f1_score(y, preds, average='macro')

        initial_weights = np.ones(len(self.estimators)) / len(self.estimators)
        res = minimize(
            objective,
            initial_weights,
            method='Nelder-Mead',
            options={'maxiter': 200}
        )
        self.weights_ = np.clip(res.x, 0, 1)
        self.weights_ /= np.sum(self.weights_)
        return self

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for _, m in self.estimators])
        return np.average(probs, axis=0, weights=self.weights_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


pa_data = pd.read_csv("gene_presence_absence_cleaned.csv", index_col=0)
X_all = pa_data.T
X_all.index = X_all.index.str.replace(r'\.fna.*$', '', regex=True).str.strip()

meta_df = pd.read_csv("file_with_host_date_ip.csv")
id_col = 'ID' if 'ID' in meta_df.columns else meta_df.columns[0]
meta_df[id_col] = meta_df[id_col].astype(str).str.replace(r'\.fna.*$', '', regex=True).str.strip()
meta_df = meta_df.set_index(id_col)

common_ids = X_all.index.intersection(meta_df.index)
X_all = X_all.loc[common_ids]
meta_df = meta_df.loc[common_ids]

host_labels_raw = meta_df['Host'].fillna('Unknown').values
label_encoder = LabelEncoder()
y_all_encoded = label_encoder.fit_transform(host_labels_raw)
host_classes = [c for c in label_encoder.classes_ if str(c) not in ['Unknown', 'nan']]

meta_df['TIME_PERIOD'] = meta_df['DATE'].astype(str).str.strip()
meta_df['IP_CLEAN'] = meta_df['IP'].astype(str).str.strip()
counts = meta_df.groupby(['TIME_PERIOD', 'IP_CLEAN']).size()
valid_schemes = counts[counts > 200].index.tolist()

base_out_dir = "Ultimate_Binary_Results_fig4_modified"
os.makedirs(base_out_dir, exist_ok=True)

target_schemes = [0]

for scheme_idx in target_schemes:
    if scheme_idx >= len(valid_schemes):
        print(f"⚠️ 警告: 索引 {scheme_idx} 超出有效时空分法数量，已跳过。")
        continue

    final_scheme = valid_schemes[scheme_idx]
    time_p, ip_p = final_scheme
    display_scheme = f"{time_p} ({ip_p})"

    scheme_dir = os.path.join(base_out_dir, f"Scheme_{scheme_idx}")
    os.makedirs(scheme_dir, exist_ok=True)

    is_a = (meta_df['TIME_PERIOD'] == time_p) & (meta_df['IP_CLEAN'] == ip_p)
    X_B_df = X_all[~is_a]

    print(f"\n{'#' * 80}")
    print(f"🚀 [启动时空分法] Scheme {scheme_idx}: {display_scheme}")
    print(f"   B集: {len(X_B_df)} 样本")
    print(f"   输出目录: {scheme_dir}")
    print(f"{'#' * 80}")

    cv_records = []

    for target_host in host_classes:
        print(f"\n{'=' * 60}")
        print(f"⚔️ 战役打响: 【{target_host}】 (Scheme {scheme_idx})")
        print(f"{'=' * 60}")

        target_idx = label_encoder.transform([target_host])[0]
        y_all_binary = (y_all_encoded == target_idx).astype(int)
        y_B_binary = y_all_binary[~is_a]

        rf_boruta_cv = RandomForestClassifier(
            n_jobs=9,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
        boruta_cv = BorutaPy(
            rf_boruta_cv,
            n_estimators='auto',
            verbose=0,
            random_state=42,
            max_iter=35
        )
        boruta_cv.fit(X_B_df.values, y_B_binary)

        sel_genes_cv = (
            X_B_df.columns[boruta_cv.support_].tolist()
            or X_B_df.columns[boruta_cv.support_weak_].tolist()
            or X_B_df.columns.tolist()
        )

        X_B_cv_sel = X_B_df[sel_genes_cv]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_B_cv_sel, y_B_binary)):
            print(f"\n   >>> 正在执行 Fold {fold_idx + 1}/5 ...")

            X_tr_f = X_B_cv_sel.iloc[train_idx].values
            y_tr_f = y_B_binary[train_idx]
            X_val_f = X_B_cv_sel.iloc[val_idx].values
            y_val_f = y_B_binary[val_idx]

            print(f"       [1/3] SMOTEN 数据平衡...")
            sm = SMOTEN(random_state=42, k_neighbors=3)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_f, y_tr_f)

            print(f"       [2/3] 模型调参...")
            lgbm_search = RandomizedSearchCV(
                lgb.LGBMClassifier(n_jobs=5, verbose=-1, random_state=42),
                {'learning_rate': [0.01, 0.05], 'num_leaves': [31, 63]},
                n_iter=3,
                cv=3,
                scoring='f1_macro',
                random_state=42
            ).fit(X_tr_res, y_tr_res)

            xgb_search = RandomizedSearchCV(
                xgb.XGBClassifier(n_jobs=5, eval_metric='logloss', random_state=42),
                {'learning_rate': [0.01, 0.05], 'max_depth': [4, 6]},
                n_iter=3,
                cv=3,
                scoring='f1_macro',
                random_state=42
            ).fit(X_tr_res, y_tr_res)

            base_models = [
                ("Tuned_LGBM", lgbm_search.best_estimator_),
                ("Tuned_XGB", xgb_search.best_estimator_),
                ("RF", RandomForestClassifier(n_estimators=300, n_jobs=9, random_state=42)),
                ("SVM", SVC(kernel='linear', C=1.0, probability=True, random_state=42)),
                ("MLP", MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, early_stopping=True, random_state=42))
            ]

            fitted_base = []
            for name, m in base_models:
                clf = clone(m).fit(X_tr_res, y_tr_res)
                fitted_base.append((name, clf))

                preds = clf.predict(X_val_f)
                scores = get_positive_class_score(clf, X_val_f)
                metrics_dict = compute_binary_metrics(y_val_f, preds, scores)

                append_metric_records(
                    cv_records,
                    target_host=target_host,
                    model_name=name,
                    model_type="Base",
                    metrics_dict=metrics_dict,
                    fold=fold_idx + 1
                )

            print(f"       [3/3] 训练集成模型...")
            ensembles = {
                "Hard_Voting": VotingClassifier(estimators=fitted_base, voting='hard'),
                "Soft_Voting": VotingClassifier(estimators=fitted_base, voting='soft'),
                "Weighted_Soft": VotingClassifier(estimators=fitted_base, voting='soft', weights=[1, 2, 0, 0, 0]),
                "Stacking": StackingClassifier(
                    estimators=fitted_base,
                    final_estimator=RandomForestClassifier(max_depth=3, random_state=42)
                ),
                "Nelder_Mead": NelderMeadEnsemble(estimators=fitted_base)
            }

            for ens_name, ens_model in ensembles.items():
                ens_model.fit(X_tr_res, y_tr_res)

                preds = ens_model.predict(X_val_f)
                scores = get_positive_class_score(ens_model, X_val_f)
                metrics_dict = compute_binary_metrics(y_val_f, preds, scores)

                append_metric_records(
                    cv_records,
                    target_host=target_host,
                    model_name=ens_name,
                    model_type="Ensemble",
                    metrics_dict=metrics_dict,
                    fold=fold_idx + 1
                )

    print(f"\n📊 正在生成 Scheme {scheme_idx} 的 B集 CV 结果并保存至 {scheme_dir}/ ...")
    sns.set_theme(style="whitegrid", font_scale=0.9)

    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv(
        os.path.join(scheme_dir, "Plot1_CV_MultiMetric_LongTable.csv"),
        index=False
    )

    cv_mean_df = (
        df_cv
        .groupby(["Target_Host", "Model", "Model_Type", "Metric"], as_index=False)["Score"]
        .mean()
        .rename(columns={"Score": "Mean_Score"})
    )
    cv_mean_df.to_csv(
        os.path.join(scheme_dir, "CV_MultiMetric_MeanTable.csv"),
        index=False
    )

    best_model_df = (
        cv_mean_df[cv_mean_df["Metric"] == "Macro F1"]
        .sort_values(["Target_Host", "Mean_Score"], ascending=[True, False])
        .groupby("Target_Host", as_index=False)
        .first()
        .rename(columns={"Model": "Best_Model_By_CV_MacroF1", "Mean_Score": "Best_CV_MacroF1"})
    )
    best_model_df.to_csv(
        os.path.join(scheme_dir, "Best_Model_By_CV_MacroF1.csv"),
        index=False
    )

    plot_cv_multi_metric_boxplots(
        df_cv=df_cv,
        scheme_idx=scheme_idx,
        display_scheme=display_scheme,
        out_path=os.path.join(scheme_dir, "Plot1_CV_MultiMetric_Boxplots.png")
    )

print(f"\n🎉 运行完成！Scheme {target_schemes} 的 B集 5-fold CV、Plot1 和 CV 结果表均已完成！")
