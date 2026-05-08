import os
import re
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_fscore_support, make_scorer
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
SCHEME_IDX = 0
TARGET_HOSTS = ['human', 'pig', 'chicken']
HOST_COLORS = {'human': '#E41A1C', 'pig': '#377EB8', 'chicken': '#FFD700'}
RANDOM_STATE = 42
N_SPLITS_OUTER = 5
N_SPLITS_META = 5
TOP_N_GENES = 100
FIG7A_BINARY_THRESHOLDS = {'human': 0.5, 'pig': 0.5, 'chicken': 0.3}
DO_CHICKEN_OVERSAMPLING = True
CHICKEN_BINARY_POS_RATIO = 0.5
DO_META_CHICKEN_OVERSAMPLING = True
DO_BINARY_HYPERPARAM_SEARCH = True
DO_META_HYPERPARAM_SEARCH = True
N_ITER_BINARY_SEARCH = 20
N_ITER_META_SEARCH = 30
BINARY_TUNING_CV = 3
META_TUNING_CV = 5
SEARCH_N_JOBS = 4
TUNING_SCORING_BY_HOST = {'human': 'average_precision', 'pig': 'average_precision', 'chicken': 'f1'}
PA_MATRIX_PATH = 'gene_presence_absence_cleaned.csv'
META_PATH = 'file_with_host_date_ip.csv'
OUT_DIR = os.path.join('Ultimate_Binary_Results', f'Scheme_{SCHEME_IDX}', 'Fig7_TunedBase_StackedLGBMMeta_ChickenRandomOversampling')
FIGURE_DIR = os.path.join(OUT_DIR, 'Figures')
TABLE_DIR = os.path.join(OUT_DIR, 'Tables')
MODEL_DIR = os.path.join(OUT_DIR, 'Models')
PLOT_DATA_DIR = os.path.join(OUT_DIR, 'Plot_Data')
SEARCH_DIR = os.path.join(OUT_DIR, 'Hyperparameter_Search')
for d in [OUT_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR, PLOT_DATA_DIR, SEARCH_DIR]:
    os.makedirs(d, exist_ok=True)
FIG5_BASE_DIR = os.path.join('Ultimate_Binary_Results', f'Scheme_{SCHEME_IDX}', 'Fig5_SHAP_Bset_NelderMead_TreeOnly')
FIG5_TOP_DIR = os.path.join(FIG5_BASE_DIR, 'Top100_From_B_SHAP')
FIG6_TABLE_DIR = os.path.join('Ultimate_Binary_Results', f'Scheme_{SCHEME_IDX}', 'Fig6_Bset_Top100_BootstrapSampleOnly_NoRefit', 'Tables')

class NelderMeadEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators, oversample_positive=False, target_pos_ratio=0.5, random_state=42):
        self.estimators = estimators
        self.oversample_positive = oversample_positive
        self.target_pos_ratio = target_pos_ratio
        self.random_state = random_state
        self.weights_ = None
        self.classes_ = None

    def _align_proba(self, proba, model_classes):
        model_classes = list(model_classes)
        aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=float)
        for i, cls in enumerate(self.classes_):
            if cls in model_classes:
                src_idx = model_classes.index(cls)
                aligned[:, i] = proba[:, src_idx]
            else:
                aligned[:, i] = 0.0
        return aligned

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_arr = np.asarray(y).astype(int)
        unique_classes, class_counts = np.unique(y_arr, return_counts=True)
        if len(unique_classes) < 2:
            raise ValueError('NelderMeadEnsemble 训练数据中只有一个类别，无法训练。')
        min_class_count = int(class_counts.min())
        n_splits = min(3, min_class_count)
        if n_splits < 2:
            self.weights_ = np.ones(len(self.estimators), dtype=float)
            self.weights_ = self.weights_ / self.weights_.sum()
            return self
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((len(y_arr), len(self.classes_), len(self.estimators)))
        for i, (name, model) in enumerate(self.estimators):
            for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y_arr), start=1):
                if isinstance(X, pd.DataFrame):
                    X_tr = X.iloc[tr_idx].copy()
                    X_va = X.iloc[va_idx].copy()
                else:
                    X_tr = X[tr_idx]
                    X_va = X[va_idx]
                y_tr = y_arr[tr_idx]
                if self.oversample_positive:
                    X_tr, y_tr = random_oversample_binary_positive(X_tr, y_tr, target_pos_ratio=self.target_pos_ratio, random_state=self.random_state + fold_idx + i * 100)
                m = clone(model)
                m.fit(X_tr, y_tr)
                proba_va = m.predict_proba(X_va)
                proba_va = self._align_proba(proba_va, m.classes_)
                oof_preds[va_idx, :, i] = proba_va

        def objective(w):
            w = np.clip(w, 0, 1)
            if np.sum(w) == 0:
                w = np.ones(len(w), dtype=float)
            w = w / np.sum(w)
            blended = np.average(oof_preds, axis=2, weights=w)
            preds = self.classes_[np.argmax(blended, axis=1)]
            return -f1_score(y_arr, preds, average='macro', zero_division=0)
        initial_weights = np.ones(len(self.estimators)) / len(self.estimators)
        res = minimize(objective, initial_weights, method='Nelder-Mead', options={'maxiter': 200})
        self.weights_ = np.clip(res.x, 0, 1)
        if np.sum(self.weights_) == 0:
            self.weights_ = np.ones(len(self.estimators), dtype=float)
        self.weights_ = self.weights_ / np.sum(self.weights_)
        return self

    def predict_proba(self, X):
        probs = []
        for _, m in self.estimators:
            p = m.predict_proba(X)
            if p.shape[1] == 2:
                probs.append(p)
            else:
                raise ValueError(f'模型 {m.__class__.__name__} predict_proba 不是二分类两列输出: {p.shape}')
        probs = np.array(probs)
        return np.average(probs, axis=0, weights=self.weights_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

def clean_sample_id(sample_id):
    return re.sub('\\.fna.*$', '', str(sample_id)).strip()

def sanitize_name(text):
    return str(text).replace(' ', '_').replace('/', '_').replace('\\', '_')

def clean_host_name(x):
    return str(x).strip().lower()

def clean_gene_name(raw_name):
    if '~~~' in str(raw_name):
        return f"{str(raw_name).split('~~~')[0]} cluster"
    return str(raw_name)

def random_oversample_binary_positive(X, y, target_pos_ratio=0.5, random_state=42):
    y_arr = np.asarray(y).astype(int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return (X, y_arr)
    current_pos = len(pos_idx)
    current_neg = len(neg_idx)
    desired_pos = int(np.ceil(target_pos_ratio / max(1e-08, 1.0 - target_pos_ratio) * current_neg))
    if desired_pos <= current_pos:
        return (X, y_arr)
    rng = np.random.default_rng(random_state)
    extra_n = desired_pos - current_pos
    extra_pos_idx = rng.choice(pos_idx, size=extra_n, replace=True)
    new_idx = np.concatenate([np.arange(len(y_arr)), extra_pos_idx])
    rng.shuffle(new_idx)
    if isinstance(X, pd.DataFrame):
        X_resampled = X.iloc[new_idx].copy()
    else:
        X_resampled = X[new_idx]
    y_resampled = y_arr[new_idx]
    return (X_resampled, y_resampled)

def random_oversample_multiclass_chicken(X, y, chicken_label='chicken', random_state=42):
    y_series = pd.Series(y, index=X.index).astype(str).str.lower()
    counts = y_series.value_counts()
    if chicken_label not in counts.index:
        return (X, y_series.values)
    max_count = int(counts.max())
    chicken_count = int(counts[chicken_label])
    if chicken_count >= max_count:
        return (X, y_series.values)
    chicken_ids = y_series[y_series == chicken_label].index
    rng = np.random.default_rng(random_state)
    extra_n = max_count - chicken_count
    extra_ids = rng.choice(chicken_ids, size=extra_n, replace=True)
    X_extra = X.loc[extra_ids].copy()
    y_extra = pd.Series([chicken_label] * extra_n, index=X_extra.index)
    X_resampled = pd.concat([X, X_extra], axis=0)
    y_resampled = pd.concat([y_series, y_extra], axis=0)
    shuffle_order = rng.permutation(len(X_resampled))
    X_resampled = X_resampled.iloc[shuffle_order].copy()
    y_resampled = y_resampled.iloc[shuffle_order].values
    return (X_resampled, y_resampled)

def savefig_png_pdf(fig, out_prefix, dpi=600):
    png_path = out_prefix + '.png'
    pdf_path = out_prefix + '.pdf'
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', facecolor='white')

def get_base_prob_cols():
    return [f'P_{h}' for h in TARGET_HOSTS]

def get_meta_prob_cols():
    return [f'Meta_P_{h}' for h in TARGET_HOSTS]

def load_top_features_for_host(host_name, X_columns):
    fig5_joblib_path = os.path.join(FIG5_TOP_DIR, f'{host_name}_Bset_NelderMead_TreeOnly_SHAP_Top{TOP_N_GENES}.joblib')
    if os.path.exists(fig5_joblib_path):
        features = list(joblib.load(fig5_joblib_path))
        source = fig5_joblib_path
    else:
        fig6_csv_path = os.path.join(FIG6_TABLE_DIR, f'Fig6_{host_name}_Used_Top100_Features.csv')
        if os.path.exists(fig6_csv_path):
            df = pd.read_csv(fig6_csv_path)
            if 'Feature' not in df.columns:
                raise ValueError(f'{fig6_csv_path} 中没有 Feature 列。')
            features = df['Feature'].astype(str).tolist()
            source = fig6_csv_path
        else:
            raise FileNotFoundError(f'未找到 {host_name} 的 Top100 genes。\n优先查找:\n{fig5_joblib_path}\n备用查找:\n{fig6_csv_path}')
    features = [f for f in features if f in X_columns]
    features = features[:TOP_N_GENES]
    if len(features) == 0:
        raise ValueError(f'{host_name} 的 Top100 genes 在当前矩阵中一个都找不到。')
    return (features, source)

def set_model_random_state_and_jobs(model, random_state, n_jobs=-1):
    m = clone(model)
    try:
        params = m.get_params()
    except Exception:
        return m
    update_params = {}
    model_class = m.__class__.__name__.lower()
    if 'random_state' in params:
        update_params['random_state'] = random_state
    if 'n_jobs' in params:
        update_params['n_jobs'] = n_jobs
    if 'xgb' in model_class or 'xgboost' in model_class:
        if 'verbosity' in params:
            update_params['verbosity'] = 0
        if 'eval_metric' in params:
            update_params['eval_metric'] = 'logloss'
    elif 'lgbm' in model_class or 'lightgbm' in model_class:
        if 'verbosity' in params:
            update_params['verbosity'] = -1
        if 'verbose' in params:
            update_params['verbose'] = -1
    elif 'verbose' in params:
        update_params['verbose'] = 0
    try:
        m.set_params(**update_params)
    except Exception as e:
        pass
    return m

def get_binary_cv(y, max_splits=3, random_state=42):
    y_arr = np.asarray(y).astype(int)
    _, counts = np.unique(y_arr, return_counts=True)
    min_count = int(counts.min())
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        raise ValueError('当前二分类任务某一类样本太少，无法做 CV。')
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def get_multiclass_cv(y, max_splits=5, random_state=42):
    y_series = pd.Series(y).astype(str).str.lower()
    min_count = int(y_series.value_counts().min())
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        raise ValueError('当前三分类任务某一类样本太少，无法做 CV。')
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def make_lgbm_binary_base(random_state=42, n_jobs=1):
    return lgb.LGBMClassifier(objective='binary', random_state=random_state, n_jobs=n_jobs, verbosity=-1)

def make_xgb_binary_base(random_state=42, n_jobs=1):
    return xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=random_state, n_jobs=n_jobs, verbosity=0)

def make_rf_binary_base(random_state=42, n_jobs=1):
    return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

def get_binary_param_distributions(model_name, y_train_bin):
    y_train_bin = np.asarray(y_train_bin).astype(int)
    n_pos = int(np.sum(y_train_bin == 1))
    n_neg = int(np.sum(y_train_bin == 0))
    if n_pos == 0:
        spw = 1.0
    else:
        spw = max(1.0, n_neg / max(n_pos, 1))
    spw_values = sorted(list(set([1.0, round(spw * 0.5, 3), round(spw, 3), round(spw * 1.5, 3), round(spw * 2.0, 3), round(spw * 3.0, 3)])))
    if model_name == 'LGBM':
        return {'n_estimators': [200, 300, 500, 800, 1000], 'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08], 'num_leaves': [7, 15, 31, 63, 127], 'max_depth': [-1, 3, 5, 7, 9], 'min_child_samples': [5, 10, 20, 40, 80], 'subsample': [0.65, 0.75, 0.85, 1.0], 'colsample_bytree': [0.65, 0.75, 0.85, 1.0], 'reg_alpha': [0.0, 0.01, 0.1, 0.5, 1.0], 'reg_lambda': [0.0, 0.01, 0.1, 0.5, 1.0, 2.0], 'class_weight': [None, 'balanced']}
    if model_name == 'XGB':
        return {'n_estimators': [200, 300, 500, 800, 1000], 'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08], 'max_depth': [2, 3, 4, 5, 7], 'min_child_weight': [1, 3, 5, 10], 'subsample': [0.65, 0.75, 0.85, 1.0], 'colsample_bytree': [0.65, 0.75, 0.85, 1.0], 'reg_alpha': [0.0, 0.01, 0.1, 0.5, 1.0], 'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0], 'scale_pos_weight': spw_values, 'max_delta_step': [0, 1, 2]}
    if model_name == 'RF':
        return {'n_estimators': [300, 500, 800, 1000], 'max_depth': [None, 5, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5, 10], 'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7], 'class_weight': ['balanced', 'balanced_subsample']}
    raise ValueError(f'未知模型: {model_name}')

def make_binary_estimator_by_name(model_name, random_state=42, n_jobs=1):
    if model_name == 'LGBM':
        return make_lgbm_binary_base(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'XGB':
        return make_xgb_binary_base(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'RF':
        return make_rf_binary_base(random_state=random_state, n_jobs=n_jobs)
    raise ValueError(f'未知模型: {model_name}')

def tune_one_binary_model(model_name, X_train, y_train_bin, scoring, host_name, random_state=42):
    estimator = make_binary_estimator_by_name(model_name, random_state=random_state, n_jobs=1)
    param_dist = get_binary_param_distributions(model_name=model_name, y_train_bin=y_train_bin)
    cv = get_binary_cv(y=y_train_bin, max_splits=BINARY_TUNING_CV, random_state=random_state)
    search = RandomizedSearchCV(estimator=estimator, param_distributions=param_dist, n_iter=N_ITER_BINARY_SEARCH, scoring=scoring, cv=cv, random_state=random_state, n_jobs=SEARCH_N_JOBS, verbose=1, refit=True, error_score=np.nan, return_train_score=False)
    search.fit(X_train, y_train_bin)
    best_model = search.best_estimator_
    best_model = set_model_random_state_and_jobs(best_model, random_state=random_state, n_jobs=-1)
    result_df = pd.DataFrame(search.cv_results_)
    result_path = os.path.join(SEARCH_DIR, f'{host_name}_{model_name}_RandomSearch_Results.csv')
    result_df.to_csv(result_path, index=False)
    best_record = {'Host': host_name, 'Model': model_name, 'Scoring': scoring, 'Best_CV_Score': float(search.best_score_), 'Best_Params': search.best_params_, 'Search_Result_Path': result_path}
    return (best_model, best_record)

def tune_binary_models_for_host(host_name, X_train_full, y_train_multi, features):
    X_train = X_train_full[features].copy()
    y_train_bin = y_train_multi.astype(str).str.lower().eq(host_name).astype(int).values
    scoring = TUNING_SCORING_BY_HOST.get(host_name, 'average_precision')
    tuned_templates = []
    records = []
    for model_name in ['LGBM', 'XGB', 'RF']:
        best_model, record = tune_one_binary_model(model_name=model_name, X_train=X_train, y_train_bin=y_train_bin, scoring=scoring, host_name=host_name, random_state=RANDOM_STATE)
        tuned_templates.append((model_name, best_model))
        records.append(record)
    return (tuned_templates, records)

def make_default_binary_templates(random_state=42):
    return [('LGBM', lgb.LGBMClassifier(objective='binary', n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=random_state, n_jobs=-1, verbosity=-1)), ('XGB', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1, verbosity=0)), ('RF', RandomForestClassifier(n_estimators=500, max_features='sqrt', class_weight='balanced_subsample', random_state=random_state, n_jobs=-1))]

def fit_three_tree_nelder_model(tree_templates, X_train_df, y_train, random_state=42, oversample_positive=False, target_pos_ratio=0.5):
    y_train = np.asarray(y_train).astype(int)
    if len(np.unique(y_train)) < 2:
        raise ValueError('当前训练数据只有一个类别，无法训练二分类模型。')
    if oversample_positive:
        X_fit_df, y_fit = random_oversample_binary_positive(X_train_df, y_train, target_pos_ratio=target_pos_ratio, random_state=random_state)
    else:
        X_fit_df = X_train_df
        y_fit = y_train
    fitted_estimators = []
    for name, template_model in tree_templates:
        m = set_model_random_state_and_jobs(template_model, random_state=random_state, n_jobs=-1)
        m.fit(X_fit_df, y_fit)
        fitted_estimators.append((name, m))
    ensemble = NelderMeadEnsemble(estimators=fitted_estimators, oversample_positive=oversample_positive, target_pos_ratio=target_pos_ratio, random_state=random_state)
    ensemble.fit(X_train_df, y_train)
    return ensemble

def predict_positive_probability(model, X_df):
    prob = model.predict_proba(X_df)
    if prob.shape[1] != 2:
        raise ValueError(f'predict_proba 输出不是两列: {prob.shape}')
    return prob[:, 1]

def make_lgbm_meta_classifier(random_state=42, n_jobs=1, params=None):
    base_params = dict(objective='multiclass', random_state=random_state, n_jobs=n_jobs, verbosity=-1, class_weight='balanced')
    if params is not None:
        base_params.update(params)
    return lgb.LGBMClassifier(**base_params)

def get_meta_param_distributions():
    return {'n_estimators': [50, 100, 200, 300, 500], 'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1], 'num_leaves': [3, 5, 7, 15, 31], 'max_depth': [2, 3, 4, 5, -1], 'min_child_samples': [3, 5, 10, 20, 40], 'subsample': [0.7, 0.85, 1.0], 'colsample_bytree': [0.7, 0.85, 1.0], 'reg_alpha': [0.0, 0.01, 0.1, 0.5, 1.0], 'reg_lambda': [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]}

def tune_lgbm_meta_classifier(X_meta, y_meta, random_state=42):
    y_series = pd.Series(y_meta, index=X_meta.index).astype(str).str.lower()
    cv = get_multiclass_cv(y=y_series, max_splits=META_TUNING_CV, random_state=random_state)
    macro_f1_scorer = make_scorer(f1_score, average='macro', labels=TARGET_HOSTS, zero_division=0)
    estimator = make_lgbm_meta_classifier(random_state=random_state, n_jobs=1)
    search = RandomizedSearchCV(estimator=estimator, param_distributions=get_meta_param_distributions(), n_iter=N_ITER_META_SEARCH, scoring=macro_f1_scorer, cv=cv, random_state=random_state, n_jobs=SEARCH_N_JOBS, verbose=1, refit=True, error_score=np.nan, return_train_score=False)
    search.fit(X_meta, y_series.values)
    result_df = pd.DataFrame(search.cv_results_)
    result_path = os.path.join(SEARCH_DIR, 'Meta_LGBM_RandomSearch_Results.csv')
    result_df.to_csv(result_path, index=False)
    best_params = search.best_params_
    best_record = pd.DataFrame([{'Meta_Model': 'LGBMClassifier', 'Scoring': 'macro_f1', 'Best_CV_Score': float(search.best_score_), 'Best_Params': best_params, 'Search_Result_Path': result_path}])
    best_record.to_csv(os.path.join(TABLE_DIR, 'Fig7_Meta_LGBM_Best_Params.csv'), index=False)
    return best_params

def align_proba_to_target_hosts(proba, model_classes):
    model_classes = [str(c).lower() for c in model_classes]
    aligned = np.zeros((proba.shape[0], len(TARGET_HOSTS)), dtype=float)
    for i, host in enumerate(TARGET_HOSTS):
        if host not in model_classes:
            raise ValueError(f'Meta 模型中缺少类别: {host}，当前类别: {model_classes}')
        src_idx = model_classes.index(host)
        aligned[:, i] = proba[:, src_idx]
    return aligned

def fit_meta_oof_predictions_lgbm(X_meta, y, best_meta_params, n_splits=5, random_state=42):
    y_series = pd.Series(y, index=X_meta.index).astype(str).str.lower()
    cv = get_multiclass_cv(y=y_series, max_splits=n_splits, random_state=random_state)
    meta_prob_oof = pd.DataFrame(index=X_meta.index, columns=get_meta_prob_cols(), dtype=float)
    fold_records = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_series.values), start=1):
        train_ids = X_meta.index[train_idx]
        val_ids = X_meta.index[val_idx]
        X_train = X_meta.loc[train_ids].copy()
        X_val = X_meta.loc[val_ids].copy()
        y_train = y_series.loc[train_ids].values
        y_val = y_series.loc[val_ids].values
        meta_model = make_lgbm_meta_classifier(random_state=random_state + fold_idx, n_jobs=-1, params=best_meta_params)
        if DO_META_CHICKEN_OVERSAMPLING:
            X_train_fit, y_train_fit = random_oversample_multiclass_chicken(X_train, y_train, chicken_label='chicken', random_state=random_state + fold_idx)
        else:
            X_train_fit = X_train
            y_train_fit = y_train
        meta_model.fit(X_train_fit, y_train_fit)
        proba_val = meta_model.predict_proba(X_val)
        aligned = align_proba_to_target_hosts(proba_val, meta_model.classes_)
        meta_prob_oof.loc[val_ids, get_meta_prob_cols()] = aligned
        pred_val = [TARGET_HOSTS[i] for i in np.argmax(aligned, axis=1)]
        fold_records.append({'Fold': fold_idx, 'N_Train': len(train_ids), 'N_Val': len(val_ids), 'Val_Accuracy': accuracy_score(y_val, pred_val), 'Val_Macro_F1': f1_score(y_val, pred_val, labels=TARGET_HOSTS, average='macro', zero_division=0), 'Val_MCC': matthews_corrcoef(y_val, pred_val), 'Train_Host_Counts': pd.Series(y_train).value_counts().to_dict(), 'Val_Host_Counts': pd.Series(y_val).value_counts().to_dict()})
    meta_prob_oof = meta_prob_oof.astype(float)
    if meta_prob_oof.isna().any().any():
        raise ValueError('Meta OOF 概率中存在 NA，请检查 meta OOF 流程。')
    return (meta_prob_oof, pd.DataFrame(fold_records))

def fit_final_lgbm_meta_classifier(X_meta_b_oof, y_b, best_meta_params, random_state=42):
    y_series = pd.Series(y_b, index=X_meta_b_oof.index).astype(str).str.lower()
    if DO_META_CHICKEN_OVERSAMPLING:
        X_fit, y_fit = random_oversample_multiclass_chicken(X_meta_b_oof, y_series.values, chicken_label='chicken', random_state=random_state)
    else:
        X_fit = X_meta_b_oof
        y_fit = y_series.values
    meta_model = make_lgbm_meta_classifier(random_state=random_state, n_jobs=-1, params=best_meta_params)
    meta_model.fit(X_fit, y_fit)
    return meta_model

def predict_meta_prob_df(meta_model, X_meta, index):
    proba = meta_model.predict_proba(X_meta)
    aligned = align_proba_to_target_hosts(proba, meta_model.classes_)
    meta_prob_df = pd.DataFrame(aligned, index=index, columns=get_meta_prob_cols())
    return meta_prob_df

def build_stacked_prediction_df(base_prob_df, meta_prob_df, true_host_series):
    out = base_prob_df.copy()
    out.insert(0, 'SampleID', out.index)
    out['True_Host'] = true_host_series.loc[out.index].astype(str).str.lower().values
    for col in get_meta_prob_cols():
        out[col] = meta_prob_df.loc[out.index, col].astype(float).values
    meta_values = out[get_meta_prob_cols()].values.astype(float)
    pred_idx = np.argmax(meta_values, axis=1)
    out['Predicted_Host'] = [TARGET_HOSTS[i] for i in pred_idx]
    out['Confidence'] = np.max(meta_values, axis=1)
    base_values = out[get_base_prob_cols()].values.astype(float)
    base_idx = np.argmax(base_values, axis=1)
    out['Base_Argmax_Host'] = [TARGET_HOSTS[i] for i in base_idx]
    out['Base_Argmax_Confidence'] = np.max(base_values, axis=1)
    out['Correct'] = out['True_Host'].astype(str).str.lower() == out['Predicted_Host'].astype(str).str.lower()
    return out

def compute_binary_six_metrics(y_true_binary, y_score, threshold=0.5):
    y_true_binary = np.asarray(y_true_binary).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred_binary = (y_score >= threshold).astype(int)
    row = {}
    if len(np.unique(y_true_binary)) >= 2:
        row['AUC'] = roc_auc_score(y_true_binary, y_score)
        row['PR'] = average_precision_score(y_true_binary, y_score)
    else:
        row['AUC'] = np.nan
        row['PR'] = np.nan
    row['Accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    row['Macro F1'] = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    row['MCC'] = matthews_corrcoef(y_true_binary, y_pred_binary)
    row['Recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    return row

def compute_multiclass_metrics(y_true, y_pred, dataset_name):
    y_true = pd.Series(y_true).astype(str).str.lower().values
    y_pred = pd.Series(y_pred).astype(str).str.lower().values
    metric_dict = {'Accuracy': accuracy_score(y_true, y_pred), 'Macro F1': f1_score(y_true, y_pred, labels=TARGET_HOSTS, average='macro', zero_division=0), 'Weighted F1': f1_score(y_true, y_pred, labels=TARGET_HOSTS, average='weighted', zero_division=0), 'Macro Precision': precision_score(y_true, y_pred, labels=TARGET_HOSTS, average='macro', zero_division=0), 'Macro Recall / Balanced Acc': recall_score(y_true, y_pred, labels=TARGET_HOSTS, average='macro', zero_division=0), 'MCC': matthews_corrcoef(y_true, y_pred)}
    rows = []
    for metric, value in metric_dict.items():
        rows.append({'Dataset': dataset_name, 'Metric': metric, 'Value': value})
    return pd.DataFrame(rows)

def compute_per_class_metrics(y_true, y_pred, dataset_name):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=TARGET_HOSTS, zero_division=0)
    rows = []
    for host, p, r, f, s in zip(TARGET_HOSTS, precision, recall, f1, support):
        rows.append({'Dataset': dataset_name, 'Host': host, 'Precision': p, 'Recall': r, 'F1': f, 'Support': s})
    return pd.DataFrame(rows)

def save_cm_table(y_true, y_pred, out_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_HOSTS, normalize='true' if normalize else None)
    cm_df = pd.DataFrame(cm, index=[f'True_{h}' for h in TARGET_HOSTS], columns=[f'Pred_{h}' for h in TARGET_HOSTS])
    cm_df.to_csv(out_path)
    return cm_df
pa_data = pd.read_csv(PA_MATRIX_PATH, index_col=0)
X_all = pa_data.T
X_all.index = X_all.index.to_series().apply(clean_sample_id)
meta_df = pd.read_csv(META_PATH)
id_col = 'ID' if 'ID' in meta_df.columns else meta_df.columns[0]
meta_df[id_col] = meta_df[id_col].astype(str).apply(clean_sample_id)
meta_df = meta_df.set_index(id_col)
common_ids = X_all.index.intersection(meta_df.index)
X_all = X_all.loc[common_ids].copy()
meta_df = meta_df.loc[common_ids].copy()
host_col = 'Host' if 'Host' in meta_df.columns else meta_df.columns[1]
meta_df['Host_Clean'] = meta_df[host_col].apply(clean_host_name)
valid_host_mask = meta_df['Host_Clean'].isin(TARGET_HOSTS)
X_all = X_all.loc[valid_host_mask].copy()
meta_df = meta_df.loc[valid_host_mask].copy()
meta_df['TIME_PERIOD'] = meta_df['DATE'].astype(str).str.strip()
meta_df['IP_CLEAN'] = meta_df['IP'].astype(str).str.strip()
counts = meta_df.groupby(['TIME_PERIOD', 'IP_CLEAN']).size()
valid_schemes = counts[counts > 200].index.tolist()
if SCHEME_IDX >= len(valid_schemes):
    raise ValueError(f'SCHEME_IDX={SCHEME_IDX} 超出 valid_schemes 范围。')
final_scheme = valid_schemes[SCHEME_IDX]
time_p, ip_p = final_scheme
is_a = (meta_df['TIME_PERIOD'] == time_p) & (meta_df['IP_CLEAN'] == ip_p)
X_A_full = X_all.loc[is_a].copy()
X_B_full = X_all.loc[~is_a].copy()
y_A = meta_df.loc[X_A_full.index, 'Host_Clean'].copy()
y_B = meta_df.loc[X_B_full.index, 'Host_Clean'].copy()
features_by_host = {}
feature_source_records = []
for target_host in TARGET_HOSTS:
    features, feature_source = load_top_features_for_host(host_name=target_host, X_columns=X_all.columns)
    features_by_host[target_host] = features
    feature_source_records.append({'Host': target_host, 'N_Features': len(features), 'Feature_Source': feature_source})
    pd.DataFrame({'Rank': np.arange(1, len(features) + 1), 'Feature': features, 'Clean_Name': [clean_gene_name(f) for f in features]}).to_csv(os.path.join(TABLE_DIR, f'Fig7_{target_host}_Used_Top{len(features)}_Features.csv'), index=False)
pd.DataFrame(feature_source_records).to_csv(os.path.join(TABLE_DIR, 'Fig7_Top100_Feature_Sources.csv'), index=False)
tree_templates_by_host = {}
search_best_records = []
if DO_BINARY_HYPERPARAM_SEARCH:
    for target_host in TARGET_HOSTS:
        features = features_by_host[target_host]
        tuned_templates, records = tune_binary_models_for_host(host_name=target_host, X_train_full=X_B_full, y_train_multi=y_B, features=features)
        tree_templates_by_host[target_host] = tuned_templates
        search_best_records.extend(records)
else:
    for target_host in TARGET_HOSTS:
        tree_templates_by_host[target_host] = make_default_binary_templates(random_state=RANDOM_STATE)
pd.DataFrame(search_best_records).to_csv(os.path.join(TABLE_DIR, 'Fig7_Binary_Base_Best_Hyperparams.csv'), index=False)
for target_host in TARGET_HOSTS:
    joblib.dump({'host': target_host, 'features': features_by_host[target_host], 'tree_templates': tree_templates_by_host[target_host], 'note': 'Tuned LGBM/XGB/RF templates for one-vs-rest binary base classifier'}, os.path.join(MODEL_DIR, f'Fig7_TunedBaseTemplates_{target_host}.joblib'), compress=3)
class_counts_b = y_B.value_counts()
min_class_b = int(class_counts_b.min())
actual_outer_splits = min(N_SPLITS_OUTER, min_class_b)
if actual_outer_splits < 2:
    raise ValueError('B 集每类样本太少，无法进行 OOF。')
skf = StratifiedKFold(n_splits=actual_outer_splits, shuffle=True, random_state=RANDOM_STATE)
B_prob_oof = pd.DataFrame(index=X_B_full.index, columns=get_base_prob_cols(), dtype=float)
cv_weight_records = []
cv_fold_records = []
y_B_array = y_B.values
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_B_full, y_B_array), start=1):
    train_ids = X_B_full.index[train_idx]
    val_ids = X_B_full.index[val_idx]
    X_train_full = X_B_full.loc[train_ids]
    X_val_full = X_B_full.loc[val_ids]
    y_train_multi = y_B.loc[train_ids]
    y_val_multi = y_B.loc[val_ids]
    cv_fold_records.append({'Fold': fold_idx, 'N_Train': len(train_ids), 'N_Val': len(val_ids), 'Train_Host_Counts': y_train_multi.value_counts().to_dict(), 'Val_Host_Counts': y_val_multi.value_counts().to_dict()})
    for target_host in TARGET_HOSTS:
        features = features_by_host[target_host]
        X_train = X_train_full[features].copy()
        X_val = X_val_full[features].copy()
        y_train_bin = y_train_multi.astype(str).str.lower().eq(target_host).astype(int).values
        model = fit_three_tree_nelder_model(tree_templates=tree_templates_by_host[target_host], X_train_df=X_train, y_train=y_train_bin, random_state=RANDOM_STATE + fold_idx, oversample_positive=DO_CHICKEN_OVERSAMPLING and target_host == 'chicken', target_pos_ratio=CHICKEN_BINARY_POS_RATIO)
        prob_val = predict_positive_probability(model, X_val)
        B_prob_oof.loc[val_ids, f'P_{target_host}'] = prob_val
        model_names = [name for name, _ in model.estimators]
        for name, weight in zip(model_names, model.weights_):
            cv_weight_records.append({'Dataset': 'B-set base OOF', 'Fold': fold_idx, 'Host_Model': target_host, 'Base_Model': name, 'Weight': float(weight)})
B_prob_oof = B_prob_oof.astype(float)
if B_prob_oof.isna().any().any():
    raise ValueError('B_prob_oof 中存在 NA，请检查 base OOF 预测流程。')
B_base_prob_path = os.path.join(TABLE_DIR, 'Fig7_Bset_5foldOOF_TunedBase_OneVsRest_Probabilities.csv')
B_prob_oof_out = B_prob_oof.copy()
B_prob_oof_out.insert(0, 'SampleID', B_prob_oof_out.index)
B_prob_oof_out['True_Host'] = y_B.loc[B_prob_oof.index].values
B_prob_oof_out.to_csv(B_base_prob_path, index=False)
pd.DataFrame(cv_weight_records).to_csv(os.path.join(TABLE_DIR, 'Fig7_Bset_5foldCV_TunedBase_NelderMead_Weights.csv'), index=False)
pd.DataFrame(cv_fold_records).to_csv(os.path.join(TABLE_DIR, 'Fig7_Bset_5foldCV_TunedBase_Fold_Summary.csv'), index=False)
X_meta_B_oof = B_prob_oof[get_base_prob_cols()].copy()
if DO_META_HYPERPARAM_SEARCH:
    best_meta_params = tune_lgbm_meta_classifier(X_meta=X_meta_B_oof, y_meta=y_B, random_state=RANDOM_STATE)
else:
    best_meta_params = {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 10, 'subsample': 0.85, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.1}
B_meta_prob_oof, meta_oof_fold_df = fit_meta_oof_predictions_lgbm(X_meta=X_meta_B_oof, y=y_B, best_meta_params=best_meta_params, n_splits=N_SPLITS_META, random_state=RANDOM_STATE)
meta_oof_fold_df.to_csv(os.path.join(TABLE_DIR, 'Fig7_Bset_LGBMMeta_OOF_Fold_Summary.csv'), index=False)
B_pred_df = build_stacked_prediction_df(base_prob_df=B_prob_oof, meta_prob_df=B_meta_prob_oof, true_host_series=y_B)
B_pred_path = os.path.join(TABLE_DIR, 'Fig7_Bset_5foldOOF_TunedBase_LGBMMeta_Integrated_ThreeClass_Predictions.csv')
B_pred_df.to_csv(B_pred_path, index=False)
A_prob_df = pd.DataFrame(index=X_A_full.index, columns=get_base_prob_cols(), dtype=float)
final_weight_records = []
final_models_by_host = {}
for target_host in TARGET_HOSTS:
    features = features_by_host[target_host]
    X_train = X_B_full[features].copy()
    X_test = X_A_full[features].copy()
    y_train_bin = y_B.astype(str).str.lower().eq(target_host).astype(int).values
    final_model = fit_three_tree_nelder_model(tree_templates=tree_templates_by_host[target_host], X_train_df=X_train, y_train=y_train_bin, random_state=RANDOM_STATE, oversample_positive=DO_CHICKEN_OVERSAMPLING and target_host == 'chicken', target_pos_ratio=CHICKEN_BINARY_POS_RATIO)
    final_models_by_host[target_host] = final_model
    prob_a = predict_positive_probability(final_model, X_test)
    A_prob_df.loc[X_A_full.index, f'P_{target_host}'] = prob_a
    model_names = [name for name, _ in final_model.estimators]
    for name, weight in zip(model_names, final_model.weights_):
        final_weight_records.append({'Dataset': 'Full B training -> independent A-set', 'Host_Model': target_host, 'Base_Model': name, 'Weight': float(weight)})
    model_save_path = os.path.join(MODEL_DIR, f'Fig7_Final_TunedBase_{target_host}_NelderMead_Top{len(features)}.joblib')
    joblib.dump({'target_host': target_host, 'model': final_model, 'selected_features': features, 'tree_models': model_names, 'weights': final_model.weights_, 'training_set': 'Full B set', 'test_set': 'Independent A set', 'base_output': f'P_{target_host}'}, model_save_path, compress=3)
A_prob_df = A_prob_df.astype(float)
A_base_prob_path = os.path.join(TABLE_DIR, 'Fig7_Aset_Independent_TunedBase_OneVsRest_Probabilities.csv')
A_prob_df_out = A_prob_df.copy()
A_prob_df_out.insert(0, 'SampleID', A_prob_df_out.index)
A_prob_df_out['True_Host'] = y_A.loc[A_prob_df.index].values
A_prob_df_out.to_csv(A_base_prob_path, index=False)
pd.DataFrame(final_weight_records).to_csv(os.path.join(TABLE_DIR, 'Fig7_Final_FullB_TunedBase_NelderMead_Weights.csv'), index=False)
final_meta_model = fit_final_lgbm_meta_classifier(X_meta_b_oof=X_meta_B_oof, y_b=y_B, best_meta_params=best_meta_params, random_state=RANDOM_STATE)
meta_model_path = os.path.join(MODEL_DIR, 'Fig7_Final_LGBMMeta_Integrator.joblib')
joblib.dump({'model': final_meta_model, 'meta_features': get_base_prob_cols(), 'meta_probability_columns': get_meta_prob_cols(), 'classes_order': TARGET_HOSTS, 'best_meta_params': best_meta_params, 'training_input': 'B-set OOF probabilities from tuned one-vs-rest base classifiers', 'integration_rule': 'LGBMClassifier meta-classifier on [P_human, P_pig, P_chicken]', 'class_weight': 'balanced'}, meta_model_path, compress=3)
A_meta_prob_df = predict_meta_prob_df(meta_model=final_meta_model, X_meta=A_prob_df[get_base_prob_cols()].copy(), index=A_prob_df.index)
A_pred_df = build_stacked_prediction_df(base_prob_df=A_prob_df, meta_prob_df=A_meta_prob_df, true_host_series=y_A)
A_pred_path = os.path.join(TABLE_DIR, 'Fig7_Aset_Independent_TunedBase_LGBMMeta_Integrated_ThreeClass_Predictions.csv')
A_pred_df.to_csv(A_pred_path, index=False)
B_multi_metrics = compute_multiclass_metrics(y_true=B_pred_df['True_Host'], y_pred=B_pred_df['Predicted_Host'], dataset_name='B-set 5-fold CV')
A_multi_metrics = compute_multiclass_metrics(y_true=A_pred_df['True_Host'], y_pred=A_pred_df['Predicted_Host'], dataset_name='Independent A-set')
multi_metrics_df = pd.concat([B_multi_metrics, A_multi_metrics], axis=0, ignore_index=True)
multi_metrics_path = os.path.join(TABLE_DIR, 'Fig7_TunedBase_LGBMMeta_Integrated_ThreeClass_Metrics_BsetCV_vs_Aset.csv')
multi_metrics_df.to_csv(multi_metrics_path, index=False)
B_per_class_df = compute_per_class_metrics(y_true=B_pred_df['True_Host'], y_pred=B_pred_df['Predicted_Host'], dataset_name='B-set 5-fold CV')
A_per_class_df = compute_per_class_metrics(y_true=A_pred_df['True_Host'], y_pred=A_pred_df['Predicted_Host'], dataset_name='Independent A-set')
per_class_df = pd.concat([B_per_class_df, A_per_class_df], axis=0, ignore_index=True)
per_class_path = os.path.join(TABLE_DIR, 'Fig7_TunedBase_LGBMMeta_PerClass_Metrics_BsetCV_vs_Aset.csv')
per_class_df.to_csv(per_class_path, index=False)
binary_metric_records = []
for dataset_name, pred_df in [('B-set 5-fold CV', B_pred_df), ('Independent A-set', A_pred_df)]:
    true_host = pred_df['True_Host'].astype(str).str.lower()
    for target_host in TARGET_HOSTS:
        y_true_bin = true_host.eq(target_host).astype(int).values
        y_score = pred_df[f'P_{target_host}'].astype(float).values
        threshold = float(FIG7A_BINARY_THRESHOLDS.get(target_host, 0.5))
        metrics = compute_binary_six_metrics(y_true_binary=y_true_bin, y_score=y_score, threshold=threshold)
        for metric, value in metrics.items():
            binary_metric_records.append({'Dataset': dataset_name, 'Host_Model': f'{target_host} vs rest', 'Target_Host': target_host, 'Metric': metric, 'Value': value, 'Threshold': threshold})
binary_metrics_df = pd.DataFrame(binary_metric_records)
binary_metrics_path = os.path.join(TABLE_DIR, 'Fig7_TunedBase_OneVsRest_Binary_SixMetrics_BsetCV_vs_Aset.csv')
binary_metrics_df.to_csv(binary_metrics_path, index=False)
save_cm_table(B_pred_df['True_Host'], B_pred_df['Predicted_Host'], os.path.join(TABLE_DIR, 'Fig7_Bset_TunedBase_LGBMMeta_ConfusionMatrix_Count.csv'), normalize=False)
save_cm_table(A_pred_df['True_Host'], A_pred_df['Predicted_Host'], os.path.join(TABLE_DIR, 'Fig7_Aset_TunedBase_LGBMMeta_ConfusionMatrix_Count.csv'), normalize=False)
save_cm_table(B_pred_df['True_Host'], B_pred_df['Predicted_Host'], os.path.join(TABLE_DIR, 'Fig7_Bset_TunedBase_LGBMMeta_ConfusionMatrix_Normalized.csv'), normalize=True)
save_cm_table(A_pred_df['True_Host'], A_pred_df['Predicted_Host'], os.path.join(TABLE_DIR, 'Fig7_Aset_TunedBase_LGBMMeta_ConfusionMatrix_Normalized.csv'), normalize=True)
binary_a_df = binary_metrics_df[binary_metrics_df['Dataset'] == 'Independent A-set'].copy()
binary_a_plotdata_path = os.path.join(PLOT_DATA_DIR, 'Fig7A_Aset_TunedBase_OneVsRest_SixMetrics_PlotData.csv')
binary_a_df.to_csv(binary_a_plotdata_path, index=False)
fig7b_plotdata_path = os.path.join(PLOT_DATA_DIR, 'Fig7B_TunedBase_LGBMMeta_ThreeClass_Metrics_PlotData.csv')
multi_metrics_df.to_csv(fig7b_plotdata_path, index=False)
fig7cd_plotdata_path = os.path.join(PLOT_DATA_DIR, 'Fig7C_Fig7D_Aset_TunedBase_LGBMMeta_Prediction_Data.csv')
A_pred_df.to_csv(fig7cd_plotdata_path, index=False)
summary_records = []
summary_records.append({'Item': 'Scheme', 'Value': f'Scheme_{SCHEME_IDX}: {time_p} | {ip_p}'})
summary_records.append({'Item': 'A_set_size', 'Value': len(X_A_full)})
summary_records.append({'Item': 'B_set_size', 'Value': len(X_B_full)})
summary_records.append({'Item': 'Base_model_framework', 'Value': 'Three tuned one-vs-rest binary classifiers: human vs rest, pig vs rest, chicken vs rest'})
summary_records.append({'Item': 'Base_models_in_each_NelderMead', 'Value': 'Tuned LGBM + tuned XGB + tuned RF; Nelder-Mead optimized ensemble weights'})
summary_records.append({'Item': 'Feature_set', 'Value': f'Host-specific Top{TOP_N_GENES} genes from B-set SHAP/Fig5'})
summary_records.append({'Item': 'Binary_hyperparameter_search', 'Value': f'RandomizedSearchCV on B set; n_iter={N_ITER_BINARY_SEARCH}; scoring={TUNING_SCORING_BY_HOST}'})
summary_records.append({'Item': 'Stacked_meta_integrator', 'Value': 'LightGBM multiclass meta-classifier trained on B-set OOF probabilities [P_human, P_pig, P_chicken]'})
summary_records.append({'Item': 'Meta_hyperparameter_search', 'Value': f'RandomizedSearchCV on B-set OOF probabilities; n_iter={N_ITER_META_SEARCH}; scoring=macro F1'})
summary_records.append({'Item': 'B_set_validation', 'Value': f'Base one-vs-rest {actual_outer_splits}-fold OOF + LightGBM meta OOF prediction'})
summary_records.append({'Item': 'A_set_validation', 'Value': 'Final tuned base models trained on full B set; final LightGBM meta trained on B OOF probabilities; evaluated once on independent A set'})
summary_records.append({'Item': 'Chicken_random_oversampling', 'Value': f'Binary chicken vs rest oversampling={DO_CHICKEN_OVERSAMPLING}, target_pos_ratio={CHICKEN_BINARY_POS_RATIO}; Meta chicken oversampling={DO_META_CHICKEN_OVERSAMPLING}; training folds only; A set untouched; no SMOTE'})
summary_records.append({'Item': 'Fig7A_binary_thresholds', 'Value': str(FIG7A_BINARY_THRESHOLDS)})
summary_records.append({'Item': 'Final_three_class_rule', 'Value': 'Predicted host = argmax(Meta_P_human, Meta_P_pig, Meta_P_chicken)'})
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(TABLE_DIR, 'Fig7_TunedBase_LGBMMeta_Run_Summary.csv'), index=False)
