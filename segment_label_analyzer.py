import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ==========================================================
# MANUAL LABEL SETTINGS
# ==========================================================
INTERESTING_TIMES = [
    16.0, 22.0, 25.0, 24.0, 26.0, 27.0, 33.0, 37.0, 40.0, 46.0,
    51.0, 52.0, 68.0, 71.0, 74.0, 84.0, 88.0, 92.0, 93.0, 94.0,
    99.0, 101.0, 105.0, 107.0, 113.0, 114.0, 116.0, 123.0, 124.0, 125.0,
    127.0, 135.0, 143.0, 147.0, 154.0, 157.0, 160.0, 173.0, 178.0, 180.0,
    189.0, 195.0, 197.0, 201.0, 203.0, 208.0, 215.0, 221.0, 224.0, 226.0,
    229.0, 239.0, 243.0, 246.0, 251.0, 257.0, 258.0, 260.0, 263.0, 264.0,
    265.0, 271.0, 274.0, 279.0, 283.0, 286.0, 293.0, 302.0, 307.0, 309.0,
    313.0, 318.0, 325.0, 327.0, 331.0, 335.0, 343.0, 351.0, 357.0, 361.0,
    369.0, 375.0, 378.0, 382.0, 384.0, 386.0, 388.0, 390.0, 413.0, 414.0,
    416.0, 422.0, 424.0, 428.0, 436.0, 438.0, 449.0, 452.0, 455.0, 456.0,
    458.0, 460.0, 464.0, 467.0, 472.0, 479.0, 486.0, 488.0, 489.0, 497.0,
    505.0, 520.0, 521.0, 525.0, 530.0, 532.0, 534.0, 536.0, 543.0, 547.0,
    549.0, 552.0, 560.0, 563.0, 564.0, 573.0, 584.0, 586.0, 589.0, 591.0,
    596.0, 601.0, 609.0, 614.0, 615.0, 617.0, 631.0, 636.0, 643.0, 645.0,
    649.0, 650.0,
]

INTERESTING_TIMES_FILE = None
TOLERANCE_SEC = 0.0
CSV_PATH = "video_features_out/video_features.csv"
OUT_DIR = "segment_label_analysis_out"
SAVE_PLOTS_PREFIX = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "mean_motion_intensity",
    "std_motion_intensity",
    "max_motion_intensity",
    "motion_peak_count",
    "motion_peak_rate_hz",
    "motion_periodicity_score",
    "dominant_motion_freq_hz",
    "mean_flow_magnitude",
    "flow_rhythm_score",
    "dominant_flow_freq_hz",
    "cut_count",
    "cut_frequency_hz",
    "mean_camera_dx",
    "mean_camera_dy",
    "mean_camera_motion",
    "camera_motion_stability",
    "pose_detected_ratio",
    "pose_motion_mean",
    "pose_motion_std",
    "pose_trajectory_length_mean",
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "contrast_mean",
    "colorfulness_mean",
]


def parse_times(times_text: str) -> List[float]:
    if not times_text:
        return []
    out = []
    for x in times_text.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return sorted(out)


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"start_sec", "end_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def label_segments_by_times(
    df: pd.DataFrame,
    interesting_times: List[float],
    tolerance_sec: float = 0.0,
) -> pd.DataFrame:
    df = df.copy()
    df["interesting_label"] = 0

    if not interesting_times:
        return df

    labels = []
    for _, row in df.iterrows():
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        is_interesting = 0
        for t in interesting_times:
            if (start - tolerance_sec) <= t <= (end + tolerance_sec):
                is_interesting = 1
                break
        labels.append(is_interesting)

    df["interesting_label"] = labels
    return df


def summarize_labels(df: pd.DataFrame) -> None:
    total = len(df)
    positives = int(df["interesting_label"].sum())
    negatives = total - positives
    print("=" * 80)
    print(f"Total segments:       {total}")
    print(f"Interesting segments: {positives}")
    print(f"Other segments:       {negatives}")
    print(f"Positive ratio:       {positives / max(total, 1):.3f}")
    print("=" * 80)


def point_biserial_like_scores(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    y = df["interesting_label"].astype(float).values

    for col in feature_cols:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue

        x_m = x[mask]
        y_m = y[mask]
        corr = np.corrcoef(x_m, y_m)[0, 1] if np.std(x_m) > 1e-12 and np.std(y_m) > 1e-12 else 0.0
        pos_mean = float(np.mean(x_m[y_m == 1])) if np.any(y_m == 1) else np.nan
        neg_mean = float(np.mean(x_m[y_m == 0])) if np.any(y_m == 0) else np.nan
        diff = pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan

        rows.append(
            {
                "feature": col,
                "corr_with_label": float(corr),
                "abs_corr": float(abs(corr)),
                "interesting_mean": pos_mean,
                "other_mean": neg_mean,
                "mean_difference": diff,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return out


def _threshold_predictions(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_prob >= threshold).astype(int)


def _safe_predict_proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("Model does not support predict_proba.")


def _permutation_importance_df(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
            "abs_permutation_importance_mean": np.abs(result.importances_mean),
        }
    ).sort_values("abs_permutation_importance_mean", ascending=False).reset_index(drop=True)

    return out


def evaluate_models_tuned(
    df: pd.DataFrame,
    feature_cols: List[str],
    scoring: str = "roc_auc",
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], dict, Dict[str, Dict[str, np.ndarray]]]:
    usable_cols = [c for c in feature_cols if c in df.columns]
    if not usable_cols:
        raise ValueError("None of the requested feature columns were found in the CSV.")

    X = df[usable_cols].apply(pd.to_numeric, errors="coerce")
    y = df["interesting_label"].astype(int)

    if y.nunique() < 2:
        raise ValueError("You need both interesting and non-interesting segments for statistical evaluation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    logreg_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    logreg_param_dist = {
        "clf__C": np.logspace(-3, 2, 20),
        "clf__solver": ["lbfgs", "liblinear"],
    }

    logreg_search = RandomizedSearchCV(
        estimator=logreg_pipe,
        param_distributions=logreg_param_dist,
        n_iter=20,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    logreg_search.fit(X_train, y_train)

    best_logreg = logreg_search.best_estimator_
    y_prob_lr = _safe_predict_proba_positive(best_logreg, X_test)
    y_pred_lr = _threshold_predictions(y_prob_lr, threshold=threshold)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

    lr_coefs = best_logreg.named_steps["clf"].coef_[0]
    lr_coef_df = pd.DataFrame(
        {
            "feature": usable_cols,
            "logreg_coef": lr_coefs,
            "abs_logreg_coef": np.abs(lr_coefs),
        }
    ).sort_values("abs_logreg_coef", ascending=False).reset_index(drop=True)

    # Random Forest
    rf_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )

    rf_param_dist = {
        "clf__n_estimators": [200, 300, 500, 800],
        "clf__max_depth": [None, 4, 6, 8, 12, 16, 24],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", 0.5, 0.8],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=rf_param_dist,
        n_iter=30,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_
    y_prob_rf = _safe_predict_proba_positive(best_rf, X_test)
    y_pred_rf = _threshold_predictions(y_prob_rf, threshold=threshold)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    rf_importance = best_rf.named_steps["clf"].feature_importances_
    rf_df = pd.DataFrame(
        {
            "feature": usable_cols,
            "rf_importance": rf_importance,
        }
    ).sort_values("rf_importance", ascending=False).reset_index(drop=True)

    # HistGradientBoosting
    hgb_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    hgb_param_dist = {
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__max_iter": [100, 200, 300, 500],
        "clf__max_depth": [None, 3, 5, 8, 12],
        "clf__min_samples_leaf": [10, 20, 30, 50],
        "clf__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "clf__max_bins": [64, 128, 255],
    }

    hgb_search = RandomizedSearchCV(
        estimator=hgb_pipe,
        param_distributions=hgb_param_dist,
        n_iter=30,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    hgb_search.fit(X_train, y_train)

    best_hgb = hgb_search.best_estimator_
    y_prob_hgb = _safe_predict_proba_positive(best_hgb, X_test)
    y_pred_hgb = _threshold_predictions(y_prob_hgb, threshold=threshold)
    auc_hgb = roc_auc_score(y_test, y_prob_hgb)
    fpr_hgb, tpr_hgb, _ = roc_curve(y_test, y_prob_hgb)

    hgb_perm_df = _permutation_importance_df(
        best_hgb,
        X_test,
        y_test,
        usable_cols,
        scoring=scoring,
        n_repeats=10,
        random_state=42,
    )

    comparison_df = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "cv_best_score": float(logreg_search.best_score_),
                "test_roc_auc": float(auc_lr),
                "best_params": json.dumps(logreg_search.best_params_, ensure_ascii=False),
            },
            {
                "model": "random_forest",
                "cv_best_score": float(rf_search.best_score_),
                "test_roc_auc": float(auc_rf),
                "best_params": json.dumps(rf_search.best_params_, ensure_ascii=False),
            },
            {
                "model": "hist_gradient_boosting",
                "cv_best_score": float(hgb_search.best_score_),
                "test_roc_auc": float(auc_hgb),
                "best_params": json.dumps(hgb_search.best_params_, ensure_ascii=False),
            },
        ]
    ).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)

    metrics = {
        "scoring_used_for_search": scoring,
        "classification_threshold": float(threshold),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "models": {
            "logistic_regression": {
                "best_params": logreg_search.best_params_,
                "cv_best_score": float(logreg_search.best_score_),
                "test_roc_auc": float(auc_lr),
                "classification_report": classification_report(y_test, y_pred_lr, zero_division=0),
            },
            "random_forest": {
                "best_params": rf_search.best_params_,
                "cv_best_score": float(rf_search.best_score_),
                "test_roc_auc": float(auc_rf),
                "classification_report": classification_report(y_test, y_pred_rf, zero_division=0),
            },
            "hist_gradient_boosting": {
                "best_params": hgb_search.best_params_,
                "cv_best_score": float(hgb_search.best_score_),
                "test_roc_auc": float(auc_hgb),
                "classification_report": classification_report(y_test, y_pred_hgb, zero_division=0),
            },
        },
    }

    importance_outputs = {
        "logistic_regression": lr_coef_df,
        "random_forest": rf_df,
        "hist_gradient_boosting": hgb_perm_df,
    }

    roc_curve_data = {
        "logistic_regression": {"fpr": fpr_lr, "tpr": tpr_lr, "auc": float(auc_lr)},
        "random_forest": {"fpr": fpr_rf, "tpr": tpr_rf, "auc": float(auc_rf)},
        "hist_gradient_boosting": {"fpr": fpr_hgb, "tpr": tpr_hgb, "auc": float(auc_hgb)},
    }

    return comparison_df, importance_outputs, metrics, roc_curve_data


def compute_feature_significance(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows = []

    interesting = df[df["interesting_label"] == 1]
    other = df[df["interesting_label"] == 0]

    for col in feature_cols:
        if col not in df.columns:
            continue

        x1 = pd.to_numeric(interesting[col], errors="coerce").dropna().values
        x0 = pd.to_numeric(other[col], errors="coerce").dropna().values

        if len(x1) < 3 or len(x0) < 3:
            continue

        mean1 = float(np.mean(x1))
        mean0 = float(np.mean(x0))
        std1 = float(np.std(x1, ddof=1))
        std0 = float(np.std(x0, ddof=1))

        t_stat, p_value = ttest_ind(x1, x0, equal_var=False, nan_policy="omit")

        pooled_sd = np.sqrt(((len(x1) - 1) * std1**2 + (len(x0) - 1) * std0**2) / max(len(x1) + len(x0) - 2, 1))
        cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd > 1e-12 else 0.0

        rows.append(
            {
                "feature": col,
                "interesting_mean": mean1,
                "interesting_std": std1,
                "other_mean": mean0,
                "other_std": std0,
                "mean_difference": mean1 - mean0,
                "welch_t_p_value": float(p_value),
                "cohen_d": float(cohen_d),
                "abs_cohen_d": float(abs(cohen_d)),
                "n_interesting": int(len(x1)),
                "n_other": int(len(x0)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["welch_t_p_value", "abs_cohen_d"], ascending=[True, False]).reset_index(drop=True)
    return out


def rank_features_consensus(
    corr_df: pd.DataFrame,
    importance_outputs: Dict[str, pd.DataFrame],
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    all_features = set()

    if not corr_df.empty:
        all_features.update(corr_df["feature"].tolist())
    if not importance_outputs["logistic_regression"].empty:
        all_features.update(importance_outputs["logistic_regression"]["feature"].tolist())
    if not importance_outputs["random_forest"].empty:
        all_features.update(importance_outputs["random_forest"]["feature"].tolist())
    if not importance_outputs["hist_gradient_boosting"].empty:
        all_features.update(importance_outputs["hist_gradient_boosting"]["feature"].tolist())
    if not stats_df.empty:
        all_features.update(stats_df["feature"].tolist())

    rows = []
    for feat in sorted(all_features):
        row = {"feature": feat}

        corr_match = corr_df[corr_df["feature"] == feat]
        row["abs_corr"] = float(corr_match["abs_corr"].iloc[0]) if not corr_match.empty else 0.0

        lr_match = importance_outputs["logistic_regression"][importance_outputs["logistic_regression"]["feature"] == feat]
        row["abs_logreg_coef"] = float(lr_match["abs_logreg_coef"].iloc[0]) if not lr_match.empty else 0.0

        rf_match = importance_outputs["random_forest"][importance_outputs["random_forest"]["feature"] == feat]
        row["rf_importance"] = float(rf_match["rf_importance"].iloc[0]) if not rf_match.empty else 0.0

        hgb_match = importance_outputs["hist_gradient_boosting"][
            importance_outputs["hist_gradient_boosting"]["feature"] == feat
        ]
        row["abs_hgb_perm"] = (
            float(hgb_match["abs_permutation_importance_mean"].iloc[0]) if not hgb_match.empty else 0.0
        )

        stats_match = stats_df[stats_df["feature"] == feat]
        row["abs_cohen_d"] = float(stats_match["abs_cohen_d"].iloc[0]) if not stats_match.empty else 0.0
        row["p_value"] = float(stats_match["welch_t_p_value"].iloc[0]) if not stats_match.empty else 1.0

        rows.append(row)

    rank_df = pd.DataFrame(rows)

    if rank_df.empty:
        return rank_df

    rank_df["rank_corr"] = rank_df["abs_corr"].rank(ascending=False, method="average")
    rank_df["rank_logreg"] = rank_df["abs_logreg_coef"].rank(ascending=False, method="average")
    rank_df["rank_rf"] = rank_df["rf_importance"].rank(ascending=False, method="average")
    rank_df["rank_hgb"] = rank_df["abs_hgb_perm"].rank(ascending=False, method="average")
    rank_df["rank_effect"] = rank_df["abs_cohen_d"].rank(ascending=False, method="average")
    rank_df["rank_p"] = rank_df["p_value"].rank(ascending=True, method="average")

    rank_df["consensus_rank_score"] = (
        rank_df["rank_corr"]
        + rank_df["rank_logreg"]
        + rank_df["rank_rf"]
        + rank_df["rank_hgb"]
        + rank_df["rank_effect"]
        + rank_df["rank_p"]
    )

    rank_df = rank_df.sort_values("consensus_rank_score", ascending=True).reset_index(drop=True)
    return rank_df


def plot_roc_curves(
    roc_curve_data: Dict[str, Dict[str, np.ndarray]],
    out_path: str = None,
) -> None:
    plt.figure(figsize=(10, 7))

    for model_name, data in roc_curve_data.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc_value = data["auc"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves for tuned models")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curve plot to: {out_path}")
    plt.show()


def plot_top3_feature_stats(
    stats_df: pd.DataFrame,
    top_features: List[str],
    out_path: str = None,
) -> None:
    stats_plot = stats_df[stats_df["feature"].isin(top_features)].copy()
    stats_plot["feature"] = pd.Categorical(stats_plot["feature"], categories=top_features, ordered=True)
    stats_plot = stats_plot.sort_values("feature")

    x = np.arange(len(stats_plot))
    width = 0.35

    plt.figure(figsize=(12, 7))
    plt.bar(
        x - width / 2,
        stats_plot["other_mean"].values,
        width=width,
        yerr=stats_plot["other_std"].values,
        capsize=5,
        label="not interesting",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        stats_plot["interesting_mean"].values,
        width=width,
        yerr=stats_plot["interesting_std"].values,
        capsize=5,
        label="interesting",
        alpha=0.8,
    )

    for i, row in enumerate(stats_plot.itertuples(index=False)):
        p_txt = f"p={row.welch_t_p_value:.2e}"
        d_txt = f"d={row.cohen_d:.2f}"
        top_y = max(
            row.interesting_mean + row.interesting_std,
            row.other_mean + row.other_std,
        )
        plt.text(i, top_y * 1.05 if top_y != 0 else 0.05, f"{p_txt}\n{d_txt}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, stats_plot["feature"].tolist(), rotation=20, ha="right")
    plt.ylabel("Feature value")
    plt.title("Top 3 discriminative features: mean ± SD with Welch p-values")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved top-feature stats plot to: {out_path}")
    plt.show()


def save_outputs_tuned(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    importance_outputs: Dict[str, pd.DataFrame],
    metrics: dict,
    stats_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    out_dir: str,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labeled_csv = out / "labeled_segments.csv"
    corr_csv = out / "label_feature_correlations.csv"
    comparison_csv = out / "tuned_model_comparison.csv"
    lr_importance_csv = out / "logistic_regression_importance.csv"
    rf_importance_csv = out / "random_forest_importance.csv"
    hgb_importance_csv = out / "hist_gradient_boosting_permutation_importance.csv"
    stats_csv = out / "feature_significance_tests.csv"
    consensus_csv = out / "feature_consensus_ranking.csv"
    metrics_json = out / "tuned_model_metrics.json"

    df.to_csv(labeled_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    importance_outputs["logistic_regression"].to_csv(lr_importance_csv, index=False)
    importance_outputs["random_forest"].to_csv(rf_importance_csv, index=False)
    importance_outputs["hist_gradient_boosting"].to_csv(hgb_importance_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    consensus_df.to_csv(consensus_csv, index=False)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved labeled segments:                    {labeled_csv}")
    print(f"Saved correlations:                       {corr_csv}")
    print(f"Saved tuned model comparison:             {comparison_csv}")
    print(f"Saved logistic regression importance:     {lr_importance_csv}")
    print(f"Saved random forest importance:           {rf_importance_csv}")
    print(f"Saved HGB permutation importance:         {hgb_importance_csv}")
    print(f"Saved significance tests:                 {stats_csv}")
    print(f"Saved consensus ranking:                  {consensus_csv}")
    print(f"Saved tuned model metrics:                {metrics_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a video feature CSV against manually labeled interesting times."
    )
    parser.add_argument("--csv-path", default=CSV_PATH, help="Path to video_features.csv")
    parser.add_argument(
        "--interesting-times",
        type=str,
        default="",
        help="Optional comma-separated times in seconds that override INTERESTING_TIMES in the script.",
    )
    parser.add_argument(
        "--interesting-times-file",
        type=str,
        default=INTERESTING_TIMES_FILE,
        help="Optional text file with one interesting time per line.",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=TOLERANCE_SEC,
        help="Extra tolerance when matching a labeled time to a segment window.",
    )
    parser.add_argument(
        "--feature-list",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns to analyze.",
    )
    parser.add_argument("--out-dir", default=OUT_DIR, help="Directory for output CSV/JSON files.")
    parser.add_argument(
        "--save-plots-prefix",
        default=SAVE_PLOTS_PREFIX,
        help="Optional prefix for saving plot PNGs.",
    )
    parser.add_argument(
        "--scoring",
        default="roc_auc",
        choices=["roc_auc", "f1", "accuracy", "balanced_accuracy"],
        help="Metric to optimize during hyperparameter search.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Threshold for converting predicted probabilities into class labels.",
    )
    args = parser.parse_args()

    df = load_csv(args.csv_path)

    interesting_times = list(INTERESTING_TIMES)
    if args.interesting_times:
        interesting_times = parse_times(args.interesting_times)

    if args.interesting_times_file:
        with open(args.interesting_times_file, "r", encoding="utf-8") as f:
            from_file = [float(line.strip()) for line in f if line.strip()]
        interesting_times = sorted(set(interesting_times + from_file))

    if not interesting_times:
        raise ValueError(
            "No interesting times were provided. Edit INTERESTING_TIMES at the top of the script, "
            "or use --interesting-times / --interesting-times-file."
        )

    print("Using interesting times (seconds):")
    print(interesting_times)

    feature_cols = [x.strip() for x in args.feature_list.split(",") if x.strip()]

    df = label_segments_by_times(df, interesting_times, tolerance_sec=args.tolerance_sec)
    summarize_labels(df)

    corr_df = point_biserial_like_scores(df, feature_cols)
    print("Top simple correlations with interesting labels:")
    if corr_df.empty:
        print("No valid feature correlations could be computed.")
    else:
        print(corr_df.head(15).to_string(index=False))

    comparison_df, importance_outputs, metrics, roc_curve_data = evaluate_models_tuned(
        df=df,
        feature_cols=feature_cols,
        scoring=args.scoring,
        threshold=args.classification_threshold,
    )

    stats_df = compute_feature_significance(df, feature_cols)
    consensus_df = rank_features_consensus(corr_df, importance_outputs, stats_df)
    top3_features = consensus_df["feature"].head(3).tolist()

    print("\nTuned model comparison:")
    print(comparison_df.to_string(index=False))

    print("\nTop logistic regression coefficients:")
    print(importance_outputs["logistic_regression"].head(15).to_string(index=False))

    print("\nTop random forest importances:")
    print(importance_outputs["random_forest"].head(15).to_string(index=False))

    print("\nTop HistGradientBoosting permutation importances:")
    print(importance_outputs["hist_gradient_boosting"].head(15).to_string(index=False))

    print("\nTop significance test results:")
    print(stats_df.head(15).to_string(index=False))

    print("\nConsensus top 10 features:")
    print(consensus_df.head(10).to_string(index=False))

    print("\nSelected top 3 descriptive features:")
    print(top3_features)

    top3_stats = stats_df[stats_df["feature"].isin(top3_features)].copy()
    top3_stats["feature"] = pd.Categorical(top3_stats["feature"], categories=top3_features, ordered=True)
    top3_stats = top3_stats.sort_values("feature")

    print("\nTop 3 feature statistics:")
    print(
        top3_stats[
            [
                "feature",
                "interesting_mean",
                "interesting_std",
                "other_mean",
                "other_std",
                "mean_difference",
                "welch_t_p_value",
                "cohen_d",
            ]
        ].to_string(index=False)
    )

    print("\nLogistic regression report:")
    print(metrics["models"]["logistic_regression"]["classification_report"])
    print(f"Logistic regression ROC-AUC: {metrics['models']['logistic_regression']['test_roc_auc']:.4f}")
    print("Best params:", metrics["models"]["logistic_regression"]["best_params"])

    print("\nRandom forest report:")
    print(metrics["models"]["random_forest"]["classification_report"])
    print(f"Random forest ROC-AUC:       {metrics['models']['random_forest']['test_roc_auc']:.4f}")
    print("Best params:", metrics["models"]["random_forest"]["best_params"])

    print("\nHistGradientBoosting report:")
    print(metrics["models"]["hist_gradient_boosting"]["classification_report"])
    print(f"HistGradientBoosting ROC-AUC:{metrics['models']['hist_gradient_boosting']['test_roc_auc']:.4f}")
    print("Best params:", metrics["models"]["hist_gradient_boosting"]["best_params"])

    plot_roc = f"{args.save_plots_prefix}_roc.png" if args.save_plots_prefix else None
    plot_top3 = f"{args.save_plots_prefix}_top3_stats.png" if args.save_plots_prefix else None

    plot_roc_curves(roc_curve_data, out_path=plot_roc)
    plot_top3_feature_stats(stats_df, top3_features, out_path=plot_top3)

    save_outputs_tuned(
        df=df,
        corr_df=corr_df,
        comparison_df=comparison_df,
        importance_outputs=importance_outputs,
        metrics=metrics,
        stats_df=stats_df,
        consensus_df=consensus_df,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()