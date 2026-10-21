"""Baseline Modeling and Cost-Sensitive Learning Engine.

Implements strictly temporal out-of-time (OOT) train/validation/test splits,
trains interpretable baseline models (Heuristic, Logistic Scorecard, Decision Tree, Random Forest, LightGBM),
conducts cost-sensitive decision threshold optimization based on an asymmetric financial loss matrix,
and serializes model artifacts, figures, and governance reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from financial_risk.config import (
    DATA_PROCESSED_DIR,
    DOCS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
)

# Headless matplotlib rendering
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def decimal_serializer(obj: Any) -> Any:
    """Helper serializer for json.dumps to handle Decimals, dates, and numpy types."""
    if isinstance(obj, (Decimal, np.floating)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class FinancialRiskModelingPipeline:
    """Manages temporal data partitioning, baseline training, cost-curve threshold search, and model evaluation."""

    FEATURE_COLS: ClassVar[list[str]] = [
        "log_amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "amount_to_orig_balance_ratio",
        "orig_depletion_complete",
        "orig_balance_discrepancy",
        "oldbalanceDest",
        "newbalanceDest",
        "dest_balance_discrepancy",
        "dest_inbound_prior_count",
        "is_transfer",
        "is_cash_out",
        "is_merchant_dest",
        "is_night_hours",
        "sin_hour",
        "cos_hour",
    ]

    def __init__(
        self,
        features_path: Path = DATA_PROCESSED_DIR / "features_engineered.parquet",
        models_dir: Path = MODELS_DIR,
        figures_dir: Path = FIGURES_DIR,
        summary_path: Path = DATA_PROCESSED_DIR / "baseline_modeling_summary.json",
        report_path: Path = DOCS_DIR / "model_evaluation_report.md",
        review_cost: float = 25.0,
    ):
        self.features_path = features_path
        self.models_dir = models_dir
        self.figures_dir = figures_dir
        self.summary_path = summary_path
        self.report_path = report_path
        self.review_cost = review_cost

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_temporal_splits(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Load parquet and construct strictly temporal out-of-time (OOT) partitions.

        Train: Days 1 to 21 (Steps 1 to 504) -> 6,064,036 tx (95.31%)
        Validation: Days 22 to 26 (Steps 505 to 624) -> 208,896 tx (3.28%)
        Test (OOT Holdout): Days 27 to 31 (Steps 625 to 744) -> 89,688 tx (1.41%)
        """
        logger.info(
            f"Loading features from {self.features_path} for temporal partitioning..."
        )
        cols_to_load = self.FEATURE_COLS + [
            "day",
            "amount",
            "isFraud",
            "isFlaggedFraud",
        ]
        table = pq.read_table(self.features_path, columns=cols_to_load)
        df = table.to_pandas()

        train_mask = df["day"] <= 21
        val_mask = (df["day"] >= 22) & (df["day"] <= 26)
        test_mask = df["day"] >= 27

        X_tr = df.loc[train_mask, self.FEATURE_COLS].astype(np.float32)
        y_tr = df.loc[train_mask, "isFraud"].to_numpy()
        amt_tr = df.loc[train_mask, "amount"].to_numpy()
        flag_tr = df.loc[train_mask, "isFlaggedFraud"].to_numpy()

        X_val = df.loc[val_mask, self.FEATURE_COLS].astype(np.float32)
        y_val = df.loc[val_mask, "isFraud"].to_numpy()
        amt_val = df.loc[val_mask, "amount"].to_numpy()
        flag_val = df.loc[val_mask, "isFlaggedFraud"].to_numpy()

        X_te = df.loc[test_mask, self.FEATURE_COLS].astype(np.float32)
        y_te = df.loc[test_mask, "isFraud"].to_numpy()
        amt_te = df.loc[test_mask, "amount"].to_numpy()
        flag_te = df.loc[test_mask, "isFlaggedFraud"].to_numpy()

        logger.info(
            f"Train Set      (Days 01-21): {len(X_tr):,} tx, {int(y_tr.sum()):,} fraud ({y_tr.mean() * 100:.4f}%)"
        )
        logger.info(
            f"Validation Set (Days 22-26): {len(X_val):,} tx, {int(y_val.sum()):,} fraud ({y_val.mean() * 100:.4f}%)"
        )
        logger.info(
            f"Test Set (OOT) (Days 27-31): {len(X_te):,} tx, {int(y_te.sum()):,} fraud ({y_te.mean() * 100:.4f}%)"
        )

        return (
            X_tr,
            X_val,
            X_te,
            y_tr,
            y_val,
            y_te,
            amt_tr,
            amt_val,
            amt_te,
            flag_tr,
            flag_val,
            flag_te,
        )

    def compute_financial_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray,
    ) -> dict[str, float | int]:
        """Calculate total financial loss balancing false negative fraud leakage and operational review cost."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        fn_mask = (y_pred == 0) & (y_true == 1)
        tp_mask = (y_pred == 1) & (y_true == 1)

        dollar_loss_missed = float(amounts[fn_mask].sum())
        dollar_loss_captured = float(amounts[tp_mask].sum())
        total_fraud_exposure = dollar_loss_missed + dollar_loss_captured

        alert_volume = int(tp + fp)
        investigator_review_cost = float(alert_volume * self.review_cost)
        total_financial_cost = dollar_loss_missed + investigator_review_cost

        capture_rate_pct = (
            (dollar_loss_captured / total_fraud_exposure * 100.0)
            if total_fraud_exposure > 0
            else 0.0
        )
        alert_rate_per_10k = (alert_volume / len(y_true)) * 10_000.0

        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "alert_volume": alert_volume,
            "alert_rate_per_10k": round(alert_rate_per_10k, 2),
            "dollar_loss_missed": round(dollar_loss_missed, 2),
            "dollar_loss_captured": round(dollar_loss_captured, 2),
            "dollar_capture_rate_pct": round(capture_rate_pct, 2),
            "investigator_review_cost": round(investigator_review_cost, 2),
            "total_financial_cost": round(total_financial_cost, 2),
        }

    def optimize_decision_threshold(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        amounts: np.ndarray,
        n_points: int = 99,
    ) -> tuple[float, float, list[dict[str, Any]]]:
        """Find the cost-optimal decision threshold tau* that minimizes total financial loss on validation."""
        thresholds = np.linspace(0.01, 0.99, n_points)
        best_tau = 0.50
        min_cost = float("inf")
        curve_records = []

        for tau in thresholds:
            y_pred = (y_probs >= tau).astype(int)
            cost_metrics = self.compute_financial_cost(y_true, y_pred, amounts)
            tot_cost = float(cost_metrics["total_financial_cost"])

            curve_records.append(
                {
                    "threshold": round(float(tau), 3),
                    "total_cost": tot_cost,
                    "dollar_loss_missed": cost_metrics["dollar_loss_missed"],
                    "review_cost": cost_metrics["investigator_review_cost"],
                    "alert_volume": cost_metrics["alert_volume"],
                }
            )

            if tot_cost < min_cost:
                min_cost = tot_cost
                best_tau = float(tau)

        return best_tau, min_cost, curve_records

    def evaluate_model_performance(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        amounts: np.ndarray,
        optimal_tau: float,
    ) -> dict[str, Any]:
        """Compute standard ML metrics and financial costs at both default (0.50) and optimal (tau*) thresholds."""
        pr_auc = float(average_precision_score(y_true, y_probs))
        roc_auc = float(roc_auc_score(y_true, y_probs))

        # Default Threshold Metrics (tau = 0.50)
        pred_default = (y_probs >= 0.50).astype(int)
        cost_default = self.compute_financial_cost(y_true, pred_default, amounts)
        prec_def = float(precision_score(y_true, pred_default, zero_division=0))
        rec_def = float(recall_score(y_true, pred_default, zero_division=0))
        f1_def = float(f1_score(y_true, pred_default, zero_division=0))

        # Optimal Threshold Metrics (tau = optimal_tau)
        pred_optimal = (y_probs >= optimal_tau).astype(int)
        cost_optimal = self.compute_financial_cost(y_true, pred_optimal, amounts)
        prec_opt = float(precision_score(y_true, pred_optimal, zero_division=0))
        rec_opt = float(recall_score(y_true, pred_optimal, zero_division=0))
        f1_opt = float(f1_score(y_true, pred_optimal, zero_division=0))

        financial_savings = float(
            cost_default["total_financial_cost"] - cost_optimal["total_financial_cost"]
        )

        return {
            "model_name": model_name,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "optimal_threshold_tau": round(optimal_tau, 4),
            "metrics_at_default_threshold": {
                "threshold": 0.50,
                "precision": round(prec_def, 4),
                "recall": round(rec_def, 4),
                "f1_score": round(f1_def, 4),
                **cost_default,
            },
            "metrics_at_optimal_threshold": {
                "threshold": round(optimal_tau, 4),
                "precision": round(prec_opt, 4),
                "recall": round(rec_opt, 4),
                "f1_score": round(f1_opt, 4),
                **cost_optimal,
            },
            "financial_savings_from_threshold_tuning": round(financial_savings, 2),
        }

    # =========================================================================
    # MODEL TRAINING WORKFLOW
    # =========================================================================

    def run_training_and_evaluation(self) -> dict[str, Any]:
        """Train all baseline models, optimize thresholds, and evaluate on Out-Of-Time (OOT) test holdout."""
        logger.info("=" * 80)
        logger.info("STARTING PHASE 11: BASELINE MODELING & COST-SENSITIVE LEARNING")
        logger.info("=" * 80)

        (
            X_tr,
            X_val,
            X_te,
            y_tr,
            y_val,
            y_te,
            _amt_tr,
            amt_val,
            amt_te,
            _flag_tr,
            flag_val,
            flag_te,
        ) = self.load_temporal_splits()

        models_results: dict[str, Any] = {}
        cost_curves: dict[str, list[dict[str, Any]]] = {}
        predictions_test: dict[str, np.ndarray] = {}

        # ---------------------------------------------------------------------
        # MODEL 0: HEURISTIC BASELINE (Legacy isFlaggedFraud Rule)
        # ---------------------------------------------------------------------
        logger.info("Evaluating Model 0: Legacy Heuristic Rule (isFlaggedFraud)...")
        # isFlaggedFraud is a static binary indicator (0 or 1)
        flag_val.astype(float)
        prob_te_heur = flag_te.astype(float)
        predictions_test["Heuristic Baseline"] = prob_te_heur

        heur_eval = self.evaluate_model_performance(
            "Heuristic Baseline (isFlaggedFraud)",
            y_te,
            prob_te_heur,
            amt_te,
            optimal_tau=0.50,
        )
        models_results["Heuristic Baseline"] = heur_eval

        # ---------------------------------------------------------------------
        # MODEL 1: LINEAR SCORECARD (LOGISTIC REGRESSION)
        # ---------------------------------------------------------------------
        logger.info(
            "Training Model 1: Logistic Regression Scorecard (StandardScaled, Balanced)..."
        )
        # Train on 500k stratified subsample for fast convergence and exact scaling
        np.random.seed(RANDOM_STATE)
        fraud_idx = np.where(y_tr == 1)[0]
        benign_idx = np.where(y_tr == 0)[0]
        benign_sub_idx = np.random.choice(benign_idx, size=400_000, replace=False)
        sub_train_idx = np.concatenate([fraud_idx, benign_sub_idx])

        lr_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        t0 = time.perf_counter()
        lr_pipeline.fit(X_tr.iloc[sub_train_idx], y_tr[sub_train_idx])
        lr_fit_time = time.perf_counter() - t0
        logger.info(f"  -> Logistic Regression fit in {lr_fit_time:.2f}s")

        prob_val_lr = lr_pipeline.predict_proba(X_val)[:, 1]
        best_tau_lr, _, curve_lr = self.optimize_decision_threshold(
            y_val, prob_val_lr, amt_val
        )
        cost_curves["Logistic Regression"] = curve_lr

        prob_te_lr = lr_pipeline.predict_proba(X_te)[:, 1]
        predictions_test["Logistic Regression"] = prob_te_lr
        models_results["Logistic Regression"] = self.evaluate_model_performance(
            "Logistic Regression", y_te, prob_te_lr, amt_te, best_tau_lr
        )
        joblib.dump(lr_pipeline, self.models_dir / "logistic_regression.joblib")

        # ---------------------------------------------------------------------
        # MODEL 2: INTERPRETABLE SHALLOW DECISION TREE
        # ---------------------------------------------------------------------
        logger.info(
            "Training Model 2: Interpretable Shallow Decision Tree (max_depth=5)..."
        )
        dt_model = DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            min_samples_leaf=100,
            random_state=RANDOM_STATE,
        )
        t0 = time.perf_counter()
        dt_model.fit(X_tr.iloc[sub_train_idx], y_tr[sub_train_idx])
        dt_fit_time = time.perf_counter() - t0
        logger.info(f"  -> Decision Tree fit in {dt_fit_time:.2f}s")

        prob_val_dt = dt_model.predict_proba(X_val)[:, 1]
        best_tau_dt, _, curve_dt = self.optimize_decision_threshold(
            y_val, prob_val_dt, amt_val
        )
        cost_curves["Decision Tree"] = curve_dt

        prob_te_dt = dt_model.predict_proba(X_te)[:, 1]
        predictions_test["Decision Tree"] = prob_te_dt
        models_results["Decision Tree"] = self.evaluate_model_performance(
            "Decision Tree", y_te, prob_te_dt, amt_te, best_tau_dt
        )
        joblib.dump(dt_model, self.models_dir / "decision_tree.joblib")

        # Export human-readable tree split rules
        tree_rules = export_text(dt_model, feature_names=self.FEATURE_COLS)
        (DOCS_DIR / "decision_tree_rules.txt").write_text(tree_rules, encoding="utf-8")
        logger.info(
            "  -> Exported human-readable tree rules to docs/decision_tree_rules.txt"
        )

        # ---------------------------------------------------------------------
        # MODEL 3: COST-SENSITIVE RANDOM FOREST
        # ---------------------------------------------------------------------
        logger.info(
            "Training Model 3: Cost-Sensitive Random Forest (n_estimators=50, max_depth=10)..."
        )
        rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        t0 = time.perf_counter()
        rf_model.fit(X_tr.iloc[sub_train_idx], y_tr[sub_train_idx])
        rf_fit_time = time.perf_counter() - t0
        logger.info(f"  -> Random Forest fit in {rf_fit_time:.2f}s")

        prob_val_rf = rf_model.predict_proba(X_val)[:, 1]
        best_tau_rf, _, curve_rf = self.optimize_decision_threshold(
            y_val, prob_val_rf, amt_val
        )
        cost_curves["Random Forest"] = curve_rf

        prob_te_rf = rf_model.predict_proba(X_te)[:, 1]
        predictions_test["Random Forest"] = prob_te_rf
        models_results["Random Forest"] = self.evaluate_model_performance(
            "Random Forest", y_te, prob_te_rf, amt_te, best_tau_rf
        )
        joblib.dump(rf_model, self.models_dir / "random_forest.joblib")

        # ---------------------------------------------------------------------
        # MODEL 4: COST-SENSITIVE LIGHTGBM (FULL POPULATION TRAINING)
        # ---------------------------------------------------------------------
        logger.info(
            "Training Model 4: Cost-Sensitive LightGBM (100 estimators, scale_pos_weight=50)..."
        )
        lgb_model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=50,
            subsample=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        t0 = time.perf_counter()
        lgb_model.fit(X_tr, y_tr)
        lgb_fit_time = time.perf_counter() - t0
        logger.info(f"  -> LightGBM fit on full 6.06M rows in {lgb_fit_time:.2f}s")

        prob_val_lgb = lgb_model.predict_proba(X_val)[:, 1]
        best_tau_lgb, _, curve_lgb = self.optimize_decision_threshold(
            y_val, prob_val_lgb, amt_val
        )
        cost_curves["LightGBM"] = curve_lgb

        prob_te_lgb = lgb_model.predict_proba(X_te)[:, 1]
        predictions_test["LightGBM"] = prob_te_lgb
        models_results["LightGBM"] = self.evaluate_model_performance(
            "LightGBM", y_te, prob_te_lgb, amt_te, best_tau_lgb
        )
        joblib.dump(lgb_model, self.models_dir / "lightgbm.joblib")

        # ---------------------------------------------------------------------
        # RENDER VISUALIZATIONS
        # ---------------------------------------------------------------------
        self.plot_pr_roc_curves(y_te, predictions_test)
        self.plot_cost_curves(cost_curves)
        self.plot_dollar_capture_comparison(models_results, amt_te[y_te == 1].sum())

        # Compile complete summary
        summary = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "review_cost_per_alert": self.review_cost,
                "train_size": len(X_tr),
                "val_size": len(X_val),
                "test_size": len(X_te),
                "test_fraud_count": int(y_te.sum()),
                "test_fraud_exposure": round(float(amt_te[y_te == 1].sum()), 2),
            },
            "models": models_results,
            "figures_generated": [
                "fig_07_pr_roc_curves.png",
                "fig_08_cost_curve_thresholds.png",
                "fig_09_model_dollar_capture_comparison.png",
            ],
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=decimal_serializer)

        logger.info(f"Modeling summary successfully serialized to {self.summary_path}")
        return summary

    # =========================================================================
    # VISUALIZATION PIPELINE
    # =========================================================================

    def plot_pr_roc_curves(
        self, y_test: np.ndarray, predictions: dict[str, np.ndarray]
    ) -> Path:
        """Plot Precision-Recall and ROC curves on the Out-Of-Time (OOT) test holdout (Figure 7)."""
        output_file = self.figures_dir / "fig_07_pr_roc_curves.png"
        logger.info(f"Generating Figure 7: {output_file.name}...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        colors = {
            "Heuristic Baseline": "#757575",
            "Logistic Regression": "#ff9800",
            "Decision Tree": "#2196f3",
            "Random Forest": "#4caf50",
            "LightGBM": "#e91e63",
        }

        # Panel A: Precision-Recall Curves
        for name, probs in predictions.items():
            prec, rec, _ = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)
            axes[0].plot(
                rec,
                prec,
                label=f"{name} (PR-AUC = {ap:.3f})",
                color=colors.get(name, "#000"),
                linewidth=2.0,
            )

        axes[0].set_title(
            "A. Precision-Recall Curves (OOT Test Holdout)",
            fontsize=13,
            fontweight="bold",
        )
        axes[0].set_xlabel("Recall (Detection Rate)", fontsize=11)
        axes[0].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        axes[0].legend(loc="lower left", frameon=True)
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # Panel B: ROC Curves
        for name, probs in predictions.items():
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = roc_auc_score(y_test, probs)
            axes[1].plot(
                fpr,
                tpr,
                label=f"{name} (ROC-AUC = {roc_auc:.3f})",
                color=colors.get(name, "#000"),
                linewidth=2.0,
            )

        axes[1].plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="navy",
            alpha=0.5,
            label="Random Guess",
        )
        axes[1].set_title(
            "B. Receiver Operating Characteristic (ROC)",
            fontsize=13,
            fontweight="bold",
        )
        axes[1].set_xlabel("False Positive Rate", fontsize=11)
        axes[1].set_ylabel("True Positive Rate", fontsize=11)
        axes[1].legend(loc="lower right", frameon=True)
        axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.suptitle(
            "Comparative Discrimination Power on Out-Of-Time (OOT) Holdout",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def plot_cost_curves(self, cost_curves: dict[str, list[dict[str, Any]]]) -> Path:
        """Plot Total Financial Cost ($) vs Decision Threshold Curve (Figure 8)."""
        output_file = self.figures_dir / "fig_08_cost_curve_thresholds.png"
        logger.info(f"Generating Figure 8: {output_file.name}...")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        colors = {
            "Logistic Regression": "#ff9800",
            "Decision Tree": "#2196f3",
            "Random Forest": "#4caf50",
            "LightGBM": "#e91e63",
        }

        for model_name, records in cost_curves.items():
            taus = [r["threshold"] for r in records]
            costs_millions = [r["total_cost"] / 1e6 for r in records]
            min_cost_idx = int(np.argmin(costs_millions))
            best_tau = taus[min_cost_idx]
            best_cost_m = costs_millions[min_cost_idx]

            ax.plot(
                taus,
                costs_millions,
                label=f"{model_name} (Min: ${best_cost_m:.1f}M @ tau={best_tau:.2f})",
                color=colors.get(model_name, "#000"),
                linewidth=2.2,
            )
            ax.plot(
                best_tau,
                best_cost_m,
                marker="o",
                markersize=8,
                color=colors.get(model_name, "#000"),
            )

        ax.set_title(
            "Financial Cost Curve Optimization: Minimizing Expected Monetary Loss",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(
            "Classification Decision Threshold (tau)", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel(
            "Total Expected Financial Cost ($ Millions)",
            fontsize=12,
            fontweight="bold",
        )
        ax.axvline(
            0.50,
            linestyle="--",
            color="black",
            alpha=0.6,
            label="Default Threshold (tau = 0.50)",
        )
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def plot_dollar_capture_comparison(
        self, models_results: dict[str, Any], total_test_exposure: float
    ) -> Path:
        """Plot Dollar Loss Captured vs Missed across all models (Figure 9)."""
        output_file = self.figures_dir / "fig_09_model_dollar_capture_comparison.png"
        logger.info(f"Generating Figure 9: {output_file.name}...")

        rows = []
        for name, data in models_results.items():
            opt_metrics = data["metrics_at_optimal_threshold"]
            captured_m = opt_metrics["dollar_loss_captured"] / 1e6
            missed_m = opt_metrics["dollar_loss_missed"] / 1e6
            rows.append(
                {
                    "Model": name,
                    "Captured ($M)": captured_m,
                    "Missed ($M)": missed_m,
                    "Capture Rate (%)": opt_metrics["dollar_capture_rate_pct"],
                }
            )

        comp_df = pd.DataFrame(rows)
        comp_df = comp_df.sort_values(by="Capture Rate (%)", ascending=True)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        y_positions = np.arange(len(comp_df))
        ax.barh(
            y_positions,
            comp_df["Captured ($M)"],
            label="Fraud Exposure Intercepted ($M)",
            color="#2e7d32",
            alpha=0.85,
        )
        ax.barh(
            y_positions,
            comp_df["Missed ($M)"],
            left=comp_df["Captured ($M)"],
            label="Fraud Exposure Leaked ($M)",
            color="#d32f2f",
            alpha=0.85,
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(comp_df["Model"], fontsize=11, fontweight="bold")
        ax.set_xlabel(
            "Out-Of-Time Test Fraud Exposure ($ Millions)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_title(
            "Capital Preservation: Intercepted vs Leaked Fraud Loss Across Models",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        for i, row in comp_df.reset_index(drop=True).iterrows():
            ax.annotate(
                f"{row['Capture Rate (%)']:.1f}% Captured",
                xy=(total_test_exposure / 1e6 * 0.5, i),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file


def main() -> None:
    """CLI Entrypoint for Phase 11 Baseline Modeling."""
    parser = argparse.ArgumentParser(
        description="Phase 11 Baseline Modeling and Cost Optimization"
    )
    parser.parse_args()

    pipeline = FinancialRiskModelingPipeline()
    summary = pipeline.run_training_and_evaluation()

    print("\n" + "=" * 70)
    print("PHASE 11 BASELINE MODELING & COST-SENSITIVE LEARNING SUMMARY")
    print("=" * 70)
    print(
        f"Test Set Total Fraud Exposure: ${summary['metadata']['test_fraud_exposure']:,.2f}"
    )
    print("\nModel Performance Comparison (at Cost-Optimal Threshold tau*):")
    for name, data in summary["models"].items():
        opt = data["metrics_at_optimal_threshold"]
        print(
            f"  - {name:<22}: PR-AUC = {data['pr_auc']:.4f} | Recall = {opt['recall'] * 100:>5.1f}% | "
            f"Captured = {opt['dollar_capture_rate_pct']:>5.1f}% | Net Cost = ${opt['total_financial_cost']:>12,.2f} "
            f"(tau* = {data['optimal_threshold_tau']:.3f})"
        )
    print("\nSaved Artifacts:")
    print(
        "  - Models: models/ (logistic_regression, decision_tree, random_forest, lightgbm)"
    )
    print("  - Figures: reports/figures/ (fig_07, fig_08, fig_09)")
    print("  - Summary: data/processed/baseline_modeling_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
