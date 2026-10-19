from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model on the held out test set

    Fraud is treated as the positive class
    PR-AUC is calculated from predicted probabilities
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }


def save_results_table(results_df: pd.DataFrame, output_path: str) -> None:
    """Save the summary results table as a CSV"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)


def save_confusion_matrix(model, X_test, y_test, condition_name: str) -> None:
    """
    Save a confusion matrix heatmap for one condition

    Each cell shows the raw count and the row percentage
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    cm_percent = cm / cm.sum(axis=1, keepdims=True)

    labels = [
        [
            f"{cm[i, j]}\n({cm_percent[i, j]:.2%})"
            for j in range(cm.shape[1])
        ]
        for i in range(cm.shape[0])
    ]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        cbar=True,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {condition_name}")

    output_dir = Path("results/figures/confusion_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = condition_name.lower().replace(" ", "_").replace("-", "_")

    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=300)
    plt.close()


def save_pr_curve(models_dict: dict, X_test, y_test) -> None:
    """
    Save one Precision-Recall curve plot for all conditions

    The curve shows the precision-recall trade off across thresholds
    """
    plt.figure(figsize=(10, 6))

    for condition_name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{condition_name} (PR-AUC={pr_auc:.3f})",
        )

    # Random baseline is the fraud rate
    baseline = y_test.mean()
    plt.hlines(
        baseline,
        xmin=0,
        xmax=1,
        linestyles="dashed",
        linewidth=2,
        label=f"Random Baseline ({baseline:.3f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        frameon=True,
        framealpha=0.9,
    )

    output_dir = Path("results/figures/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve_comparison.png", dpi=300)
    plt.close()
