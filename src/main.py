from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_pr_curve,
    save_results_table,
)
from preprocess import scale_data, split_data
from train import train_logistic_regression
from utils import apply_nearmiss, apply_smote, apply_smoteenn


def print_section(title: str) -> None:
    """Print a formatted section heading"""
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def print_class_distribution(y: pd.Series, label: str) -> None:
    """Print class counts for a dataset"""
    counts = y.value_counts().sort_index()
    legitimate = counts.get(0, 0)
    fraud = counts.get(1, 0)
    print(f"{label}: Legitimate = {legitimate:,}, Fraud = {fraud:,}")


def print_metrics(condition_name: str, results: dict[str, float], training_size: int) -> None:
    """Print evaluation metrics for one condition"""
    print(f"\n{condition_name}")
    print(f"  Training set size : {training_size:,}")
    print(f"  Precision         : {results['Precision']:.4f}")
    print(f"  Recall            : {results['Recall']:.4f}")
    print(f"  F1-score          : {results['F1-score']:.4f}")
    print(f"  ROC-AUC           : {results['ROC-AUC']:.4f}")
    print(f"  PR-AUC            : {results['PR-AUC']:.4f}")


def run_condition(
    condition_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    all_results: list[dict[str, Any]],
    models: dict[str, LogisticRegression],
    class_weight: Optional[str] = None,
) -> None:
    """
    Train and evaluate one experimental condition

    All conditions use the same held out test set
    This keeps the evaluation consistent across conditions
    """
    try:
        model = train_logistic_regression(X_train, y_train, class_weight=class_weight)
        results = evaluate_model(model, X_test, y_test)
        save_confusion_matrix(model, X_test, y_test, condition_name)
    except Exception as e:
        raise RuntimeError(f"Error during {condition_name}: {e}") from e

    all_results.append({
        "Condition": condition_name,
        **results,
        "Training Set Size": len(X_train),
    })

    models[condition_name] = model
    print_metrics(condition_name, results, len(X_train))


def main() -> None:
    """
    Run the full five condition fraud detection experiment

    Pipeline order is fixed as split -> scale -> resample -> train -> evaluate
    Scaling is fitted on training data only
    Resampling is applied after scaling so distance based methods use the same scaled feature space
    """
    data_path = Path("data/creditcard.csv")

    print_section("Credit Card Fraud Detection Pipeline")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    required_columns = {"Class", "Amount", "Time"}
    missing = required_columns - set(df.columns)
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return

    print(f"Dataset loaded: {data_path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Split first before any fitting
    # This prevents scaler and resampler statistics from using the test set
    try:
        X_train, X_test, y_train, y_test = split_data(df)
    except Exception as e:
        print(f"Error during data split: {e}")
        return

    print_section("Train/Test Split")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape    : {X_test.shape}")
    print_class_distribution(y_train, "Training distribution")
    print_class_distribution(y_test, "Test distribution")

    # Fit the scaler on training data only
    try:
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return

    print_section("Feature Scaling")
    print("Scaled features: Amount, Time")

    all_results: list[dict[str, Any]] = []
    models: dict[str, LogisticRegression] = {}

    print_section("Model Evaluation")

    try:
        # Condition 1 no imbalance handling
        # This is the reference condition
        run_condition(
            "Baseline",
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            all_results, models,
        )

        # Condition 2 class weighting
        # The training distribution is unchanged
        run_condition(
            "Class Weighted",
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            all_results, models,
            class_weight="balanced",
        )

        # Conditions 3 to 5 apply resampling to the training set only

        # Condition 3 SMOTE resampling
        print("\nApplying SMOTE...")
        X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)
        print_class_distribution(y_train_smote, "SMOTE distribution")
        run_condition(
            "SMOTE",
            X_train_smote, y_train_smote,
            X_test_scaled, y_test,
            all_results, models,
        )

        # Condition 4 NearMiss-1 resampling
        print("\nApplying NearMiss-1...")
        X_train_nearmiss, y_train_nearmiss = apply_nearmiss(X_train_scaled, y_train)
        print_class_distribution(y_train_nearmiss, "NearMiss-1 distribution")
        run_condition(
            "NearMiss-1",
            X_train_nearmiss, y_train_nearmiss,
            X_test_scaled, y_test,
            all_results, models,
        )

        # Condition 5 SMOTEENN resampling
        print("\nApplying SMOTEENN...")
        X_train_smoteenn, y_train_smoteenn = apply_smoteenn(X_train_scaled, y_train)
        print_class_distribution(y_train_smoteenn, "SMOTEENN distribution")
        run_condition(
            "SMOTEENN",
            X_train_smoteenn, y_train_smoteenn,
            X_test_scaled, y_test,
            all_results, models,
        )

    except Exception as e:
        print(e)
        return

    if not all_results:
        print("Error: No results were generated.")
        return

    results_df = pd.DataFrame(all_results)

    # Round metrics for the dissertation results table
    metric_columns = ["Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
    results_df[metric_columns] = results_df[metric_columns].round(4)

    save_results_table(results_df, "results/tables/results_summary.csv")
    save_pr_curve(models, X_test_scaled, y_test)

    print_section("Final Results Summary")
    print("")
    print(
        f"{'Condition':<15}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1-score':>12}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
        f"{'Train Size':>15}"
    )
    print("-" * 82)

    for _, row in results_df.iterrows():
        print(
            f"{row['Condition']:<15}"
            f"{row['Precision']:>10.4f}"
            f"{row['Recall']:>10.4f}"
            f"{row['F1-score']:>12.4f}"
            f"{row['ROC-AUC']:>10.4f}"
            f"{row['PR-AUC']:>10.4f}"
            f"{int(row['Training Set Size']):>15,}"
        )


if __name__ == "__main__":
    main()
