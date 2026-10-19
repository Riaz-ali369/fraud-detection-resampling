from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: Optional[str] = None,
) -> LogisticRegression:
    """
    Train a Logistic Regression model for binary classification

    Class weighting can be enabled without changing the training data
    """
    # Fixed Logistic Regression configuration for all conditions
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,            # avoids early convergence issues
        random_state=42,          # ensures reproducible training
        class_weight=class_weight,
    )

    model.fit(X_train, y_train)
    return model
