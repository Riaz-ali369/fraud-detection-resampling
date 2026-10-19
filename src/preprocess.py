from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and test sets

    Stratification keeps the rare fraud class represented in both sets
    The fixed random state makes the split reproducible
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Use the same stratified split for every condition
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test


def scale_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale Amount and Time using the training set only

    This prevents test set leakage
    It also keeps these features on comparable ranges for distance based resampling
    V1 to V28 are already PCA transformed in the original dataset
    """
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit on train and transform both sets
    X_train_scaled["Amount"] = amount_scaler.fit_transform(X_train[["Amount"]])
    X_test_scaled["Amount"] = amount_scaler.transform(X_test[["Amount"]])

    X_train_scaled["Time"] = time_scaler.fit_transform(X_train[["Time"]])
    X_test_scaled["Time"] = time_scaler.transform(X_test[["Time"]])

    return X_train_scaled, X_test_scaled