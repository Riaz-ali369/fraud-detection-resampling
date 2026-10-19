from typing import Tuple

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to the training data

    SMOTE creates synthetic minority samples using nearest neighbours
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def apply_nearmiss(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply NearMiss-1 undersampling to the training data

    NearMiss-1 keeps majority samples close to minority samples
    This creates a much smaller balanced training set
    """
    nearmiss = NearMiss(version=1)
    X_resampled, y_resampled = nearmiss.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def apply_smoteenn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTEENN hybrid resampling to the training data

    SMOTEENN combines SMOTE oversampling with Edited Nearest Neighbour cleaning
    """
    smoteenn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
