# Credit Card Fraud Detection - Imbalance Strategy Comparison

This project investigates how different imbalance-handling strategies affect credit card fraud detection performance using Logistic Regression on the ULB credit card fraud dataset.

The aim is not to build the most complex fraud detection model. The aim is to compare several imbalance-handling approaches under the same controlled experimental conditions.

## Project Overview

Credit card fraud datasets are usually highly imbalanced. Fraudulent transactions make up only a very small proportion of the data. This makes accuracy misleading because a model can achieve high accuracy while still failing to detect fraud.

This project compares five experimental conditions:

1. Baseline Logistic Regression with no imbalance handling
2. Class-weighted Logistic Regression
3. SMOTE oversampling
4. NearMiss-1 undersampling
5. SMOTEENN hybrid resampling

All conditions use the same dataset, train/test split, preprocessing steps, model type and evaluation metrics. This allows the effect of each imbalance-handling strategy to be compared fairly.

## Dataset

The project uses the ULB Credit Card Fraud Detection dataset.

The dataset should be placed at:

data/creditcard.csv

The dataset is not included in the submitted project folder due to file size. It can be downloaded from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymised numerical transaction features, including PCA-transformed variables V1 to V28, Time, Amount and the target class label.

## Project Structure

FRAUD-DETECTION-RESAMPLING/
  data/
    creditcard.csv

  notebook/
    exploration.ipynb

  results/
    figures/
      confusion_matrices/
      eda/
      evaluation/
    tables/
      results_summary.csv

  src/
    evaluate.py
    main.py
    preprocess.py
    train.py
    utils.py

  README.md
  requirements.txt

## Installation

Install the required Python packages with:

pip install -r requirements.txt

The main libraries used are:

- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

## Exploratory Data Analysis

This notebook checks the dataset structure, class distribution, feature distributions and correlations. It is used to support preprocessing and evaluation decisions before running the main modelling pipeline.

The notebook produces EDA figures such as:

- class distribution
- amount distribution
- time distribution
- PCA feature correlation heatmap
- amount/time violin plots

EDA figures are saved to:

results/figures/eda/

## Running the Main Pipeline

After placing creditcard.csv in the data folder, run:

python src/main.py

The pipeline follows this order:

split -> scale -> resample -> train -> evaluate

The train/test split is performed before scaling or resampling to avoid data leakage. Scaling is fitted on the training data only and resampling is applied only to the training set.

## Experimental Conditions

The main pipeline evaluates the following conditions:

1. Baseline: Logistic Regression trained on the original imbalanced training data.
2. Class-Weighted: Logistic Regression with class_weight="balanced", without changing the training data distribution.
3. SMOTE: Oversamples the minority fraud class by generating synthetic examples.
4. NearMiss-1: Undersamples the majority legitimate class by retaining selected majority samples close to minority samples.
5. SMOTEENN: Combines SMOTE oversampling with Edited Nearest Neighbours cleaning.

Each condition is evaluated on the same untouched test set.

## Outputs

Running the main pipeline generates the results table at:

results/tables/results_summary.csv

Figures are saved in:

results/figures/

The outputs include:

- a summary table of evaluation metrics
- confusion matrices for each experimental condition
- a combined Precision-Recall curve comparing all conditions

Confusion matrices are saved in:

results/figures/confusion_matrices/

The combined Precision-Recall curve is saved in:

results/figures/evaluation/

## Evaluation Metrics

The models are evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## Reproducibility

The project uses a fixed random_state=42 where applicable. This includes the train/test split, Logistic Regression model and stochastic resampling methods.

This ensures that results are reproducible across runs when the same dataset and package versions are used.

The same fixed test set is used for all experimental conditions.

## Notes

This project was developed as part of a final year dissertation investigating imbalance-handling strategies for credit card fraud detection.

The implementation is intentionally focused on a controlled comparison. Logistic Regression is used as a fixed model so that performance differences can be attributed mainly to the imbalance-handling strategy being tested.