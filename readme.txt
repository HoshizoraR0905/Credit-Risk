# Credit Risk Modeling with Cost-Sensitive Threshold Optimization

## Overview
This project builds a credit default prediction model using logistic regression and
focuses on decision-oriented evaluation by optimizing the classification threshold
under asymmetric costs of false positives and false negatives.

## Problem Statement
Given borrower features X, predict probability of default Y, and choose a decision
rule that minimizes expected misclassification cost.

## Methods
- Data preprocessing with sklearn Pipeline (imputation, scaling, one-hot encoding)
- Logistic regression with L2 regularization
- Cross-validated ROC-AUC for model selection
- Cost-sensitive threshold optimization

## Results
- ROC-AUC: 0.78
- Optimal threshold shifts from 0.5 to 0.32 under cost ratio 5:1 (FN:FP)
- Achieved 18% reduction in expected cost compared to naive threshold

## Tech Stack
Python, Pandas, scikit-learn, Jupyter

## Files
- `notebook.ipynb`: main analysis
- `utils.py`: helper functions