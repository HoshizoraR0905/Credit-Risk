# Credit Risk Modeling with Profit-Based Decision Optimization

## Overview

This project builds a credit risk model using logistic regression and focuses on
decision-oriented evaluation by separating probability estimation from loan
approval decisions. Instead of using a fixed probability threshold, a
profit-based decision rule is constructed using predicted default probabilities
and loan-level features.

## Problem Statement

Given borrower features X, estimate the probability of default, and design a
decision rule that maximizes expected profit. The decision accounts for
asymmetric outcomes: approving a default leads to principal loss, while
rejecting a non-default leads to missed interest income.

## Methods
- Data preprocessing using sklearn Pipeline (imputation, scaling, one-hot encoding)
- Logistic regression with L2 regularization
- Cross-validated ROC-AUC for model selection
- Profit-based decision rule using expected profit combining predicted default probability, loan amount, and interest rate

## Results
- ROC-AUC: 0.89 (5-fold cross-validation)
- Profit-based decision rule improves total realized profit by ~4% on test data
- Evaluation includes both statistical metrics and decision-based metrics

## Key Insight
Separating prediction and decision leads to more meaningful evaluation.
While the predictive model generalizes well in terms of ROC-AUC, decision rules
based on thresholds or profit can be sensitive to data splits. This highlights
the difference between predictive accuracy and decision-level performance.

## Tech Stack
Python, Pandas, scikit-learn, FAISS, SentenceTransformers, Ollama

## Files
- main.ipynb: model training and evaluation
- build_index.py: document embedding and FAISS index construction
- retrieve.py: retrieval of relevant rules and cases
- explain.py: RAG-based explanation generation
