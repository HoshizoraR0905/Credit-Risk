from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


def explain_logreg_one_pipeline(
    pipe, X_row_df: pd.DataFrame, k: int = 3
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Return:
      pd_hat: predicted probability of class 1
      logit:  log-odds for class 1
      reasons: list of dicts with keys:
          feature, impact, direction, value(optional)
    Notes:
      This assumes the last step of the pipeline is LogisticRegression (named 'clf').
      And that the step before it produces a numeric matrix with feature names available.
    """
    if len(X_row_df) != 1:
        raise ValueError("X_row_df must have exactly one row")

    # ---- predict ----
    pd_hat = float(pipe.predict_proba(X_row_df)[0, 1])

    # ---- get transformed features ----
    # assumes pipeline steps: preprocessor -> clf
    # adjust names if your pipeline uses different step names
    pre = pipe.named_steps.get("preprocess", None)
    clf = pipe.named_steps.get("model", None)
    if pre is None or clf is None:
        raise ValueError("Pipeline must have named steps: 'preprocessor' and 'clf'")

    Z = pre.transform(X_row_df)  # shape (1, d)
    Z = np.asarray(Z).reshape(1, -1)

    # feature names
    if hasattr(pre, "get_feature_names_out"):
        feat_names = list(pre.get_feature_names_out())
    else:
        feat_names = [f"f{i}" for i in range(Z.shape[1])]

    coef = np.asarray(clf.coef_).reshape(-1)  # shape (d,)
    intercept = float(np.asarray(clf.intercept_).reshape(-1)[0])

    contrib = Z.reshape(-1) * coef  # per-feature contribution to logit
    logit = float(intercept + contrib.sum())

    # top-k by absolute contribution
    idx = np.argsort(np.abs(contrib))[::-1][:k]

    reasons = []
    for j in idx:
        impact = float(contrib[j])
        direction = "increases_risk" if impact > 0 else "decreases_risk"
        reasons.append(
            {
                "feature": str(feat_names[j]),
                "impact": impact,
                "direction": direction,
            }
        )

    return pd_hat, logit, reasons
