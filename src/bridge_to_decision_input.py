import joblib
import pandas as pd

from src.inputs import DecisionInput
from src.logreg_explain import explain_logreg_one_pipeline


def make_decision_input(
    model_path: str = "artifacts/best_clf.joblib",
    row_path: str = "artifacts/sample_row.pkl",
    applicant_id: str = "sample_row",
    k: int = 3,
) -> DecisionInput:
    pipe = joblib.load(model_path)
    X_row = pd.read_pickle(row_path)

    pd_hat, logit, reasons = explain_logreg_one_pipeline(pipe, X_row, k=k)
    decision = "Rejected" if pd_hat >= 0.5 else "Approved"
    top_factors = [(r["feature"], r["impact"]) for r in reasons]

    raw = {}
    for col in X_row.columns:
        v = X_row.iloc[0][col]
        if isinstance(v, (int, float)):
            raw[col] = float(v)

    return DecisionInput(
        applicant_id=applicant_id,
        decision=decision,
        pd=float(pd_hat),
        top_factors=top_factors,
        raw_features=raw,
    )


def main():
    x = make_decision_input()
    print("pd_hat:", x.pd)
    print("decision:", x.decision)
    print("top_factors:", x.top_factors)
    print("raw_features keys:", list(x.raw_features.keys())[:10])


if __name__ == "__main__":
    main()
