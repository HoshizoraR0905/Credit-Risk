from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DecisionInput:
    applicant_id: str
    decision: str              # "Approved" / "Rejected"
    pd: float                  # predicted default probability
    top_factors: List[Tuple[str, float]]  # (feature_name, contribution_value)
    raw_features: Dict[str, float]        # optional: actual feature values (utilization etc.)


def build_applicant_summary(x: DecisionInput) -> str:
    # Keep it compact + audit-friendly
    factors = ", ".join([f"{name}={val:.4f}" for name, val in x.top_factors])
    # include a couple raw feature values if available (nice for human readability)
    extras = []
    for key in ["utilization", "credit_history_years", "dti"]:
        if key in x.raw_features:
            extras.append(f"{key}={x.raw_features[key]}")
    extras_str = (", ".join(extras)) if extras else "N/A"

    return (
        f"Applicant ID: {x.applicant_id}\n"
        f"Decision: {x.decision}\n"
        f"Model PD: {x.pd:.3f}\n"
        f"Top model factors (signed contributions): {factors}\n"
        f"Selected raw features: {extras_str}"
    )


def build_retrieval_query(x: DecisionInput) -> str:
    # Use top factor feature names to guide retrieval toward relevant rules/cases
    keys = [name for name, _ in x.top_factors[:5]]
    q = f"{x.decision} credit decision; top model drivers: " + ", ".join(keys)

    # add raw feature values if we have them (only if present)
    for k in ["utilization", "credit_history_years", "dti", "loan_percent_income", "loan_int_rate"]:
        if k in x.raw_features:
            q += f"; {k}={x.raw_features[k]}"
    return q

