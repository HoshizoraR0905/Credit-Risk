import json
from pathlib import Path
from src.inputs import DecisionInput, build_applicant_summary, build_retrieval_query
from src.bridge_to_decision_input import make_decision_input

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("data/index")
OLLAMA_MODEL = "qwen2.5:7b-instruct"


def load_index_and_docs():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    docs = json.loads((INDEX_DIR / "docs.json").read_text(encoding="utf-8"))
    return index, docs


def retrieve(query: str, k: int = 5):
    """
    Retrieve a mixed set of evidence: prefer some rules + some cases.
    """
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    index, docs = load_index_and_docs()
    scores, ids = index.search(q, 10)  # over-retrieve first

    candidates = []
    for score, idx in zip(scores[0], ids[0]):
        d = docs[int(idx)]
        candidates.append({"score": float(score), "id": d["id"], "type": d["type"], "text": d["text"]})

    rules = [c for c in candidates if c["type"] == "rule"]
    cases = [c for c in candidates if c["type"] == "case"]

    # choose a balanced subset
    out = []
    out.extend(rules[: max(2, k // 2)])     # at least 2 rules if available
    out.extend(cases[: (k - len(out))])     # fill rest with cases
    return out[:k]



def build_prompt(applicant_summary: str, retrieved_docs):
    evidence = "\n".join(
    [f"[E{i+1}] ({d['type']}, score={d['score']:.3f}) {d['text']}" for i, d in enumerate(retrieved_docs)]
    )
    prompt = f"""
You are a credit risk analyst assistant. Write in clear, professional English.

Task:
Explain the model decision using TWO sources only:
(A) the model outputs (PD + top model factors) provided in Applicant summary
(B) the retrieved evidence items [E1..Ek]

Rules:
- Do NOT infer anything from missing evidence ("absence of evidence" is not evidence).
- Do NOT claim facts not present in (A) or (B).
- When referencing retrieved items, cite them as [E#].
- When referencing model factors (top_factors / PD), cite them as [M].
- If you cannot support a statement, write: "Not supported."

Model evidence [M]:
- PD and top_factors in Applicant summary are ground truth model outputs.

Applicant summary:
{applicant_summary}python explain.py

Evidence (retrieved rules/cases):
{evidence}

Output format:
1) Decision explanation (3-6 sentences). Add citations like [E1][E3] at the end of each sentence.
2) Key risk drivers (bullet list). Each bullet must end with citations like [E2].
3) If you propose any next steps, each bullet must be directly supported by evidence; otherwise write "Not supported by evidence".
"""
    return prompt.strip()


if __name__ == "__main__":
    x = make_decision_input(k=3)

    applicant_summary = build_applicant_summary(x)
    query = build_retrieval_query(x)

    docs = retrieve(query, k=5)

    prompt = build_prompt(applicant_summary, docs)
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    print("\n=== Retrieved Evidence (top-5) ===")
    for d in docs:
        print(f"[{d['score']:.3f}] {d['id']} ({d['type']}): {d['text']}")
    print("=== End Evidence ===\n")
    print(resp["message"]["content"])
