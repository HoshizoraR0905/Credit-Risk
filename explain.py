import json
from pathlib import Path

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
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    index, docs = load_index_and_docs()
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        d = docs[int(idx)]
        results.append({
            "score": float(score),
            "case_id": d.get("case_id"),
            "text": d.get("text"),
            "raw": d.get("raw", {})
        })

    return results


def build_prompt(retrieved_docs):
    evidence = "\n".join(
        [
            f"[E{i+1}] Case ID: {d['case_id']}, score={d['score']:.3f}\n{d['text']}"
            for i, d in enumerate(retrieved_docs)
        ]
    )

    prompt = f"""
You are a credit risk analyst assistant. Write in clear, professional English.

Task:
Explain the credit decision using only the retrieved evidence items [E1..Ek].

Rules:
- Do NOT claim facts not present in the retrieved evidence.
- When referencing retrieved items, cite them as [E#].
- If a statement is not supported by evidence, write: "Not supported."
- Explain the decision using the profit-based decision layer, not a simple 0.5 probability threshold, unless the evidence explicitly compares against the 0.5 rule.

Evidence:
{evidence}

Output format:
1) Decision explanation: 3-6 sentences, with citations like [E1].
2) Key risk drivers: bullet list, each bullet ending with citations.
3) Business interpretation: 2-3 sentences explaining how the profit-based decision layer supports the decision.
"""
    return prompt.strip()


if __name__ == "__main__":
    query = "Explain rejected applicants with high default probability, high decision score, and low or negative expected profit."

    docs = retrieve(query, k=5)
    prompt = build_prompt(docs)

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    print("\n=== Retrieved Evidence (top-5) ===")
    for d in docs:
        print(f"[{d['score']:.3f}] {d['case_id']}: {d['text']}")
    print("=== End Evidence ===\n")

    print(resp["message"]["content"])