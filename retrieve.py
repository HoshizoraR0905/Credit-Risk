import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("data/index")


def load_index_and_docs():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    docs = json.loads((INDEX_DIR / "docs.json").read_text(encoding="utf-8"))
    return index, docs


def search(query: str, k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    index, docs = load_index_and_docs()
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({"score": float(score), **docs[int(idx)]})
    return results


if __name__ == "__main__":
    query = "Rejected applicant with utilization 82% and short credit history"
    res = search(query, k=5)
    for r in res:
        print(f"[{r['score']:.3f}] {r['id']} ({r['type']}): {r['text']}")
