import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("artifacts/decision_cases.csv")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_docs(path: Path):
    df = pd.read_csv(path)

    docs = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Prefer explanation_seed if you created it in main.ipynb
        if "explanation_seed" in df.columns:
            text = str(row["explanation_seed"])
        else:
            text = (
                f"Case ID: {row_dict.get('case_id')}. "
                f"Decision: {row_dict.get('model_decision')}. "
                f"Predicted default probability: {row_dict.get('default_prob')}. "
                f"Decision score: {row_dict.get('decision_score')}. "
                f"Profit threshold: {row_dict.get('profit_threshold')}. "
                f"Default 0.5 decision: {row_dict.get('default_05_decision')}."
            )

        docs.append({
            "case_id": row_dict.get("case_id", f"case_{len(docs):04d}"),
            "text": text,
            "raw": row_dict
        })

    return docs


def main():
    docs = load_docs(INPUT_PATH)
    texts = [d["text"] for d in docs]

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    emb = np.asarray(emb, dtype="float32")
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    (OUT_DIR / "docs.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Built index with {len(docs)} docs, dim={dim}")
    print(f"Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()