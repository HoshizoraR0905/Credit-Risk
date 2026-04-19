import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/docs.jsonl")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_docs(path: Path):
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def main():
    docs = load_docs(DATA_PATH)
    texts = [d["text"] for d in docs]

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    emb = np.asarray(emb, dtype="float32")
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity when embeddings are normalized
    index.add(emb)

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    (OUT_DIR / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built index with {len(docs)} docs, dim={dim}")
    print(f"Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()