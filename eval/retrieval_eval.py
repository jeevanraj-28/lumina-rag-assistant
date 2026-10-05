"""Retrieval evaluation for Lumina RAG.

Measures how often the chunk that contains the answer is retrieved, for a grid of
chunk sizes, overlaps and top-k values. It uses the same `split_text` chunker as
the app and the same search as the app (cosine similarity on L2-normalised
vectors, which is what FAISS IndexFlatIP computes).

Metrics
  recall@k  share of questions where a retrieved chunk from the right file
            contains the answer phrase
  MRR       mean of 1/rank of the first correct chunk (0 if not retrieved)

Usage (from the repository root)
  # Real embeddings - needs Ollama running and `ollama pull all-minilm`
  python eval/retrieval_eval.py --embedder ollama

  # Keyword baseline - needs only numpy, useful to compare against
  python eval/retrieval_eval.py --embedder tfidf

  # Your own documents and questions
  python eval/retrieval_eval.py --docs path/to/docs --questions path/to/questions.jsonl

Each line of the questions file is JSON with:
  question, file (document file name), answer_phrase (exact text from that file)
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"


def load_split_text():
    """Load the app's chunker without importing the whole FastAPI app."""
    source_path = ROOT / "app" / "main.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "split_text")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["split_text"]


split_text = load_split_text()


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- embedders
class TfidfEmbedder:
    """Keyword baseline in plain numpy. Fitted on the chunks of each run."""

    name = "tfidf"

    @staticmethod
    def tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, texts: list[str]) -> None:
        docs = [set(self.tokens(t)) for t in texts]
        df = Counter(term for doc in docs for term in doc)
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)], dtype="float32")

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocab)), dtype="float32")
        for row, text in enumerate(texts):
            for term, count in Counter(self.tokens(text)).items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = count
        return l2_normalise(matrix * self.idf)


class OllamaEmbedder:
    """The embedding model the app uses, called through Ollama's /api/embed."""

    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/"), model
        self.name = f"ollama:{model}"

    def fit(self, texts: list[str]) -> None:  # nothing to fit
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests

        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            response = requests.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": texts[start:start + 32]},
                timeout=300,
            )
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return l2_normalise(np.asarray(vectors, dtype="float32"))


# ---------------------------------------------------------------- evaluation
def load_docs(folder: Path) -> dict[str, str]:
    docs = {p.name: p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(folder.iterdir()) if p.suffix.lower() in {".md", ".txt"}}
    if not docs:
        raise SystemExit(f"No .md or .txt files found in {folder}")
    return docs


def load_questions(path: Path, docs: dict[str, str]) -> list[dict]:
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for q in questions:
        if q["file"] not in docs:
            raise SystemExit(f"{q['id']}: file {q['file']} not in docs folder")
        if normalise(q["answer_phrase"]) not in normalise(docs[q["file"]]):
            raise SystemExit(f"{q['id']}: answer_phrase not found in {q['file']}")
    return questions


def evaluate(docs, questions, embedder, chunk_size, overlap, ks):
    chunks, owners = [], []
    for name, text in docs.items():
        for chunk in split_text(text, chunk_size, overlap):
            chunks.append(chunk)
            owners.append(name)
    embedder.fit(chunks)
    chunk_vectors = embedder.embed(chunks)
    question_vectors = embedder.embed([q["question"] for q in questions])
    scores = question_vectors @ chunk_vectors.T  # cosine similarity
    max_k = max(ks)
    ranks, misses = [], []
    for qi, q in enumerate(questions):
        order = np.argsort(-scores[qi])[:max_k]
        phrase = normalise(q["answer_phrase"])
        rank = next((r for r, ci in enumerate(order, start=1)
                     if owners[ci] == q["file"] and phrase in normalise(chunks[ci])), None)
        ranks.append(rank)
        if rank is None or rank > min(ks):
            misses.append({"id": q["id"], "question": q["question"], "rank": rank,
                           "top_chunk": chunks[order[0]][:200]})
    n = len(questions)
    result = {"chunk_size": chunk_size, "overlap": overlap, "n_chunks": len(chunks),
              "mrr": sum(1 / r for r in ranks if r) / n}
    for k in ks:
        result[f"recall@{k}"] = sum(1 for r in ranks if r and r <= k) / n
    return result, misses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=["ollama", "tfidf"], default="ollama")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="all-minilm")
    parser.add_argument("--docs", type=Path, default=EVAL_DIR / "sample_docs")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 600, 900])
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 150])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--out", type=Path, default=None, help="Markdown report path")
    args = parser.parse_args()

    docs = load_docs(args.docs)
    questions = load_questions(args.questions, docs)
    embedder = OllamaEmbedder(args.ollama_url, args.model) if args.embedder == "ollama" else TfidfEmbedder()

    rows, all_misses = [], {}
    for size in args.chunk_sizes:
        for overlap in args.overlaps:
            if overlap >= size:
                continue
            row, misses = evaluate(docs, questions, embedder, size, overlap, args.ks)
            rows.append(row)
            all_misses[f"{size}/{overlap}"] = misses
            print(f"chunk={size:<5} overlap={overlap:<4} " +
                  " ".join(f"recall@{k}={row[f'recall@{k}']:.2f}" for k in args.ks) +
                  f" MRR={row['mrr']:.3f}")

    best = max(rows, key=lambda r: (r[f"recall@{sorted(args.ks)[len(args.ks) // 2]}"], r["mrr"]))
    header = ["chunk size", "overlap", "chunks"] + [f"recall@{k}" for k in args.ks] + ["MRR"]
    lines = [
        f"# Retrieval evaluation ({embedder.name})",
        "",
        f"Run on {date.today().isoformat()} with {len(docs)} documents and {len(questions)} questions.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in rows:
        cells = [r["chunk_size"], r["overlap"], r["n_chunks"]] + [f"{r[f'recall@{k}']:.2f}" for k in args.ks] + [f"{r['mrr']:.3f}"]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += ["", f"Best: chunk size {best['chunk_size']}, overlap {best['overlap']}.", ""]
    out = args.out or EVAL_DIR / f"results_{args.embedder}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (out.with_suffix(".misses.json")).write_text(json.dumps(all_misses, indent=2), encoding="utf-8")
    print(f"\nReport written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
