"""Tests for the text chunker used before embedding.

Run from the repository root:
    python -m unittest discover -s tests -v

The test loads `split_text` straight from app/main.py's source, so it runs
without Ollama, FAISS or FastAPI installed.
"""
import ast
import re
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def load_split_text():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "split_text")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace["split_text"]


split_text = load_split_text()


def make_text(n_words: int = 3000) -> tuple[list[str], str]:
    words = [f"w{i}" for i in range(n_words)]
    paragraphs = []
    for start in range(0, n_words, 120):
        sentences = [" ".join(words[j:j + 12]) for j in range(start, min(start + 120, n_words), 12)]
        paragraphs.append(". ".join(sentences))
    return words, "\n\n".join(paragraphs)


class ChunkingTests(unittest.TestCase):
    CONFIGS = [(900, 150), (600, 0), (300, 100), (1200, 200), (200, 50)]

    def test_no_text_is_lost(self):
        """Every word in the document must appear in at least one chunk."""
        words, text = make_text()
        for size, overlap in self.CONFIGS:
            with self.subTest(size=size, overlap=overlap):
                joined = " ".join(split_text(text, size, overlap))
                missing = [w for w in words if not re.search(rf"\b{w}\b", joined)]
                self.assertEqual(missing, [], f"{len(missing)} words never indexed")

    def test_chunks_respect_size_limit(self):
        _, text = make_text()
        for size, overlap in self.CONFIGS:
            with self.subTest(size=size, overlap=overlap):
                chunks = split_text(text, size, overlap)
                self.assertLessEqual(max(len(c) for c in chunks), size)

    def test_overlap_is_carried_forward(self):
        _, text = make_text()
        chunks = split_text(text, 900, 150)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertTrue(current.startswith(previous[-150:]))

    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("A short note.", 900, 150), ["A short note."])

    def test_unbroken_text_is_hard_split(self):
        text = "x" * 2500
        chunks = split_text(text, 900, 0)
        self.assertEqual("".join(chunks), text)


if __name__ == "__main__":
    unittest.main()
