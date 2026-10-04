"""The UI must show the real model name, never a rebranded one."""
import ast
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def load(name):
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace[name]


brand_model_name = load("brand_model_name")


class ModelNameTests(unittest.TestCase):
    def test_qwen_keeps_its_name(self):
        self.assertEqual(brand_model_name("qwen2.5:3b"), "Qwen 2.5 3B")
        self.assertEqual(brand_model_name("qwen3:8b"), "Qwen 3 8B")

    def test_other_families(self):
        self.assertEqual(brand_model_name("deepseek-r1:7b"), "DeepSeek R1 7B")
        self.assertEqual(brand_model_name("phi4-mini"), "Phi-4 Mini")

    def test_answers_are_not_rewritten(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertNotIn("sanitize_response_text", source)
        self.assertNotIn("Never mention", source)


if __name__ == "__main__":
    unittest.main()
