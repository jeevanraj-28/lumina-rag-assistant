# Retrieval evaluation (tfidf)

Run on 2026-10-03 with 3 documents and 20 questions.

| chunk size | overlap | chunks | recall@1 | recall@3 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 0 | 37 | 0.40 | 0.70 | 0.80 | 0.556 |
| 300 | 150 | 86 | 0.50 | 0.75 | 0.90 | 0.643 |
| 600 | 0 | 16 | 0.50 | 0.80 | 0.85 | 0.635 |
| 600 | 150 | 24 | 0.55 | 0.75 | 0.80 | 0.660 |
| 900 | 0 | 11 | 0.45 | 0.80 | 0.85 | 0.629 |
| 900 | 150 | 14 | 0.60 | 0.85 | 0.85 | 0.717 |

Best: chunk size 900, overlap 150.
