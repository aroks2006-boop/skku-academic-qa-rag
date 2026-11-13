import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import retrieve

EVAL_DIR = Path(__file__).parent


def first_hit(hits: list[dict], case: dict) -> int:
    for rank, h in enumerate(hits, 1):
        if h["source"] == case["source"] and h["page"] in case["pages"]:
            return rank
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=EVAL_DIR / "retrieval_cases.jsonl", type=Path)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--modes", default="vector,hybrid,rerank")
    args = parser.parse_args()

    lines = args.cases.read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    modes = args.modes.split(",")

    table = {m: [first_hit(retrieve(c["question"], 20, m), c) for c in cases] for m in modes}

    print("".join(f"{m:>8}" for m in modes) + "  question")
    for i, case in enumerate(cases):
        print("".join(f"{table[m][i] or '-':>8}" for m in modes) + f"  {case['question']}")

    print()
    for m in modes:
        ranks = table[m]
        n = len(ranks)
        hit_k = sum(0 < r <= args.k for r in ranks) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{m:>8}  Hit@{args.k}: {hit_k:.0%}   MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()
