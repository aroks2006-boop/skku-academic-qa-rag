import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

import rag
from config import TOP_K
from run_eval import SCORES, judge, load_questions

EVAL_DIR = Path(__file__).parent
CACHE_DIR = EVAL_DIR / "answers" / "ablation"
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
}
RULE_MARKERS = ("표에서는", "답에 쓴 숫자가", "여러 조각이")
BASE_PROMPT = "\n".join(l for l in rag.RAG_SYSTEM.splitlines() if not any(m in l for m in RULE_MARKERS))

CONFIGS = [
    ("1_vector_md_4o", "vector", "skku_md", "gpt-4o-mini", "base"),
    ("2_hybrid_md_4o", "hybrid", "skku_md", "gpt-4o-mini", "base"),
    ("3_rerank_md_4o", "rerank", "skku_md", "gpt-4o-mini", "base"),
    ("4_rerank_linear_4o", "rerank", "skku_academic", "gpt-4o-mini", "base"),
    ("5_rerank_linear_41", "rerank", "skku_academic", "gpt-4.1-mini", "base"),
    ("6_final_rules_41", "rerank", "skku_academic", "gpt-4.1-mini", "rules"),
]


class Meter:
    def __init__(self) -> None:
        self.tokens = Counter()
        self.client = OpenAI()
        chat_create = self.client.chat.completions.create
        embed_create = self.client.embeddings.create

        def metered_chat(**kwargs):
            res = chat_create(**kwargs)
            self.tokens[(kwargs["model"], "in")] += res.usage.prompt_tokens
            self.tokens[(kwargs["model"], "out")] += res.usage.completion_tokens
            return res

        def metered_embed(**kwargs):
            res = embed_create(**kwargs)
            self.tokens[(kwargs["model"], "in")] += res.usage.prompt_tokens
            return res

        self.client.chat.completions.create = metered_chat
        self.client.embeddings.create = metered_embed

    def cost(self) -> float:
        total = 0.0
        for (model, kind), n in self.tokens.items():
            price_in, price_out = PRICES.get(model, (0.0, 0.0))
            total += n * (price_in if kind == "in" else price_out) / 1_000_000
        return total


def use_collection(name: str) -> None:
    rag.COLLECTION = name
    rag.collection.cache_clear()
    rag.corpus.cache_clear()
    rag.bm25.cache_clear()


def generate(name: str, mode: str, collection: str, model: str, prompt: str, questions: list[dict]) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    use_collection(collection)
    rag.CHAT_MODEL = model
    system = BASE_PROMPT if prompt == "base" else rag.RAG_SYSTEM
    for q in questions:
        if q["id"] in cache:
            continue
        meter = Meter()
        rag.client = lambda m=meter: m.client
        start = time.perf_counter()
        hits = rag.retrieve(q["question"], TOP_K, mode)
        user = f"[문서]\n{rag.format_context(hits)}\n\n[질문]\n{q['question']}"
        text = rag.chat(system, user, model)
        cache[q["id"]] = {"answer": text, "latency": time.perf_counter() - start, "cost": meter.cost()}
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}] {q['id']}")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=EVAL_DIR / "questions.jsonl", type=Path)
    parser.add_argument("--out", default=EVAL_DIR / "ablation.csv", type=Path)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    rows = []
    for name, mode, collection, model, prompt in CONFIGS:
        answers = generate(name, mode, collection, model, prompt, questions)
        rag.client = lambda c=OpenAI(): c
        for q in questions:
            item = answers[q["id"]]
            verdict = judge(q, item["answer"])["verdict"]
            rows.append({"config": name, "id": q["id"], "verdict": verdict, "score": SCORES[verdict],
                         "latency": round(item["latency"], 2), "cost": item["cost"]})

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'config':<22}{'정답률':>8}{'부분포함':>10}{'평균지연(s)':>12}{'질문당 비용($)':>16}")
    for name, *_ in CONFIGS:
        items = [r for r in rows if r["config"] == name]
        n = len(items)
        acc = sum(r["verdict"] == "correct" for r in items) / n
        lenient = sum(r["score"] for r in items) / n
        latency = sum(r["latency"] for r in items) / n
        cost = sum(r["cost"] for r in items) / n
        print(f"{name:<22}{acc:>8.1%}{lenient:>10.1%}{latency:>12.2f}{cost:>16.5f}")
    print(f"\n상세 결과: {args.out}")


if __name__ == "__main__":
    main()
