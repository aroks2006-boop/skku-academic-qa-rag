import csv
import json
import random
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent
OUT = EVAL_DIR / "human_grading.jsonl"
PER_VERDICT = {"correct": 12, "partial": 7, "incorrect": 7, "abstain": 4}


def main() -> None:
    if OUT.exists():
        print(f"{OUT} 가 이미 있습니다. 다시 뽑으려면 파일을 지우세요.")
        return
    questions = {}
    for name in ("questions.jsonl", "heldout.jsonl"):
        for line in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                questions[q["id"]] = q

    pools = defaultdict(list)
    for name in ("results.csv", "results_heldout.csv"):
        with (EVAL_DIR / name).open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["run"] == "1" and row["answerable"] == "True":
                    pools[row["verdict"]].append(row)

    rng = random.Random(42)
    picked = []
    for verdict, n in PER_VERDICT.items():
        picked.extend(rng.sample(pools[verdict], min(n, len(pools[verdict]))))
    rng.shuffle(picked)

    lines = []
    for i, row in enumerate(picked, 1):
        q = questions[row["id"]]
        lines.append(json.dumps({
            "sample": f"s{i:02d}", "id": row["id"], "system": row["system"], "question": q["question"],
            "gold": q["gold"], "answer": row["answer"], "llm_verdict": row["verdict"], "human_verdict": None,
        }, ensure_ascii=False))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)}개 표본 → {OUT}")


if __name__ == "__main__":
    main()
