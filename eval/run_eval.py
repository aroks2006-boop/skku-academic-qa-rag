import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import product
from statistics import mean, pstdev
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import JUDGE_MODEL
from rag import answer, answer_plain, chat

EVAL_DIR = Path(__file__).parent
ANSWER_DIR = EVAL_DIR / "answers"
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "abstain": 0.0}

JUDGE_SYSTEM = """당신은 대학 학사 안내 답변의 채점자입니다.
[정답]과 [필수 키워드]를 기준으로 [답변]을 채점하고 JSON으로만 출력하세요.
- correct: 정답의 핵심 사실(숫자, 조건, 대상)이 모두 맞고 틀린 내용이 없음
- partial: 핵심 일부만 맞거나, 맞지만 조건 구분이 빠짐
- incorrect: 핵심 사실이 틀리거나 정답과 모순됨
- abstain: 답을 모른다고 하거나, 문서에서 확인할 수 없다고 하거나, 답변을 회피함
형식: {"verdict": "correct|partial|incorrect|abstain", "reason": "한 문장"}"""


def load_questions(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_kingobot(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    rows = csv.DictReader(text.splitlines())
    return {row["id"]: row["answer"].strip() for row in rows if (row.get("answer") or "").strip()}


def generate(system: str, question: str) -> str:
    if system == "rag":
        return answer(question)["answer"]
    return answer_plain(question)


def collect_answers(system: str, questions: list[dict], kingobot: dict[str, str]) -> dict[str, str]:
    if system == "kingobot":
        return kingobot
    ANSWER_DIR.mkdir(exist_ok=True)
    cache_path = ANSWER_DIR / f"{system}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for q in questions:
        if q["id"] not in cache:
            cache[q["id"]] = generate(system, q["question"])
            print(f"[{system}] {q['id']} 답변 생성")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    normalized = text.replace(" ", "")
    return sum(k.replace(" ", "") in normalized for k in keywords) / len(keywords)


def judge(q: dict, text: str) -> dict:
    prompt = (
        f"[질문]\n{q['question']}\n\n[정답]\n{q['gold']}\n\n"
        f"[필수 키워드]\n{', '.join(q.get('keywords', []))}\n\n[답변]\n{text}"
    )
    raw = chat(JUDGE_SYSTEM, prompt, JUDGE_MODEL)
    start, end = raw.find("{"), raw.rfind("}") + 1
    result = json.loads(raw[start:end])
    if result.get("verdict") not in SCORES:
        result["verdict"] = "incorrect"
    if not q.get("answerable", True):
        result["verdict"] = "correct" if result["verdict"] in ("abstain", "correct") else "incorrect"
    return result


def summarize(rows: list[dict], systems: list[str]) -> None:
    by_system = defaultdict(list)
    by_category = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
        by_category[(r["system"], r["category"])].append(r)

    print(f"\n{'system':<10}{'n':>4}{'정답률':>8}{'부분포함':>10}{'근거있음 정답률':>14}{'키워드':>8}{'회피':>6}")
    for s in systems:
        items = by_system[s]
        if not items:
            continue
        n = len(items)
        answerable = [r for r in items if r["answerable"]] or items
        strict = sum(r["verdict"] == "correct" for r in items) / n
        lenient = sum(r["score"] for r in items) / n
        strict_ans = sum(r["verdict"] == "correct" for r in answerable) / len(answerable)
        kw = sum(r["keyword_recall"] for r in answerable) / len(answerable)
        abstain = sum(r["verdict"] == "abstain" for r in answerable) / len(answerable)
        print(f"{s:<10}{n:>4}{strict:>8.1%}{lenient:>10.1%}{strict_ans:>14.1%}{kw:>8.1%}{abstain:>6.0%}")

    print("\n카테고리별 부분포함 점수")
    for (s, cat), items in sorted(by_category.items()):
        print(f"  {s:<10}{cat:<16}{sum(r['score'] for r in items) / len(items):.1%} (n={len(items)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=EVAL_DIR / "questions.jsonl", type=Path)
    parser.add_argument("--kingobot", default=EVAL_DIR / "kingobot_answers.csv", type=Path)
    parser.add_argument("--systems", default="rag,plain,kingobot")
    parser.add_argument("--out", default=EVAL_DIR / "results.csv", type=Path)
    parser.add_argument("--repeat", type=int, default=1, help="채점 반복 횟수 (답변은 캐시 재사용)")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    kingobot = load_kingobot(args.kingobot)
    systems = [s.strip() for s in args.systems.split(",") if s.strip() != "kingobot" or kingobot]
    answers_by_system = {s: collect_answers(s, questions, kingobot) for s in systems}

    rows = []
    for run, system in product(range(1, args.repeat + 1), systems):
        answers = answers_by_system[system]
        for q in questions:
            if q["id"] not in answers:
                continue
            text = answers[q["id"]]
            graded = judge(q, text)
            rows.append({
                "run": run,
                "system": system,
                "id": q["id"],
                "category": q.get("category", "기타"),
                "answerable": q.get("answerable", True),
                "question": q["question"],
                "answer": text,
                "verdict": graded["verdict"],
                "score": SCORES[graded["verdict"]],
                "keyword_recall": keyword_recall(text, q.get("keywords", [])),
                "reason": graded.get("reason", ""),
            })

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows, systems)
    if args.repeat > 1:
        summarize_runs(rows, systems, args.repeat)
    print(f"\n상세 결과: {args.out}")


def summarize_runs(rows: list[dict], systems: list[str], repeat: int) -> None:
    print(f"\n채점 {repeat}회 반복: 정답률 평균 ± 표준편차 (최소~최대)")
    for s in systems:
        accs = []
        for run in range(1, repeat + 1):
            items = [r for r in rows if r["system"] == s and r["run"] == run]
            accs.append(sum(r["verdict"] == "correct" for r in items) / len(items))
        print(f"  {s:<10}{mean(accs):.1%} ± {pstdev(accs):.1%} ({min(accs):.1%}~{max(accs):.1%})")


if __name__ == "__main__":
    main()
