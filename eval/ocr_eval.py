import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

import ocr
from config import RAW_DIR

EVAL_DIR = Path(__file__).parent
CACHE_DIR = EVAL_DIR / "answers" / "ocr"
PDF = RAW_DIR / "2026 학사제도 안내.pdf"

KEY_CELLS = {
    11: [("7학기이상", "2.13"), ("5~6학기", "2.19"), ("3~4학기", "14:00"), ("1~2학기", "2.20"), ("5~6학기", "8.14"), ("1~2학기", "8.18"), ("1학기", "2.3.(화)")],
    16: [("0~3학점", "1/6"), ("4~6학점", "1/3"), ("7~9학점", "1/2"), ("10학점이상", "전액"), ("2학점", "1/3"), ("3학점", "1/2")],
    27: [("1학년", "35학점"), ("2학년", "81학점"), ("3학년", "98학점"), ("3학년", "105학점"), ("3학년", "106학점"), ("4학년", "140학점"), ("4학년", "173학점"), ("5학년", "189학점"), ("6학년", "230학점")],
    49: [("137", "185"), ("112", "154"), ("154", "219"), ("바이오메카트로닉스", "38/24"), ("바이오메카트로닉스", "60/42"), ("식품생명", "36/"), ("스포츠과학", "42/24"), ("의예과", "81"), ("글로벌바이오메디컬", "64/43"), ("예시3", "63/45"), ("전공코어",)],
    72: [("15학점", "5학기"), ("20학점", "1~4학기"), ("27학점",), ("24학점",), ("18학점", "5학기")],
    96: [("현장실습3", "4학점"), ("현장실습4", "5학점"), ("12주", "18학점"), ("현장실습1,2,3", "1학점")],
    107: [("40학점", "55학점", "70학점", "85학점", "100학점", "115학점")],
    117: [("단일", "39", "15", "54"), ("복수", "36", "3", "39"), ("수료", "120")],
    125: [("기말", "6.29"), ("기말", "12.24")],
    143: [("500원", "750원"), ("700원",), ("1,000원",)],
    12: [("전체강좌", "3.3"), ("수강신청불가", "3.4"), ("폐반확정일", "3.9"), ("전체강좌", "8.31"), ("1차증원", "2.5"), ("2차증원", "8.13")],
}


def normalize(text: str) -> list[str]:
    return [line.replace(" ", "") for line in text.splitlines()]


def score(text: str, cells: list[tuple]) -> int:
    lines = normalize(text)
    return sum(any(all(t.replace(" ", "") in line for t in cell) for line in lines) for cell in cells)


def transcribe(model: str, page: int) -> tuple[str, float, str]:
    ocr.OCR_MODEL = model
    start = time.perf_counter()
    try:
        text = ocr.transcribe(OpenAI(), ocr.render_png(PDF, page))
        status = "ok"
    except ocr.Truncated:
        text, status = "", "거부/출력과다"
    return text, time.perf_counter() - start, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gpt-4o-mini,gpt-4.1,gpt-5-mini")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models.split(","):
        path = CACHE_DIR / f"{model}.json"
        cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        for page, cells in KEY_CELLS.items():
            if str(page) not in cache:
                text, latency, status = transcribe(model, page)
                cache[str(page)] = {"text": text, "latency": latency, "status": status}
                path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[{model}] p.{page} {status}")
            item = cache[str(page)]
            rows.append({"model": model, "page": page, "status": item["status"], "hit": score(item["text"], cells),
                         "total": len(cells), "latency": round(item["latency"], 1)})

    with (EVAL_DIR / "ocr_eval.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(len(c) for c in KEY_CELLS.values())
    print(f"\n{'model':<14}{'핵심 칸 정확도':>14}{'실패 페이지':>12}{'페이지당 시간(s)':>18}")
    for model in args.models.split(","):
        items = [r for r in rows if r["model"] == model]
        hits = sum(r["hit"] for r in items)
        fails = sum(r["status"] != "ok" for r in items)
        latency = sum(r["latency"] for r in items) / len(items)
        print(f"{model:<14}{hits / total:>13.1%} ({hits}/{total}){fails:>6}{latency:>16.1f}")
    print("\n페이지별")
    for page in KEY_CELLS:
        print(f"  p.{page:<4}" + "  ".join(f"{r['model']}:{r['hit']}/{r['total']}" for r in rows if r["page"] == page))


if __name__ == "__main__":
    main()
