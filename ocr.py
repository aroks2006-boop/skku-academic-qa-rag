import argparse
import base64
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from openai import OpenAI

from config import OCR_DIR, OCR_DPI, OCR_MIN_CHARS, OCR_MODEL, OCR_WORKERS, RAW_DIR

OCR_PROMPT = """이 이미지는 성균관대학교 학사 안내 문서의 한 페이지입니다.
페이지의 모든 글자를 빠짐없이 원문 그대로 옮겨 적으세요.
- 표는 Markdown 표로 옮기고, 병합된 칸은 해당 값을 각 행에 반복해 적습니다.
- 제목·소제목은 # 으로 구분합니다.
- 숫자, 학점, 학번, 날짜는 정확히 옮기고 추측해서 채우지 않습니다.
- 설명이나 요약을 덧붙이지 말고 옮긴 텍스트만 출력합니다.
- 표 구분선은 열 개수만큼만 | --- | 로 한 번 적고, 빈 칸을 반복해서 채우지 않습니다.
- 이미지가 페이지 일부라면 보이는 부분만 옮깁니다.
- 글자가 없는 페이지면 빈 문자열을 출력합니다."""

MAX_TOKENS = 20000
MAX_CHARS = 12000
SPLIT_PARTS = 3
SPLIT_DPI = 250


REFUSAL = re.compile(r"(고해상도|선명한|확대).{0,60}(이미지|보내|제공)|판독할 수 없")


class Truncated(Exception):
    pass


def cache_path(pdf: Path, page_no: int) -> Path:
    return OCR_DIR / pdf.stem / f"p{page_no:04d}.md"


def render_png(pdf: Path, page_no: int) -> bytes:
    with pymupdf.open(pdf) as doc:
        return doc[page_no - 1].get_pixmap(dpi=OCR_DPI).tobytes("png")


def render_parts(pdf: Path, page_no: int, parts: int = SPLIT_PARTS) -> list[bytes]:
    with pymupdf.open(pdf) as doc:
        page = doc[page_no - 1]
        r = page.rect
        step = r.height / parts
        pad = step * 0.08
        clips = [pymupdf.Rect(r.x0, max(r.y0, r.y0 + i * step - pad), r.x1, min(r.y1, r.y0 + (i + 1) * step + pad)) for i in range(parts)]
        return [page.get_pixmap(dpi=SPLIT_DPI, clip=c).tobytes("png") for c in clips]


def transcribe(client: OpenAI, png: bytes) -> str:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    reasoning = OCR_MODEL.startswith(("gpt-5", "o"))
    sampling = {} if reasoning else {"temperature": 0}
    res = client.chat.completions.create(
        model=OCR_MODEL,
        max_completion_tokens=MAX_TOKENS if reasoning else min(MAX_TOKENS, 16000),
        **sampling,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
            ],
        }],
    )
    choice = res.choices[0]
    text = (choice.message.content or "").strip()
    if choice.finish_reason == "length" or len(text) > MAX_CHARS or REFUSAL.search(text[:300]):
        raise Truncated
    return text


def transcribe_page(client: OpenAI, pdf: Path, page_no: int) -> str:
    try:
        return transcribe(client, render_png(pdf, page_no))
    except Truncated:
        print(f"OCR {pdf.name} p.{page_no} 출력 과다 → {SPLIT_PARTS}분할 재시도")
    texts = []
    for i, png in enumerate(render_parts(pdf, page_no), 1):
        try:
            texts.append(transcribe(client, png))
        except Truncated:
            print(f"OCR {pdf.name} p.{page_no} 분할 {i} 실패, 건너뜀")
    return "\n\n".join(texts)


def ocr_page(client: OpenAI, pdf: Path, page_no: int) -> str:
    path = cache_path(pdf, page_no)
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = transcribe_page(client, pdf, page_no)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"OCR {pdf.name} p.{page_no}")
    return text


def pages_needing_ocr(pdf: Path) -> list[int]:
    with pymupdf.open(pdf) as doc:
        return [i + 1 for i, page in enumerate(doc) if len(page.get_text().strip()) < OCR_MIN_CHARS]


def ocr_pages(pdf: Path, page_nos: list[int]) -> dict[int, str]:
    client = OpenAI()
    with ThreadPoolExecutor(OCR_WORKERS) as pool:
        texts = pool.map(lambda n: ocr_page(client, pdf, n), page_nos)
    return dict(zip(page_nos, texts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", help="예: 1-5 또는 3,10,20 (생략 시 OCR 필요한 전체 페이지)")
    args = parser.parse_args()

    for pdf in sorted(RAW_DIR.rglob("*.pdf")):
        targets = pages_needing_ocr(pdf)
        if args.pages:
            wanted = set()
            for part in args.pages.split(","):
                lo, _, hi = part.partition("-")
                wanted.update(range(int(lo), int(hi or lo) + 1))
            targets = [n for n in targets if n in wanted]
        if targets:
            ocr_pages(pdf, targets)
            print(f"{pdf.name}: {len(targets)}페이지 OCR 완료 → {OCR_DIR / pdf.stem}")


if __name__ == "__main__":
    main()
