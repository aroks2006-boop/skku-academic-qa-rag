import hashlib
import re
from pathlib import Path

import chromadb
import pymupdf
from openai import OpenAI

from config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, EMBED_MODEL, OCR_MIN_CHARS, RAW_DIR
from ocr import ocr_pages

EMBED_BATCH = 100
CHAPTER = re.compile(r"\s*0?\d{1,2}$")
NUMBERED = re.compile(r"^\d{1,2}\.\s+\S")
TABLE_TITLE = re.compile(r"학번|기준표|이수표")
UNIT = re.compile(r"(대학|학부|학과|계열|융합원)$")
ARTICLE = re.compile(r"^제\s*\d+\s*조(의\s*\d+)?\s*(\([^)]{1,40}\))?")


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-{4,}", "---", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            pages = {i + 1: clean(page.get_text()) for i, page in enumerate(doc)}
        missing = [n for n, text in pages.items() if len(text) < OCR_MIN_CHARS]
        if missing:
            print(f"{path.name}: 텍스트 없는 {len(missing)}페이지 OCR")
            pages.update({n: clean(t) for n, t in ocr_pages(path, missing).items()})
        return sorted(pages.items())
    return [(1, clean(path.read_text(encoding="utf-8")))]


def split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step) if text[i : i + size].strip()]


def is_chapter(heading: str) -> bool:
    return bool(CHAPTER.search(heading))


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?(\s*:?-{3,}:?\s*\|)+\s*:?-*:?\s*", line))


def blocks(text: str, state: dict) -> list[tuple[str, str, list[str]]]:
    result = []
    kind, lines = "", []

    def flush() -> None:
        if lines:
            result.append((kind, state["title"], list(lines)))
            lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            flush()
            heading = stripped.lstrip("#").strip()
            if heading.startswith("Inspiring Future"):
                continue
            if NUMBERED.match(heading):
                state["chapter"] = heading
                state["table"] = ""
                state["title"] = heading
            elif is_chapter(heading):
                chapter = re.sub(r"[^\w\s·()~/,-]", "", CHAPTER.sub("", heading)).strip()
                state["chapter"] = chapter or state["chapter"]
            elif UNIT.search(heading) and TABLE_TITLE.search(state["table"]):
                state["title"] = f"{state['table']} > {heading}"
            else:
                state["table"] = heading if TABLE_TITLE.search(heading) else ""
                state["title"] = heading
            continue
        article = ARTICLE.match(stripped)
        if article:
            flush()
            state["table"] = ""
            state["title"] = article.group(0)
        current = "table" if stripped.startswith("|") else "text"
        if current != kind:
            flush()
            kind = current
        lines.append(stripped)
    flush()
    return result


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def linearize(header: list[str], row: str) -> str:
    pairs = []
    for i, value in enumerate(cells(row)):
        if not value:
            continue
        name = header[i] if i < len(header) else ""
        pairs.append(f"{name}: {value}" if name else value)
    if not pairs:
        return ""
    return "\n".join([f"- {pairs[0]}"] + [f"  · {p}" for p in pairs[1:]])


def table_units(lines: list[str], size: int) -> list[str]:
    has_header = len(lines) > 1 and is_separator(lines[1])
    header = cells(lines[0]) if has_header else []
    body = [r for r in lines[(2 if has_header else 0):] if not is_separator(r)]
    rows = [r for r in (linearize(header, b) for b in body) if r] if has_header else body
    if has_header and not rows:
        rows = ["- " + " | ".join(h for h in header if h)]
    units, group = [], []
    for row in rows:
        if group and len("\n".join(group + [row])) > size:
            units.append("\n".join(group))
            group = []
        group.append(row)
    if group:
        units.append("\n".join(group))
    return units


def page_chunks(text: str, state: dict, size: int = CHUNK_SIZE) -> list[tuple[str, str]]:
    units = []
    for kind, title, lines in blocks(text, state):
        body = table_units(lines, size) if kind == "table" else split("\n".join(lines), size)
        units.extend((title, u) for u in body)

    chunks, cur_title, cur = [], None, []
    for title, unit in units:
        if cur and (title != cur_title or len("\n".join(cur + [unit])) > size):
            chunks.append((cur_title, "\n".join(cur)))
            cur = []
        cur_title = title
        cur.append(unit)
    if cur:
        chunks.append((cur_title, "\n".join(cur)))
    return chunks


def context_header(path: Path, page: int, chapter: str, title: str) -> str:
    parts = [p for p in (chapter, title) if p]
    return f"[{path.stem} p.{page}] " + " > ".join(parts)


def build_chunks(raw_dir: Path) -> list[dict]:
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"})
    chunks = []
    for path in files:
        state = {"chapter": "", "title": "", "table": ""}
        for page, text in load_pages(path):
            for n, (title, body) in enumerate(page_chunks(text, state)):
                uid = hashlib.md5(f"{path.name}:{page}:{n}".encode()).hexdigest()
                text_with_header = f"{context_header(path, page, state['chapter'], title)}\n{body}"
                chunks.append({"id": uid, "text": text_with_header, "source": path.name, "page": page})
    return chunks


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        res = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        vectors.extend(d.embedding for d in res.data)
    return vectors


def main() -> None:
    chunks = build_chunks(RAW_DIR)
    if not chunks:
        print(f"{RAW_DIR} 에 PDF/TXT/MD 문서를 넣어주세요.")
        return

    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [c.name for c in db.list_collections()]:
        db.delete_collection(COLLECTION)
    col = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    texts = [c["text"] for c in chunks]
    col.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embed(OpenAI(), texts),
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    print(f"{len({c['source'] for c in chunks})}개 문서, {len(chunks)}개 청크 인덱싱 완료")


if __name__ == "__main__":
    main()
