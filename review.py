import json
import re
from pathlib import Path

import pymupdf
import streamlit as st

from config import RAW_DIR

QUESTIONS = Path(__file__).parent / "eval" / "questions.jsonl"


def load() -> list[dict]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def save(questions: list[dict]) -> None:
    text = "\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n"
    QUESTIONS.write_text(text, encoding="utf-8")


def parse_source(source: str) -> tuple[str, list[int]]:
    name = source.split(" p.")[0]
    pages = []
    for lo, hi in re.findall(r"p\.(\d+)(?:-(\d+))?", source):
        pages.extend(range(int(lo), int(hi or lo) + 1))
    return name, sorted(set(pages))


@st.cache_data
def render(name: str, page: int) -> bytes:
    with pymupdf.open(RAW_DIR / name) as doc:
        return doc[page - 1].get_pixmap(dpi=110).tobytes("png")


st.set_page_config(page_title="정답셋 검수", layout="wide")
questions = load()
done = sum(q.get("verified") is True for q in questions)
st.sidebar.progress(done / len(questions), text=f"검수 완료 {done}/{len(questions)}")
only_todo = st.sidebar.checkbox("미검수만 보기", value=True)
pool = [q for q in questions if not (only_todo and q.get("verified"))]
if not pool:
    st.success("모든 문항 검수 완료")
    st.stop()

labels = [f"{q['id']} {'✅' if q.get('verified') else '⬜'} {q['question'][:30]}" for q in pool]
picked = st.sidebar.radio("문항", range(len(pool)), format_func=lambda i: labels[i])
q = pool[picked]

left, right = st.columns([2, 3])
with left:
    st.subheader(f"{q['id']} · {q['category']}")
    st.markdown(f"**질문** {q['question']}")
    st.caption(f"출처: {q['source']}")
    gold = st.text_area("정답 (원본과 다르면 고치기)", q["gold"], height=160)
    keywords = st.text_input("필수 키워드 (쉼표 구분)", ", ".join(q.get("keywords", [])))
    keep, drop = st.columns(2)
    if keep.button("맞음 · 저장", type="primary", use_container_width=True):
        q["gold"] = gold.strip()
        q["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
        q["verified"] = True
        save(questions)
        st.rerun()
    if drop.button("문항 삭제", use_container_width=True):
        save([x for x in questions if x["id"] != q["id"]])
        st.rerun()

with right:
    name, pages = parse_source(q["source"])
    if not q.get("answerable", True):
        st.info("문서에 답이 없어야 하는 문항입니다. 문서에 정말 없는지만 확인하세요.")
    elif (RAW_DIR / name).exists() and pages:
        for page in pages:
            st.image(render(name, page), caption=f"{name} PDF p.{page}")
    else:
        st.warning("원본 PDF를 찾을 수 없습니다. data/raw/ 를 확인하세요.")
