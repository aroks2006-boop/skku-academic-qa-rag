import json
from collections import Counter
from pathlib import Path

import streamlit as st

PATH = Path(__file__).parent / "human_grading.jsonl"
LABELS = {"correct": "정답", "partial": "부분정답", "incorrect": "오답", "abstain": "회피/모름"}
GUIDE = """- **정답**: 핵심 사실(숫자·조건·대상)이 모두 맞고 틀린 내용이 없음
- **부분정답**: 핵심 일부만 맞거나, 맞지만 조건 구분이 빠짐
- **오답**: 핵심 사실이 틀리거나 정답과 모순됨
- **회피/모름**: 모른다·확인할 수 없다고 하거나 답을 피함"""


def load() -> list[dict]:
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(items: list[dict]) -> None:
    PATH.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in LABELS) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def report(items: list[dict]) -> None:
    pairs = [(x["human_verdict"], x["llm_verdict"]) for x in items]
    n = len(pairs)
    exact = sum(a == b for a, b in pairs)
    binary = sum((a == "correct") == (b == "correct") for a, b in pairs)
    st.success(f"채점 완료 {n}개")
    st.metric("4단계 판정 일치율", f"{exact / n:.1%} ({exact}/{n})")
    st.metric("정답 여부 일치율 (정답 vs 나머지)", f"{binary / n:.1%} ({binary}/{n})")
    st.metric("Cohen's kappa (4단계)", f"{kappa(pairs):.2f}")
    st.caption("kappa: 0.6 이상 상당한 일치, 0.8 이상 거의 완전한 일치")
    table = Counter(pairs)
    st.subheader("사람(행) × LLM(열)")
    header = "| 사람 \\ LLM | " + " | ".join(LABELS.values()) + " |"
    rows = [header, "|" + "---|" * (len(LABELS) + 1)]
    for h, hl in LABELS.items():
        rows.append(f"| {hl} | " + " | ".join(str(table[(h, l)]) for l in LABELS) + " |")
    st.markdown("\n".join(rows))
    st.subheader("불일치 사례")
    for x in items:
        if x["human_verdict"] != x["llm_verdict"]:
            st.markdown(f"**{x['sample']} ({x['id']}, {x['system']})** 사람: {LABELS[x['human_verdict']]} / LLM: {LABELS[x['llm_verdict']]}")
            st.caption(x["answer"][:300])


st.set_page_config(page_title="블라인드 채점", layout="wide")
items = load()
todo = [x for x in items if not x["human_verdict"]]
st.sidebar.progress((len(items) - len(todo)) / len(items), text=f"채점 {len(items) - len(todo)}/{len(items)}")
st.sidebar.markdown(GUIDE)
if not todo:
    report(items)
    if st.sidebar.button("처음부터 다시 채점"):
        save([{**x, "human_verdict": None} for x in items])
        st.rerun()
    st.stop()

x = todo[0]
st.caption(f"{x['sample']} · 어느 시스템의 답인지와 LLM 판정은 숨겨져 있습니다")
st.markdown(f"### {x['question']}")
left, right = st.columns(2)
left.markdown("**정답**")
left.info(x["gold"])
right.markdown("**채점할 답변**")
right.write(x["answer"])
cols = st.columns(len(LABELS))
for col, (key, label) in zip(cols, LABELS.items()):
    if col.button(label, use_container_width=True, key=key):
        x["human_verdict"] = key
        save(items)
        st.rerun()
