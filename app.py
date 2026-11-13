import streamlit as st

from config import TOP_K
from rag import answer

st.set_page_config(page_title="SKKU 학사 Q&A", page_icon="🎓")
st.title("SKKU 학사 Q&A")
st.caption("학사·졸업요건 문서를 근거로 답합니다. 최종 확인은 학사 공지와 소속 학과에서 해 주세요.")

with st.sidebar:
    k = st.slider("검색 문서 수", 1, 10, TOP_K)
    if st.button("대화 지우기"):
        st.session_state.messages = []

st.session_state.setdefault("messages", [])


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"근거 문서 {len(sources)}개"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['source']} p.{s['page']}**")
            st.text(s["text"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("예: 2022학번 졸업 최소 이수학점은?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("문서 검색 중..."):
            result = answer(question, k)
        st.markdown(result["answer"])
        show_sources(result["sources"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
