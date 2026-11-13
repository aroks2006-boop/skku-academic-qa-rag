# SKKU 학사 Q&A RAG

## 목표
성균관대 학사·졸업요건 질문에 문서 근거로 답하는 RAG 도구
일반 LLM, 킹고봇과 정답률 비교 평가 (설문 없이 정답셋 자동 채점)

## 스택
Python, PyMuPDF, ChromaDB, OpenAI API, Streamlit

## 구조
data/raw/ 원본 문서, ingest.py 인덱싱, rag.py 검색+답변, app.py UI, eval/ 평가

## 코드 스타일
인라인 주석 없이 깔끔하게
while보다 for 선호
