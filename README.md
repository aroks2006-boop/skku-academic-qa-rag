# SKKU 학사 Q&A RAG

성균관대 학사·졸업요건 질문에 학칙·학사제도 안내 문서를 근거로 답하는 RAG 도구.
일반 LLM, 킹고봇과 정답률을 자동 채점으로 비교한다.

## 결과 (2026-11-14, 49문항, 채점 3회 반복)

| 시스템 | 정답률 | 부분정답 포함 | 근거있는 46문항 정답률 | 회피율 |
|---|---|---|---|---|
| RAG (gpt-4.1-mini) | **89.8%** | 94.9% | 89.1% | 0% |
| 일반 LLM (gpt-4.1-mini, 검색 없음) | 10.2% | 22.8% | 10.9% | 0% |
| 킹고봇 (첫 응답 기준) | 8.2% | 8.2% | 4.3% | 90% |

- 채점 모델(gpt-4.1-mini, temperature 0) 3회 반복 표준편차 0.0%p
- 킹고봇 49문항 응답 유형: 사과 문구 29, 추천 질문 목록 7, 모름 6, 메뉴 선택 4, 직접 답변 3 (`eval/kingobot_answers.csv`의 `response_type`)
- 킹고봇은 추천 질문·메뉴를 클릭하지 않은 첫 응답 기준. q047(기숙사)은 킹고봇이 별도 지식으로 맞게 답했지만 "문서 근거 없음" 문항이라 오답 처리됨 → 근거있는 46문항 정답률을 함께 볼 것
- 이전 구성(gpt-4o-mini, 마크다운 표 그대로 인덱싱) 결과: RAG 79.6%, 일반 LLM 18.4% → `eval/answers/gpt-4o-mini/`

### 한계
- 정답셋은 OCR 텍스트를 근거로 작성했고 원본 대조 검수(`verified`)가 끝나지 않았다
- 표 처리·프롬프트 개선은 같은 정답셋의 오답을 보며 진행했으므로 수치가 다소 낙관적일 수 있다

## 설치
```
pip install -r requirements.txt
copy .env.example .env
```
`.env`에 `OPENAI_API_KEY` 입력 후 `python check_api.py`로 연결 확인

## 사용
1. `data/raw/`에 학사 요람, 졸업요건, 학칙 PDF 등을 넣기
2. `python ingest.py` 로 인덱싱 (문서를 바꾸면 다시 실행)
3. `streamlit run app.py` 로 UI 실행, 또는 `python rag.py "질문"`

## 구조
- `ocr.py`: 텍스트 없는 스캔 페이지만 gpt-5-mini로 OCR, 결과는 `data/ocr/`에 페이지별 캐시. 출력 과다·판독 거부 시 페이지를 3분할해 재시도
- `ingest.py`: 페이지를 표·본문·조항 단위로 나누고 청크마다 `[문서 p.N] 장 > 표/조항 제목` 머리말을 붙임. 표는 행마다 `열 제목: 값`으로 풀어 써서 넓은 표도 열을 헷갈리지 않게 함
- `rag.py`: 벡터 + BM25 하이브리드(RRF) 후보 20개를 LLM이 재정렬한 뒤 상위 5개로 답변 (`RETRIEVE_MODE=vector|hybrid|rerank`)
- `app.py`: Streamlit 채팅 UI, 답변별 근거 문서 표시
- `config.py`: 모델·청크·검색 설정 (`OPENAI_CHAT_MODEL`, `OPENAI_EMBED_MODEL`, `OPENAI_OCR_MODEL`)

## 평가
- `eval/questions.jsonl`: 한 줄에 한 문항 `{id, category, question, gold, keywords, source, verified}`
  - `gold`는 원문 근거로 작성, `keywords`는 정답에 반드시 들어가야 할 숫자·용어
  - `source`는 근거 파일과 PDF 페이지 번호(인쇄된 쪽번호 아님), 원본과 대조 후 `verified`를 `true`로
  - 문서에 답이 없는 문항은 `"answerable": false`: "모른다/확인 불가"라고 답해야 정답 처리
- `eval/kingobot_answers.csv`: 킹고봇에 같은 질문을 넣고 받은 첫 응답 (`id,question,answer,response_type`)
- `python eval/run_eval.py [--repeat 3]`
  - `rag`, `plain`(검색 없는 일반 LLM), `kingobot`을 LLM 채점자(correct/partial/incorrect/abstain)와 키워드 재현율로 자동 채점
  - 생성 답변은 `eval/answers/`에 캐시되어 재사용, 다시 생성하려면 해당 파일 삭제
  - `--repeat N`은 채점만 N회 반복해 평균·표준편차 출력
  - 결과는 `eval/results.csv`, 콘솔에 시스템별·카테고리별 점수 출력
- `python eval/retrieval_eval.py`: 정답 페이지가 정해진 질문(`eval/retrieval_cases.jsonl`)으로 검색 모드별 Hit@5·MRR 비교 (현재 rerank Hit@5 100%, MRR 0.900)
