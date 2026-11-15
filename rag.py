import json
import math
import re
from collections import Counter
from functools import lru_cache

import chromadb
from openai import OpenAI

from config import CANDIDATES, CHAT_MODEL, CHROMA_DIR, COLLECTION, EMBED_MODEL, RETRIEVE_MODE, TOP_K

RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75

RERANK_SYSTEM = """당신은 검색 결과 재정렬기입니다.
[질문]에 답하는 데 직접 필요한 내용을 담은 후보부터 순서대로 번호를 고르세요.
학번, 학과, 학기 등 질문의 조건과 정확히 일치하는 후보를 우선합니다.
JSON 형식으로만 출력: {"ranking": [번호, 번호, ...]}"""

NOT_FOUND = "제공된 문서에서 확인할 수 없습니다."

RAG_SYSTEM = f"""당신은 성균관대학교 학사·졸업요건 안내 도우미입니다.
반드시 아래 [문서] 내용만 근거로 한국어로 답하세요.
- 근거로 쓴 문서 번호를 문장 끝에 [1], [2] 형식으로 표기합니다.
- 학번·학과·과정별로 기준이 다르면 구분해서 설명합니다.
- 표에서는 질문의 조건(학번, 학년, 학기, 계열, 대학)에 해당하는 열과 행을 먼저 찾고, 그 교차 값만 사용합니다.
- 답에 쓴 숫자가 어느 행·열(또는 조항)에서 왔는지 한 줄로 밝힙니다.
- 여러 조각이 다른 결론을 말하면 질문 상황에 가장 직접 해당하는 조항(제적, 유급 등 결과)을 우선합니다.
- 문서에 근거가 없으면 추측하지 말고 "{NOT_FOUND}"라고만 답합니다."""

PLAIN_SYSTEM = "당신은 성균관대학교 학사·졸업요건 안내 도우미입니다. 한국어로 간결하게 답하세요."


@lru_cache
def client() -> OpenAI:
    return OpenAI()


@lru_cache
def collection():
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[가-힣]+|[a-z]+|\d+", text.lower()):
        tokens.append(word)
        if re.match(r"[가-힣]", word) and len(word) > 2:
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


class BM25:
    def __init__(self, ids: list[str], docs: list[str]):
        self.ids = ids
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, n: int) -> list[str]:
        terms = set(tokenize(query))
        scores = []
        for doc_id, tf, length in zip(self.ids, self.tfs, self.lens):
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg)
            s = sum(self.idf[t] * tf[t] * (BM25_K1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            scores.append((s, doc_id))
        return [doc_id for s, doc_id in sorted(scores, reverse=True)[:n] if s > 0]


@lru_cache
def corpus() -> dict[str, dict]:
    data = collection().get(include=["documents", "metadatas"])
    return {
        i: {"id": i, "text": doc, "source": meta["source"], "page": meta["page"]}
        for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    }


@lru_cache
def bm25() -> BM25:
    docs = corpus()
    return BM25(list(docs), [d["text"] for d in docs.values()])


def vector_search(question: str, n: int) -> list[str]:
    vec = client().embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
    return collection().query(query_embeddings=[vec], n_results=n)["ids"][0]


def fuse(rankings: list[list[str]]) -> list[tuple[str, float]]:
    scores = Counter()
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1 / (RRF_K + rank)
    return scores.most_common()


def rerank(question: str, hits: list[dict]) -> list[dict]:
    listing = "\n\n".join(f"<{i}>\n{h['text'][:700]}" for i, h in enumerate(hits))
    res = client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": f"[질문]\n{question}\n\n[후보]\n{listing}"},
        ],
    )
    try:
        picked = [int(i) for i in json.loads(res.choices[0].message.content)["ranking"]]
    except (ValueError, KeyError, TypeError):
        return hits
    order = list(dict.fromkeys(i for i in picked if 0 <= i < len(hits)))
    order += [i for i in range(len(hits)) if i not in order]
    return [hits[i] for i in order]


def retrieve(question: str, k: int = TOP_K, mode: str = RETRIEVE_MODE) -> list[dict]:
    docs = corpus()
    pool = max(k, CANDIDATES)
    if mode == "vector":
        ranked = [(i, 0.0) for i in vector_search(question, pool)]
    else:
        ranked = fuse([vector_search(question, pool * 2), bm25().search(question, pool * 2)])[:pool]
    hits = [{**docs[i], "score": s} for i, s in ranked]
    if mode == "rerank":
        hits = rerank(question, hits[:CANDIDATES])
    return hits[:k]


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({h['source']} p.{h['page']})\n{h['text']}" for i, h in enumerate(hits, 1))


def chat(system: str, user: str, model: str | None = None) -> str:
    res = client().chat.completions.create(
        model=model or CHAT_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return res.choices[0].message.content.strip()


def answer(question: str, k: int = TOP_K) -> dict:
    hits = retrieve(question, k)
    text = chat(RAG_SYSTEM, f"[문서]\n{format_context(hits)}\n\n[질문]\n{question}")
    return {"answer": text, "sources": hits}


def answer_plain(question: str) -> str:
    return chat(PLAIN_SYSTEM, question)


if __name__ == "__main__":
    import sys

    result = answer(" ".join(sys.argv[1:]) or "졸업에 필요한 최소 이수학점은?")
    print(result["answer"])
    for i, h in enumerate(result["sources"], 1):
        print(f"[{i}] {h['source']} p.{h['page']}")
