import os

from openai import OpenAI

from config import CHAT_MODEL, EMBED_MODEL


def main() -> None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("OPENAI_API_KEY 가 없습니다. .env 파일을 확인하세요.")
        return
    print(f"키 인식됨: {key[:7]}... (길이 {len(key)})")

    client = OpenAI()
    vec = client.embeddings.create(model=EMBED_MODEL, input=["테스트"]).data[0].embedding
    print(f"임베딩 OK: {EMBED_MODEL}, 차원 {len(vec)}")

    res = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": "OK라고만 답해"}],
    )
    print(f"채팅 OK: {CHAT_MODEL}, 응답 '{res.choices[0].message.content.strip()}'")


if __name__ == "__main__":
    main()
