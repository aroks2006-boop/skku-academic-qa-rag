import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OCR_DIR = ROOT / "data" / "ocr"
CHROMA_DIR = ROOT / "chroma"
COLLECTION = "skku_academic"

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-5-mini")

OCR_MIN_CHARS = 30
OCR_DPI = 200
OCR_WORKERS = 10

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
CANDIDATES = 20
RETRIEVE_MODE = os.getenv("RETRIEVE_MODE", "rerank")
