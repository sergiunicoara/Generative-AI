"""Minimal test: call Gemini embed API directly and print result shape."""
import os
os.environ["PYTHONUTF8"] = "1"

from graphrag.core.config import get_settings
from google import genai
from google.genai import types as genai_types

cfg = get_settings()
print(f"Model : {cfg.gemini_embed_model}")
print(f"Key   : {cfg.google_api_key[:8]}...{cfg.google_api_key[-4:]}")

client = genai.Client(api_key=cfg.google_api_key)

try:
    result = client.models.embed_content(
        model=cfg.gemini_embed_model,
        contents="hello world",
        config=genai_types.EmbedContentConfig(task_type="retrieval_query"),
    )
    vec = result.embeddings[0].values
    print(f"OK — embedding dim={len(vec)}, first3={[round(v,4) for v in vec[:3]]}")
except Exception as e:
    print(f"FAILED: {e}")

    # Try fallback model name
    print("\nTrying fallback: text-embedding-004 ...")
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents="hello world",
            config=genai_types.EmbedContentConfig(task_type="retrieval_query"),
        )
        vec = result.embeddings[0].values
        print(f"text-embedding-004 OK — dim={len(vec)}")
        print("=> Update GEMINI_EMBED_MODEL=text-embedding-004 in .env")
    except Exception as e2:
        print(f"text-embedding-004 also failed: {e2}")
