"""Prints the first/last 4 chars of the loaded API key so you can verify it's new."""
from graphrag.core.config import get_settings
s = get_settings()
k = s.google_api_key
if k:
    print(f"Key loaded: {k[:8]}...{k[-4:]}  (length={len(k)})")
else:
    print("No key loaded — GOOGLE_API_KEY is empty in .env")
