# app/test_openai.py
import os
from dotenv import load_dotenv

# Load keys from app/.env
load_dotenv("app/.env")

api_key = os.getenv("OPENAI_API_KEY")
model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print(f"Provider: OpenAI")
print(f"Model: {model_id}")

if not api_key:
    raise SystemExit("❌ OPENAI_API_KEY is missing. Put it in app/.env (do NOT commit).")

from openai import OpenAI
client = OpenAI(api_key=api_key)

try:
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Quick check: reply in one short sentence to confirm you are online."}],
        max_tokens=50
    )
    print("\n✅ OpenAI response:\n", resp.choices[0].message.content.strip())
except Exception as e:
    print("\n❌ Error calling OpenAI:", repr(e))
