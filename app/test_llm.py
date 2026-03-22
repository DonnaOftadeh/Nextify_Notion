# app/test_gemini.py
import os
from dotenv import load_dotenv

# Load keys and model from app/.env
load_dotenv("app/.env")

provider = os.getenv("LLM_PROVIDER", "gemini").lower()
api_key  = os.getenv("GEMINI_API_KEY")
model_id = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

print(f"Provider: {provider}")
print(f"Model: {model_id}")

if provider != "gemini":
    print("Note: LLM_PROVIDER is not 'gemini' in app/.env. (That's fine, we only use GEMINI_* here.)")

if not api_key:
    raise SystemExit("❌ GEMINI_API_KEY is missing. Put it in app/.env (do NOT commit).")

# --- minimal Gemini call ---
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel(model_id)
    resp = model.generate_content("Quick check: reply with one short sentence confirming you are online.")
    print("\n✅ Gemini response:\n", (resp.text or "").strip()[:500])
except ResourceExhausted as e:
    print("\n⚠️  Gemini quota/rate limit hit.")
    print("    Details:", e.message)
    print("    Docs: https://ai.google.dev/gemini-api/docs/rate-limits")
except PermissionDenied as e:
    print("\n❌ Permission/Key issue with Gemini. Check your API key & project access.")
    print("   Details:", e.message)
except InvalidArgument as e:
    print("\n❌ Model or request invalid. Double-check GEMINI_MODEL in app/.env.")
    print("   Details:", e.message)
except Exception as e:
    print("\n❌ Unexpected error:", repr(e))
