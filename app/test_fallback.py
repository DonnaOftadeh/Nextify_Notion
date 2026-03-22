# app/test_fallback.py
import asyncio
from app.agents import _ask_llm

async def main():
    print("➡️ Testing fallback logic...")
    prompt = "Write a one-sentence poem about the sky."
    reply = await _ask_llm(prompt)
    print("✅ Response:", reply)

if __name__ == "__main__":
    asyncio.run(main())
