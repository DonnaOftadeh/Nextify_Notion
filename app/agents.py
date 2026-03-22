# app/agents.py
from __future__ import annotations

import os
import asyncio
from typing import Dict, Any, Callable, List, Tuple

from .prompts import build_agent_prompts, SYSTEM_GUARDRAILS
try:
    from .adk_agents import run_multi_agent_adk
except Exception:
    run_multi_agent_adk = None
# ---------------- Provider order ----------------
def _provider_order() -> List[str]:
    first = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    return ["openai", "gemini"] if first == "openai" else ["gemini", "openai"]

# ---------------- Lazy clients ------------------
_GEMINI = None
_OPENAI = None

def _init_gemini():
    global _GEMINI
    if _GEMINI is not None:
        return _GEMINI
    import google.generativeai as genai  # pip install google-generativeai
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
    _GEMINI = genai.GenerativeModel(model_name)
    return _GEMINI

def _init_openai():
    global _OPENAI
    if _OPENAI is not None:
        return _OPENAI
    from openai import OpenAI  # pip install openai>=1.0
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _OPENAI = (client, model)
    return _OPENAI

async def _llm_call(provider: str, system: str, user: str) -> str:
    if provider == "gemini":
        gm = _init_gemini()
        resp = await asyncio.to_thread(
            gm.generate_content,
            [{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
            safety_settings=None,
        )
        return (resp.text or "").strip()

    if provider == "openai":
        client, model = _init_openai()
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    raise RuntimeError(f"Unknown provider: {provider}")

# --------------- Helpers -----------------------
def _human_title(step_key: str) -> str:
    # "1_problem_snapshot" -> "Problem Snapshot"
    if "_" in step_key:
        step_key = step_key.split("_", 1)[1]
    return " ".join(w.capitalize() for w in step_key.split("_"))

def _normalize_block(s: str) -> str:
    """Trim, collapse multiple blank lines, drop immediately duplicated header separators."""
    s = (s or "").strip()
    out, blanks = [], 0
    for ln in s.splitlines():
        if ln.strip() == "":
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        out.append(ln.rstrip())

    cleaned: List[str] = []
    for ln in out:
        st = ln.strip()
        if st.startswith("|") and st.endswith("|") and "---" in st:
            if cleaned and cleaned[-1].strip() == st:
                continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def _anchor_lines(section_text: str) -> List[str]:
    """Short anchors (first H2 + first table header) to maintain consistency without copying."""
    anchors: List[str] = []
    for ln in (section_text or "").splitlines():
        s = ln.strip()
        if s.startswith("## "):
            anchors.append(s[3:][:160]); break
    for ln in (section_text or "").splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|") and "---" in s:
            anchors.append(s[:160]); break
    return anchors[:2]

def _compact_context(history_sections: List[str], limit_chars: int = 7000) -> str:
    """
    Create a compact continuity context from prior sections, used ONLY on provider switches.
    We add headings and tables as-is, but cap total size. Strict instruction forbids quoting it.
    """
    if not history_sections:
        return ""
    # Join last few sections until char limit
    buf: List[str] = []
    total = 0
    for block in reversed(history_sections):
        b = block.strip()
        if not b:
            continue
        if total + len(b) > limit_chars and total > 0:
            break
        buf.append(b)
        total += len(b)
        if total >= limit_chars:
            break
    ctx = "\n\n".join(reversed(buf)).strip()
    return ctx

# --------------- Orchestrator ------------------
async def run_multi_agent(submission: Dict[str, Any],
                          progress_cb: Callable[[int, str, str], None]) -> str:
    """
    Orchestrate section-by-section using prompts.build_agent_prompts() exactly as-is.
    - Each step is generated independently (no repetition).
    - NORMAL CASE: only pass tiny 'anchors' from prior steps (for consistency).
    - PROVIDER SWITCH CASE: also pass a compact 'continuity context' (prior sections) so the new
      model knows what's been done, with explicit instruction NOT to quote or summarize it.
    """
    journey = (submission.get("journey_type") or "").lower().strip()
    payload = submission.get("payload") or {}
    evaluate_stage = payload.pop("evaluate_stage", None)
    evaluation_model = payload.pop("evaluation_model", None)
    if journey == "idea" and run_multi_agent_adk:
        history = await run_multi_agent_adk(
            payload,
            progress_cb,
            evaluate_stage=evaluate_stage,
            evaluation_model=evaluation_model,
        )
        # Merge the ADK outputs and any evaluations into a single report
        combined_report = "\n\n".join(filter(None, [
            history.get("brainstorm_md"),
            history.get("eval_brainstorm_md"),
            history.get("roadmap_md"),
            history.get("eval_roadmap_md"),
            history.get("feature_prioritization_md"),
            history.get("eval_feature_md"),
            history.get("okr_planning_md"),
            history.get("eval_okr_plan_md"),
            history.get("final_evaluation_md"),
        ]))
        return combined_report
    
    prompts_map: Dict[str, str] = build_agent_prompts(journey, payload)
    if not isinstance(prompts_map, dict) or not prompts_map:
        raise RuntimeError(f"No prompts for journey '{journey}'")

    steps: List[Tuple[str, str]] = list(prompts_map.items())

    providers = _provider_order()
    current_provider = providers[0]
    history_texts: List[str] = []   # full cleaned text per finished section
    anchors: List[str] = []         # short anchors to keep terms/numbers consistent
    out_sections: List[str] = []    # final stitched sections (with H2 headers)
    used_providers: List[Tuple[str, str]] = []

    if callable(progress_cb):
        progress_cb(0, "Orchestration", f"Provider: {current_provider}")

    for idx, (step_key, base_prompt) in enumerate(steps, start=1):
        title = _human_title(step_key)
        if callable(progress_cb):
            progress_cb(idx, title, f"Generating {title}…")

        # build the user prompt
        tail = ""
        if anchors:
            tail += (
                "\n\nConstraints for this section ONLY:\n"
                "- Use earlier outputs solely to keep names/numbers consistent.\n"
                "- Do NOT copy or summarize previous sections.\n"
                f"- Key anchors: {'; '.join(anchors[-5:])}\n"
            )

        # try provider(s)
        text = ""
        provider_used = None
        tried = []
        for prov in providers:
            if prov in tried:
                continue
            tried.append(prov)

            # If switching provider this turn, ship compact continuity context
            user_prompt = base_prompt + tail
            if prov != current_provider and history_texts:
                ctx = _compact_context(history_texts, limit_chars=7000)
                if ctx:
                    user_prompt += (
                        "\n\nContinuity context (for your awareness ONLY — do NOT quote, copy, or summarize this content; "
                        "use it merely to stay consistent with terms, names, and numbers):\n"
                        f"{ctx}\n"
                    )

            try:
                text = await _llm_call(prov, SYSTEM_GUARDRAILS, user_prompt)
                provider_used = prov
                break
            except Exception:
                continue

        if not text:
            raise RuntimeError(f"{step_key}: all providers failed")

        # if actual provider changed, update current and let UI know (status only)
        if provider_used != current_provider:
            current_provider = provider_used
            if callable(progress_cb):
                progress_cb(idx, title, f"Switched provider: {current_provider}")

        cleaned = _normalize_block(text)
        # store the exact section text (no heading) in history for potential future switches
        history_texts.append(cleaned)
        anchors.extend(_anchor_lines(cleaned))

        # stitch with H2 section header (keeps your report structure tidy)
        out_sections.append(f"## {title}\n\n{cleaned}".rstrip())
        used_providers.append((step_key, current_provider))

        await asyncio.sleep(0)

    # (Providers trail for debugging in /api/debug only; not rendered to PDF)
    out_sections.append("\n<!-- providers:" + ",".join(f"{k}:{p}" for k,p in used_providers) + " -->\n")
    return "\n\n".join(out_sections).strip()
