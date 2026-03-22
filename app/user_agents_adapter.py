# app/user_agents_adapter.py
#
# Adapter that calls YOUR agents (from your notebook) with the same wiring
# as your final, parallel architecture. If imports fail, we raise a clear
# error so you know where to plug in your code.

import asyncio
from typing import Dict, Any, List, Tuple

# ---- EDIT THESE IMPORTS to match your notebook's module/class names ----
# Example: you exported these to a python file `my_agents.py`
# from .my_agents import (
#     OrchestratorAgent, ResearchAgent, AnalystAgent,
#     SynthesizerAgent, CriticAgent, WriterAgent
# )

# For now we try both local app module and a generic name; customize as needed.
try:
    from .my_agents import (
        OrchestratorAgent, ResearchAgent, AnalystAgent,
        SynthesizerAgent, CriticAgent, WriterAgent
    )
except Exception as e:
    # If your notebook was exported to, say, 'nb_agents.py', change below:
    # from .nb_agents import ...
    raise ImportError(
        "Could not import your agents. "
        "Open app/user_agents_adapter.py and point the imports to your module. "
        f"Original error: {e}"
    )

# ------------------------------------------------------------------------


class AgentContext:
    """Shared context passed among agents (mimics your notebook state)."""
    def __init__(self, entry: str, payload: Dict[str, Any]):
        self.entry = entry
        self.payload = payload

        # Artifacts produced by each step
        self.guardrails = {}
        self.task_plan = []
        self.research = {}
        self.analysis = {}
        self.synthesis = {}
        self.critic = {}
        self.writer = {}

        # For your convenience: raw markdown where applicable
        self.research_md = ""
        self.analysis_md = ""
        self.synthesis_md = ""
        self.critic_md = ""


async def run_with_your_agents(entry: str, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Runs YOUR notebook agents in (mostly) the same order and parallelization
    you used: Orchestrator -> parallel research -> analysis -> synthesis -> critic -> writer.
    Returns (events, final_markdown).

    Each event: {"step": "...", "message": "...", "progress": int}
    """
    ctx = AgentContext(entry, payload)
    events: List[Dict[str, Any]] = []
    progress = 0

    def push(step, message, inc):
        nonlocal progress
        progress = min(100, progress + inc)
        events.append({"step": step, "message": message, "progress": progress})

    # Instantiate your agents (edit constructor args if needed)
    orchestrator = OrchestratorAgent()
    researcher = ResearchAgent()
    analyst = AnalystAgent()
    synthesizer = SynthesizerAgent()
    critic = CriticAgent()
    writer = WriterAgent()

    # 1) Orchestrate
    push("Orchestrate", "Planning tasks / guardrails…", 8)
    o = await _maybe_async(orchestrator.run(ctx))
    # Expect your orchestrator to fill ctx.guardrails and ctx.task_plan
    # If not, we set defaults:
    ctx.guardrails = getattr(ctx, "guardrails", {}) or getattr(o, "guardrails", {}) or {}
    ctx.task_plan = getattr(ctx, "task_plan", []) or getattr(o, "task_plan", []) or [
        "research", "analysis", "synthesis", "critic", "final"
    ]

    # 2) Research (allow parallel sub-tasks if your agent supports it)
    push("Research", "Gathering facts / comps…", 18)
    # If your ResearchAgent internally runs parallel subtasks, just call once:
    r = await _maybe_async(researcher.run(ctx))
    # Expect ctx.research and ctx.research_md to be set by your agent
    ctx.research = getattr(ctx, "research", {}) or getattr(r, "data", {}) or ctx.research
    ctx.research_md = getattr(ctx, "research_md", "") or getattr(r, "markdown", "") or ""

    # 3) Analysis
    push("Analysis", "Deriving insights / options…", 22)
    a = await _maybe_async(analyst.run(ctx))
    ctx.analysis = getattr(ctx, "analysis", {}) or getattr(a, "data", {}) or ctx.analysis
    ctx.analysis_md = getattr(ctx, "analysis_md", "") or getattr(a, "markdown", "") or ""

    # 4) Synthesis
    push("Synthesis", "OKRs / 90-day / MVP…", 22)
    s = await _maybe_async(synthesizer.run(ctx))
    ctx.synthesis = getattr(ctx, "synthesis", {}) or getattr(s, "data", {}) or ctx.synthesis
    ctx.synthesis_md = getattr(ctx, "synthesis_md", "") or getattr(s, "markdown", "") or ""

    # 5) Critique
    push("Critique", "Stress testing & mitigation…", 12)
    c = await _maybe_async(critic.run(ctx))
    ctx.critic = getattr(ctx, "critic", {}) or getattr(c, "data", {}) or ctx.critic
    ctx.critic_md = getattr(ctx, "critic_md", "") or getattr(c, "markdown", "") or ""

    # 6) Writer (final format = your format)
    push("Compose", "Preparing final brief…", 15)
    w = await _maybe_async(writer.run(ctx))
    # Your Writer should return either a markdown string or a dict of pieces.
    final_md = None

    # If your writer already returns a final string:
    if isinstance(w, str) and w.strip():
        final_md = w

    # Or if your writer returns structured pieces (common pattern):
    if not final_md and isinstance(w, dict):
        from .templates import render_final_markdown
        final_md = render_final_markdown(entry, payload, w)

    # Fallback: stitch the intermediate markdowns
    if not final_md:
        parts = [
            "# Final Brief (fallback)",
            "## Research", ctx.research_md,
            "## Analysis", ctx.analysis_md,
            "## Synthesis", ctx.synthesis_md,
            "## Critic Review", ctx.critic_md,
        ]
        final_md = "\n\n".join(p for p in parts if p)

    push("Complete", "Report ready.", 3)
    return events, final_md


async def _maybe_async(result_or_coro):
    """Support both sync and async agent .run() implementations."""
    if asyncio.iscoroutine(result_or_coro):
        return await result_or_coro
    return result_or_coro
