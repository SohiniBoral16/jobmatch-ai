"""
Pipeline Orchestrator — JobMatch AI
======================================
Runs the full 5-tool pipeline sequentially with observability.
Used by the Streamlit UI (app.py) so it can stream progress messages.

Usage:
    from graph.pipeline import run_pipeline

    final_state, obs_summary = run_pipeline(initial_state,
                                            progress_callback=print)

The LangGraph DAG (graph/workflow.py) is an alternative entrypoint
for programmatic invocation without progress streaming.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from graph.nodes.ranker import rank_jobs
from graph.nodes.resume_parser import parse_resume
from graph.nodes.scrapers import (
    scrape_glassdoor,
    scrape_indeed,
    scrape_linkedin,
)
from utils.observability import (
    PipelineObserver,
    instrument_node,
    setup_langsmith,
)


def run_pipeline(
    initial_state: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Execute the full JobMatch AI pipeline.

    Steps:
        1. parse_resume   — LLM extracts structured profile from PDF
        2-4. scrape_*     — Scrape LinkedIn / Indeed / Glassdoor (sequential here)
        5. rank_jobs       — LLM scores each job against the profile

    Args:
        initial_state:       GraphState-compatible dict with at least
                             resume_pdf_path, search_sites, location, etc.
        progress_callback:   Optional callable receiving status strings
                             (used by st.write() in the Streamlit UI).

    Returns:
        (final_state, observability_summary)
        final_state      — merged state after all nodes ran
        observability_summary — latency, tokens, cost, success rate dict
    """
    setup_langsmith()
    observer = PipelineObserver()

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    state = deepcopy(initial_state)
    sites = state.get("search_sites", [])

    # ── Tool 1: Parse Resume ─────────────────────────────────────────────────
    _progress("**Tool 1 / 5** — Parsing resume with AI...")
    parse_node = instrument_node("parse_resume", parse_resume, observer)
    parse_result = parse_node(state)
    state.update(parse_result)

    if state.get("error"):
        _progress(f"❌ Resume parsing failed: {state['error']}")
        observer.persist()
        return state, observer.summary()

    skills_found = len((state.get("resume_profile") or {}).get("skills", []))
    _progress(f"✅ Resume parsed — {skills_found} skills extracted")

    # ── Tools 2-4: Scrape Job Sites ──────────────────────────────────────────
    scraper_map = {
        "linkedin":  instrument_node("scrape_linkedin",  scrape_linkedin,  observer),
        "indeed":    instrument_node("scrape_indeed",    scrape_indeed,    observer),
        "glassdoor": instrument_node("scrape_glassdoor", scrape_glassdoor, observer),
    }

    total_jobs = 0
    for site in sites:
        node_fn = scraper_map.get(site)
        if not node_fn:
            continue
        _progress(f"**Scraping {site.capitalize()}...**")
        result = node_fn(state)
        state.update(result)
        count = len(result.get(f"{site}_jobs", []))
        total_jobs += count
        _progress(f"✅ {site.capitalize()}: {count} jobs found")

    if total_jobs == 0:
        state["error"] = (
            "No jobs found from any of the selected sites. "
            "Try a different search term, location, or increase HOURS_OLD in .env."
        )
        _progress(f"⚠️ {state['error']}")
        observer.persist()
        return state, observer.summary()

    # ── Tool 5: Rank Jobs ────────────────────────────────────────────────────
    _progress(f"**Tool 5 / 5** — AI-ranking {total_jobs} jobs against your profile...")
    rank_node = instrument_node("rank_jobs", rank_jobs, observer)
    rank_result = rank_node(state)
    state.update(rank_result)

    ranked_count = len(state.get("ranked_jobs", []))
    _progress(f"✅ Ranking complete — {ranked_count} jobs scored")

    observer.persist()
    return state, observer.summary()
