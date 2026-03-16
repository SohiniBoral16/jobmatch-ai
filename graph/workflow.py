"""
LangGraph Workflow — JobMatch AI
==================================
Wires all 5 tools together:
  1. parse_resume   — LLM extracts structured profile from PDF
  2. scrape_linkedin
  3. scrape_indeed   — all three run in parallel (fan-out)
  4. scrape_glassdoor
  5. rank_jobs       — LLM scores each job against the profile

Flow:
  START → parse_resume → [linkedin | indeed | glassdoor] → rank_jobs → END
                              (fan-out, parallel)      (fan-in)

NOTE: The Streamlit UI (app.py) uses graph/pipeline.py directly so it can
stream progress messages.  This file provides an alternative compiled graph
that can be invoked with:  workflow.app.invoke(initial_state)
"""

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes.resume_parser import parse_resume
from graph.nodes.scrapers import scrape_linkedin, scrape_indeed, scrape_glassdoor
from graph.nodes.ranker import rank_jobs
from utils.observability import setup_langsmith

try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator


# ── LangSmith-traced wrappers ───────────────────────────────────────────────

@traceable(name="parse_resume", run_type="chain")
def _parse_resume(state: GraphState) -> dict:
    return parse_resume(state)


@traceable(name="scrape_linkedin", run_type="tool")
def _scrape_linkedin(state: GraphState) -> dict:
    return scrape_linkedin(state)


@traceable(name="scrape_indeed", run_type="tool")
def _scrape_indeed(state: GraphState) -> dict:
    return scrape_indeed(state)


@traceable(name="scrape_glassdoor", run_type="tool")
def _scrape_glassdoor(state: GraphState) -> dict:
    return scrape_glassdoor(state)


@traceable(name="rank_jobs", run_type="chain")
def _rank_jobs(state: GraphState) -> dict:
    return rank_jobs(state)


# ── Error-gate node inserted between parse and scrapers ─────────────────────

def _error_gate(state: GraphState) -> dict:
    """Pass-through — exists only so the conditional edge can branch."""
    return {}


def _route_after_gate(state: GraphState) -> str:
    if state.get("error"):
        return "end"
    return "continue"


# ── Build and compile ────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """Build and compile the LangGraph DAG."""
    setup_langsmith()

    wf = StateGraph(GraphState)

    # Register nodes
    wf.add_node("parse_resume",    _parse_resume)
    wf.add_node("error_gate",      _error_gate)
    wf.add_node("scrape_linkedin", _scrape_linkedin)
    wf.add_node("scrape_indeed",   _scrape_indeed)
    wf.add_node("scrape_glassdoor",_scrape_glassdoor)
    wf.add_node("rank_jobs",       _rank_jobs)

    # Entry
    wf.set_entry_point("parse_resume")

    # parse_resume → error_gate (always)
    wf.add_edge("parse_resume", "error_gate")

    # error_gate → fan-out OR end
    wf.add_conditional_edges(
        "error_gate",
        _route_after_gate,
        {
            "continue": "scrape_linkedin",
            "end": END,
        },
    )
    # Fan-out: error_gate also directly triggers the other two scrapers
    wf.add_edge("error_gate", "scrape_indeed")
    wf.add_edge("error_gate", "scrape_glassdoor")

    # Fan-in: all three scrapers converge on rank_jobs
    wf.add_edge("scrape_linkedin",  "rank_jobs")
    wf.add_edge("scrape_indeed",    "rank_jobs")
    wf.add_edge("scrape_glassdoor", "rank_jobs")

    # Done
    wf.add_edge("rank_jobs", END)

    return wf.compile()


# Module-level compiled graph for direct invocation
app = build_workflow()
