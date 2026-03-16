"""
Tools 2/3/4: Parallel Job Scrapers
=====================================
Three LangGraph nodes — scrape_linkedin, scrape_indeed, scrape_glassdoor —
that each run independently (fan-out from the pipeline orchestrator).

Each node:
  1. Reads the parsed resume_profile (or user custom_search) to build queries
  2. Calls python-jobspy to scrape job listings from one site
  3. Returns {site}_jobs as a list of JobListing dicts

No LLM calls here — pure scraping.
"""

import os
from jobspy import scrape_jobs

from graph.state import GraphState, JobListing


# ── Env-configurable defaults ─────────────────────────────────────────────────
HOURS_OLD          = int(os.environ.get("HOURS_OLD", "72"))
RESULTS_PER_SEARCH = int(os.environ.get("RESULTS_PER_SEARCH", "15"))


# ── Query builder ─────────────────────────────────────────────────────────────

def _build_search_queries(state: GraphState) -> list[dict]:
    """
    Generate a list of search queries from the resume profile or user overrides.

    Each query is {"search_term": str, "location": str}.
    If the user supplied a custom_search term, that takes priority.
    Otherwise we auto-generate from the parsed resume profile.
    """
    profile     = state.get("resume_profile")
    custom_term = state.get("search_term")
    location    = state.get("location") or "Bangalore, India"

    if custom_term:
        return [{"search_term": custom_term, "location": location}]

    if not profile:
        return [{"search_term": "Senior Software Engineer", "location": location}]

    titles    = profile.get("job_titles", ["Software Engineer"])
    skills    = profile.get("skills", [])
    locations = profile.get("preferred_locations") or [location]

    queries: list[dict] = []
    top_skills = " ".join(skills[:3])

    # Targeted: top title + top skills, for up to 2 locations
    for title in titles[:2]:
        for loc in locations[:2]:
            queries.append({
                "search_term": f"{title} {top_skills}".strip(),
                "location":    loc,
            })

    # Broader: just the title (catches more results)
    for title in titles[:2]:
        queries.append({
            "search_term": title,
            "location":    locations[0],
        })

    return queries


# ── Core scraper ──────────────────────────────────────────────────────────────

def _scrape_site(site_name: str, state: GraphState) -> list[JobListing]:
    """
    Run all search queries against a single job site.

    Returns a deduplicated list of JobListing dicts.
    Errors on individual queries are caught and logged — they never
    crash the whole pipeline.
    """
    queries  = _build_search_queries(state)
    all_jobs: list[JobListing] = []

    for query in queries:
        try:
            df = scrape_jobs(
                site_name=[site_name],
                search_term=query["search_term"],
                location=query["location"],
                results_wanted=RESULTS_PER_SEARCH,
                hours_old=HOURS_OLD,
                country_indeed="India",
            )

            if df is None or len(df) == 0:
                continue

            for _, row in df.iterrows():
                job: JobListing = {
                    "title":       str(row.get("title",       "N/A")),
                    "company":     str(row.get("company",     "N/A")),
                    "location":    str(row.get("location",    "N/A")),
                    "description": str(row.get("description", "")),
                    "job_url":     str(row.get("job_url",     "")),
                    "site":        site_name,
                    "date_posted": str(row.get("date_posted", "")),
                }
                all_jobs.append(job)

        except Exception as exc:
            print(f"[{site_name}] Scrape error for '{query['search_term']}': {exc}")

    return all_jobs


# ── LangGraph nodes ───────────────────────────────────────────────────────────

def scrape_linkedin(state: GraphState) -> dict:
    """Tool 2 — Scrape LinkedIn for job listings."""
    if "linkedin" not in state.get("search_sites", []):
        return {"linkedin_jobs": []}

    print("[Tool 2] Scraping LinkedIn...")
    jobs = _scrape_site("linkedin", state)
    print(f"[Tool 2] LinkedIn: {len(jobs)} jobs found")
    return {"linkedin_jobs": jobs}


def scrape_indeed(state: GraphState) -> dict:
    """Tool 3 — Scrape Indeed for job listings."""
    if "indeed" not in state.get("search_sites", []):
        return {"indeed_jobs": []}

    print("[Tool 3] Scraping Indeed...")
    jobs = _scrape_site("indeed", state)
    print(f"[Tool 3] Indeed: {len(jobs)} jobs found")
    return {"indeed_jobs": jobs}


def scrape_glassdoor(state: GraphState) -> dict:
    """Tool 4 — Scrape Glassdoor for job listings."""
    if "glassdoor" not in state.get("search_sites", []):
        return {"glassdoor_jobs": []}

    print("[Tool 4] Scraping Glassdoor...")
    jobs = _scrape_site("glassdoor", state)
    print(f"[Tool 4] Glassdoor: {len(jobs)} jobs found")
    return {"glassdoor_jobs": jobs}
