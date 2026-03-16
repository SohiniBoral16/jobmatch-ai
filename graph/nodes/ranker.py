"""
Tool 5: AI Job Ranker
=======================
Collects all jobs from the three scrapers, deduplicates them,
then calls the LLM once per job to score it against the resume profile.

Scoring 0-100:
  80-100  Excellent — most skills match, right level, good location
  60-79   Good      — many skills match, some gaps
  40-59   Decent    — some relevance, needs growth in areas
  20-39   Weak      — few matching skills or experience mismatch
   0-19   Poor      — very different role / requirements

Returns:
    ranked_jobs — list of RankedJob sorted descending by match_score
    _telemetry  — aggregated token/cost tracking
"""

import json

from langchain_core.messages import HumanMessage

from graph.state import GraphState, RankedJob
from utils.llm import get_llm
from utils.observability import estimate_cost_usd, estimate_tokens


# ── Prompt ─────────────────────────────────────────────────────────────────--

RANK_PROMPT = """\
You are a job-matching expert. Score how well this job fits the candidate.

CANDIDATE:
- Name:              {name}
- Skills:            {skills}
- Years experience:  {years_of_experience}
- Recent titles:     {job_titles}
- Preferred cities:  {preferred_locations}
- Summary:           {summary}

JOB POSTING:
- Title:       {job_title}
- Company:     {job_company}
- Location:    {job_location}
- Description: {job_description}

Return ONLY valid JSON (no markdown, no code fences):
{{
    "match_score": <integer 0-100>,
    "match_reasoning": "<2-3 sentences covering skill fit, experience level, location>"
}}

Scoring guide:
  80-100  Excellent — strong skill overlap, right seniority, good location
  60-79   Good      — many skills match, minor gaps
  40-59   Decent    — partial match, room to grow
  20-39   Weak      — few relevant skills or level mismatch
   0-19   Poor      — very different role
"""


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs: first by URL, then by title+company."""
    seen_urls:   set[str] = set()
    seen_keys:   set[str] = set()
    unique: list[dict] = []

    for job in jobs:
        url = (job.get("job_url") or "").strip()
        key = f"{job.get('title', '')}|{job.get('company', '')}".lower()

        if url and url != "nan" and url in seen_urls:
            continue
        if key in seen_keys:
            continue

        if url and url != "nan":
            seen_urls.add(url)
        seen_keys.add(key)
        unique.append(job)

    return unique


# ── LangGraph node ─────────────────────────────────────────────────────────---

def rank_jobs(state: GraphState) -> dict:
    """
    Tool 5 — LangGraph node.

    1. Merge jobs from all three scrapers
    2. Deduplicate
    3. Score each job with the LLM
    4. Sort descending by match_score
    """
    # Combine all scraper outputs
    all_jobs: list[dict] = []
    all_jobs.extend(state.get("linkedin_jobs",  []))
    all_jobs.extend(state.get("indeed_jobs",    []))
    all_jobs.extend(state.get("glassdoor_jobs", []))

    if not all_jobs:
        return {"ranked_jobs": [], "error": "No jobs found from any scraper."}

    unique_jobs = _deduplicate(all_jobs)
    print(f"[Tool 5] {len(all_jobs)} raw → {len(unique_jobs)} after dedup")

    profile = state.get("resume_profile")

    # No profile: assign default score and skip LLM
    if not profile:
        print("[Tool 5] No resume profile — using default score of 50")
        ranked: list[RankedJob] = [
            {**job, "match_score": 50,
             "match_reasoning": "Resume profile not available — default score assigned."}
            for job in unique_jobs
        ]
        return {"ranked_jobs": ranked}

    # Score each job with the LLM
    print(f"[Tool 5] Ranking {len(unique_jobs)} jobs with AI...")
    llm = get_llm(temperature=0.0)
    model_name = getattr(llm, "model_name", "") or ""

    total_in_tok  = 0
    total_out_tok = 0
    total_cost    = 0.0
    ranked = []

    for i, job in enumerate(unique_jobs):
        try:
            prompt = RANK_PROMPT.format(
                name=profile.get("name", "Candidate"),
                skills=", ".join(profile.get("skills", [])),
                years_of_experience=profile.get("years_of_experience", 0),
                job_titles=", ".join(profile.get("job_titles", [])),
                preferred_locations=", ".join(profile.get("preferred_locations", [])),
                summary=profile.get("summary", ""),
                job_title=job.get("title", ""),
                job_company=job.get("company", ""),
                job_location=job.get("location", ""),
                job_description=(job.get("description") or "")[:2000],
            )

            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            score_data = json.loads(raw)

            in_tok  = estimate_tokens(prompt)
            out_tok = estimate_tokens(raw)
            total_in_tok  += in_tok
            total_out_tok += out_tok
            total_cost    += estimate_cost_usd(model_name, in_tok, out_tok)

            ranked_job: RankedJob = {
                **job,
                "match_score":     int(score_data.get("match_score", 0)),
                "match_reasoning": score_data.get("match_reasoning", ""),
            }
            ranked.append(ranked_job)

        except Exception as exc:
            print(f"[Tool 5] Error ranking '{job.get('title', '?')}': {exc}")
            ranked.append({
                **job,
                "match_score":     0,
                "match_reasoning": f"Ranking failed: {exc}",
            })

        if (i + 1) % 5 == 0:
            print(f"[Tool 5] Progress: {i + 1}/{len(unique_jobs)} ranked")

    # Sort best match first
    ranked.sort(key=lambda x: x["match_score"], reverse=True)

    if ranked:
        print(f"[Tool 5] Done. Top: {ranked[0]['match_score']}% — {ranked[0]['title']}")

    return {
        "ranked_jobs": ranked,
        "_telemetry": {
            "rank_jobs": {
                "input_tokens":       total_in_tok,
                "output_tokens":      total_out_tok,
                "estimated_cost_usd": round(total_cost, 8),
            }
        },
    }
