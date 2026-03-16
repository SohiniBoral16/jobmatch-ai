"""
LangGraph Shared State — flows through all pipeline nodes.

GraphState is the single source of truth for the entire pipeline run.
Every node reads from it and returns a partial dict that gets merged back in.
"""

from typing import Optional, TypedDict


class ResumeProfile(TypedDict):
    """Structured profile extracted from the candidate's resume by the LLM."""
    name: str
    skills: list[str]
    years_of_experience: int
    job_titles: list[str]
    preferred_locations: list[str]
    education: str
    summary: str


class JobListing(TypedDict):
    """A single raw job listing returned by any scraper node."""
    title: str
    company: str
    location: str
    description: str
    job_url: str
    site: str          # "linkedin" | "indeed" | "glassdoor"
    date_posted: str


class RankedJob(TypedDict):
    """A job listing enriched with AI match score + reasoning."""
    title: str
    company: str
    location: str
    description: str
    job_url: str
    site: str
    date_posted: str
    match_score: int         # 0-100
    match_reasoning: str     # 2-3 sentence AI explanation


class GraphState(TypedDict):
    """
    The complete state that flows through the LangGraph pipeline.

    Phase 1 fields (always used):
        resume_pdf_path, resume_text, search_sites, search_term,
        location, work_mode, experience_level,
        resume_profile, linkedin_jobs, indeed_jobs, glassdoor_jobs,
        ranked_jobs, error

    Phase 2 fields (used by email alerts and scheduler):
        email_recipient  -- override recipient for this run
    """

    # Inputs
    resume_pdf_path: str
    resume_text: str
    search_sites: list[str]         # ["linkedin", "indeed", "glassdoor"]
    search_term: Optional[str]      # Custom term; None = auto-generate from resume
    location: Optional[str]         # e.g. "Bangalore, India"
    work_mode: Optional[str]        # "remote" | "onsite" | "hybrid" | None
    experience_level: Optional[str] # "senior" | "lead" | "mid" | None

    # Tool 1 output
    resume_profile: Optional[ResumeProfile]

    # Tools 2/3/4 output
    linkedin_jobs: list[JobListing]
    indeed_jobs: list[JobListing]
    glassdoor_jobs: list[JobListing]

    # Tool 5 output
    ranked_jobs: list[RankedJob]

    # Phase 2
    email_recipient: Optional[str]  # Override for daily alert emails

    # Metadata
    error: Optional[str]            # Non-None means pipeline should stop
