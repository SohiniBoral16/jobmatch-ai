"""
Tool 1: Resume Parser
======================
Reads the uploaded PDF resume, extracts raw text with pdfplumber,
then calls the configured LLM (OpenAI or Claude) to return a clean
structured JSON profile: skills, experience, job titles, locations.

Returns:
    resume_text      — raw extracted string
    resume_profile   — dict matching ResumeProfile TypedDict
    _telemetry       — token/cost tracking for observability
"""

import json

import pdfplumber
from langchain_core.messages import HumanMessage

from graph.state import GraphState
from utils.llm import get_llm
from utils.observability import estimate_cost_usd, estimate_tokens


# ── LLM prompt ───────────────────────────────────────────────────────────────

RESUME_PARSE_PROMPT = """\
You are a professional resume parser. Read the resume below and extract a \
structured JSON profile.

Return ONLY valid JSON — no markdown, no code fences, no extra text.
Use exactly this structure:
{{
    "name": "Full name of the candidate",
    "skills": ["skill1", "skill2", ...],
    "years_of_experience": <integer>,
    "job_titles": ["Most recent title", "Previous title", ...],
    "preferred_locations": ["City, Country", ...],
    "education": "Highest degree and institution",
    "summary": "2-3 sentence professional summary"
}}

Rules:
- skills: include both technical (languages, frameworks, tools) and domain skills
- years_of_experience: integer; count from the earliest job year to today
- job_titles: most recent title first
- preferred_locations: infer from where the candidate has worked
- Return ONLY the JSON object

RESUME:
{resume_text}
"""


# ── PDF text extractor ───────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


# ── LangGraph node ───────────────────────────────────────────────────────────

def parse_resume(state: GraphState) -> dict:
    """
    Tool 1 — LangGraph node.

    1. Extract text from the PDF at state['resume_pdf_path']
    2. Send it to the LLM for structured JSON extraction
    3. Return the parsed profile + telemetry

    Returns a partial state dict merged back into GraphState.
    """
    pdf_path = state.get("resume_pdf_path", "")

    # Step 1 — PDF text extraction
    try:
        if pdf_path:
            resume_text = extract_text_from_pdf(pdf_path)
        else:
            # Fallback: raw text was already provided (e.g. in tests)
            resume_text = state.get("resume_text", "")

        if not resume_text.strip():
            return {"error": "Could not extract any text from the resume PDF. "
                             "Make sure it's not a scanned image-only PDF."}

    except Exception as exc:
        return {"error": f"Failed to read PDF: {exc}"}

    # Step 2 — LLM structured extraction
    try:
        llm = get_llm(temperature=0.0)
        prompt = RESUME_PARSE_PROMPT.format(resume_text=resume_text)

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if the LLM added them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        profile = json.loads(raw)

    except json.JSONDecodeError as exc:
        return {"error": f"LLM returned invalid JSON: {exc}\nRaw output: {raw[:400]}"}
    except Exception as exc:
        return {"error": f"LLM call failed in resume parser: {exc}"}

    # Telemetry
    model_name = getattr(llm, "model_name", "") or ""
    in_tok  = estimate_tokens(prompt)
    out_tok = estimate_tokens(raw)
    cost    = estimate_cost_usd(model_name, in_tok, out_tok)

    return {
        "resume_text": resume_text,
        "resume_profile": profile,
        "_telemetry": {
            "parse_resume": {
                "input_tokens":        in_tok,
                "output_tokens":       out_tok,
                "estimated_cost_usd":  cost,
            }
        },
    }
