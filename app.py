"""
JobMatch AI — Streamlit UI
============================
Phase 1: Upload resume → Search jobs → AI-rank → Apply
Phase 2: Email top matches (in-UI), Auto-fill (beta)

Run:
    python -m streamlit run app.py --server.port 8080
or:
    bash start.sh
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env relative to this file so it works from any working directory
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# Remove SOCKS proxy that conflicts with httpx
os.environ.pop("ALL_PROXY", None)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JobMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS: tighten sidebar top padding only
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 JobMatch AI")
    st.caption("LangGraph + OpenAI/Claude · Phase 1 & 2")
    st.divider()

    # ── Resume upload ─────────────────────────────────────────────────────────
    st.markdown("### 📄 Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="AI extracts your skills, experience level, job titles, and locations.",
    )

    st.divider()

    # ── Search settings ───────────────────────────────────────────────────────
    st.markdown("### 🔍 Search Settings")

    search_sites = st.multiselect(
        "Job sites",
        options=["linkedin", "indeed", "glassdoor"],
        default=["linkedin", "indeed", "glassdoor"],
    )

    custom_search = st.text_input(
        "Custom search term (optional)",
        placeholder="e.g. Senior Java Engineer",
        help="Leave blank to auto-generate from your resume",
    )

    location = st.text_input(
        "Location",
        value="Bangalore, India",
    )

    work_mode = st.selectbox(
        "Work mode",
        ["Any", "Remote", "Hybrid", "Onsite"],
    )

    experience_level = st.selectbox(
        "Experience level",
        ["Any", "Senior", "Lead", "Staff / Principal", "Mid-level"],
    )

    st.divider()

    # ── LLM indicator ─────────────────────────────────────────────────────────
    provider   = os.environ.get("LLM_PROVIDER", "openai").upper()
    model_name = (
        os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if provider == "OPENAI"
        else os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    )
    has_key = bool(
        os.environ.get("OPENAI_API_KEY")
        if provider == "OPENAI"
        else os.environ.get("ANTHROPIC_API_KEY")
    )
    key_icon = "✅" if has_key else "❌"
    st.markdown(f"**LLM:** {provider} · `{model_name}` {key_icon}")

    st.divider()

    search_clicked = st.button(
        "🚀 Search Jobs",
        use_container_width=True,
        type="primary",
    )


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🎯 JobMatch AI")
st.caption(
    "Upload your resume → let AI scrape & rank jobs from LinkedIn, Indeed, "
    "and Glassdoor → click Apply."
)


# ── Resume profile expander (shown once a search has run) ─────────────────────

if st.session_state.get("resume_profile"):
    profile = st.session_state["resume_profile"]
    with st.expander("📋 Your Parsed Resume Profile", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Name",       profile.get("name", "N/A"))
            st.metric("Experience", f"{profile.get('years_of_experience', '?')} yrs")
            st.metric("Education",  profile.get("education", "N/A"))
        with c2:
            st.markdown("**Job Titles**")
            for t in profile.get("job_titles", []):
                st.markdown(f"- {t}")
        with c3:
            st.markdown("**Top Skills**")
            skills = profile.get("skills", [])
            for s in skills[:12]:
                st.markdown(f"- {s}")
            if len(skills) > 12:
                st.caption(f"…and {len(skills) - 12} more")


# ── Run pipeline on search click ──────────────────────────────────────────────

if search_clicked:
    if not uploaded_file:
        st.error("⚠️ Please upload your resume PDF first (sidebar → Resume).")
        st.stop()
    if not search_sites:
        st.error("⚠️ Please select at least one job site.")
        st.stop()
    if not has_key:
        st.error(
            f"⚠️ No API key found for {provider}. "
            "Check your .env file."
        )
        st.stop()

    # Save uploaded PDF to a temp file the pipeline can read
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    initial_state = {
        "resume_pdf_path":  tmp_path,
        "resume_text":      "",
        "search_sites":     search_sites,
        "search_term":      custom_search.strip() or None,
        "location":         location.strip() or "Bangalore, India",
        "work_mode":        work_mode.lower() if work_mode != "Any" else None,
        "experience_level": experience_level.lower() if experience_level != "Any" else None,
        "resume_profile":   None,
        "linkedin_jobs":    [],
        "indeed_jobs":      [],
        "glassdoor_jobs":   [],
        "ranked_jobs":      [],
        "email_recipient":  None,
        "error":            None,
    }

    with st.status("🔄 Running JobMatch AI pipeline…", expanded=True) as status:
        try:
            from graph.pipeline import run_pipeline

            final_state, obs = run_pipeline(
                initial_state,
                progress_callback=st.write,
            )
        except Exception as exc:
            status.update(label="❌ Pipeline crashed", state="error")
            st.error(f"Pipeline error: {exc}")
            os.unlink(tmp_path)
            st.stop()

        # Clean up temp file
        os.unlink(tmp_path)

        if final_state.get("error"):
            status.update(label=f"⚠️ {final_state['error']}", state="error")
            st.warning(final_state["error"])
            st.stop()

        ranked = final_state.get("ranked_jobs", [])
        st.session_state["ranked_jobs"]    = ranked
        st.session_state["resume_profile"] = final_state.get("resume_profile")
        st.session_state["obs_summary"]    = obs

        status.update(
            label=f"✅ Done! Found and ranked {len(ranked)} jobs.",
            state="complete",
        )


# ── Results ───────────────────────────────────────────────────────────────────

if st.session_state.get("ranked_jobs"):
    ranked_jobs = st.session_state["ranked_jobs"]
    obs         = st.session_state.get("obs_summary", {})

    st.divider()

    # ── Stats row ─────────────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Jobs",       len(ranked_jobs))
    s2.metric("🟢 High Match (70%+)", sum(1 for j in ranked_jobs if j["match_score"] >= 70))
    s3.metric("🟡 Decent (40-69%)",   sum(1 for j in ranked_jobs if 40 <= j["match_score"] < 70))
    s4.metric("Sources",          len(set(j["site"] for j in ranked_jobs)))
    s5.metric("Top Score",        f"{ranked_jobs[0]['match_score']}%" if ranked_jobs else "—")

    # ── Observability panel ───────────────────────────────────────────────────
    if obs:
        with st.expander("📈 Pipeline Metrics", expanded=False):
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Latency",      f"{obs.get('total_latency_ms', 0)} ms")
            o2.metric("Success Rate", f"{obs.get('success_rate_pct', 0)}%")
            o3.metric("Tokens Used",
                      obs.get("total_input_tokens", 0) + obs.get("total_output_tokens", 0))
            o4.metric("Est. Cost",    f"${obs.get('estimated_total_cost_usd', 0.0):.6f}")

    # ── Phase 2: Email Top Matches ────────────────────────────────────────────
    with st.expander("✉️ Email Top Matches (Phase 2)", expanded=False):
        st.caption(
            "Sends an HTML email with the top 5 jobs to any address. "
            "Requires EMAIL_SENDER and EMAIL_PASSWORD (Gmail App Password) in .env"
        )
        email_col1, email_col2 = st.columns([3, 1])
        with email_col1:
            recipient_email = st.text_input(
                "Recipient email",
                placeholder="name@example.com",
                label_visibility="collapsed",
            )
        with email_col2:
            send_email_btn = st.button("Send Top 5 →", type="primary")

        if send_email_btn:
            if not recipient_email.strip():
                st.warning("Enter a recipient email address first.")
            else:
                try:
                    from utils.email_alerts import build_email_bodies, send_gmail_smtp_email
                    text_body, html_body = build_email_bodies(ranked_jobs[:5], obs)
                    send_gmail_smtp_email(
                        subject="JobMatch AI — Top 5 Matches",
                        text_body=text_body,
                        html_body=html_body,
                        recipient=recipient_email.strip(),
                    )
                    st.success(f"✅ Email sent to {recipient_email.strip()}")
                except Exception as exc:
                    st.error(f"Failed to send email: {exc}")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        min_score = st.slider("Min match score", 0, 100, 0, 5)
    with f2:
        site_filter = st.multiselect(
            "Filter by site", ["linkedin", "indeed", "glassdoor"]
        )
    with f3:
        text_filter = st.text_input(
            "Search in results", placeholder="Filter by title or company…"
        )

    # Apply filters
    filtered = ranked_jobs
    if min_score > 0:
        filtered = [j for j in filtered if j["match_score"] >= min_score]
    if site_filter:
        filtered = [j for j in filtered if j["site"] in site_filter]
    if text_filter:
        tl = text_filter.lower()
        filtered = [
            j for j in filtered
            if tl in j["title"].lower() or tl in j["company"].lower()
        ]

    st.markdown(f"**Showing {len(filtered)} of {len(ranked_jobs)} jobs**")

    # ── Job cards ─────────────────────────────────────────────────────────────
    for idx, job in enumerate(filtered):
        score = job["match_score"]

        if score >= 70:
            badge = "🟢"
            badge_label = "Excellent match"
        elif score >= 40:
            badge = "🟡"
            badge_label = "Good match"
        else:
            badge = "🔴"
            badge_label = "Weak match"

        site_name   = (job.get("site") or "unknown").capitalize()
        job_url     = job.get("job_url", "")
        valid_url   = job_url and str(job_url) not in ("", "nan", "None")
        date_posted = job.get("date_posted", "")
        show_date   = date_posted and str(date_posted) not in ("", "nan", "None")

        with st.container(border=True):
            # ── Header row ────────────────────────────────────────────────────
            head1, head2, head3 = st.columns([1, 4, 1])

            with head1:
                st.markdown(f"## {badge}")
                st.markdown(f"**{score}%**")
                st.caption(badge_label)

            with head2:
                st.markdown(f"### {job['title']}")
                st.markdown(f"🏢 **{job['company']}**  &nbsp;&nbsp; 📍 {job['location']}")

            with head3:
                st.markdown(f"**{site_name}**")
                if show_date:
                    st.caption(f"Posted: {date_posted}")

            # ── AI Reasoning ──────────────────────────────────────────────────
            reasoning = (job.get("match_reasoning") or "").strip()
            if reasoning:
                st.info(f"🤖 {reasoning}")

            # ── Actions ───────────────────────────────────────────────────────
            if valid_url:
                act1, act2 = st.columns([2, 2])
                with act1:
                    st.link_button(
                        f"✅ Apply on {site_name}  →",
                        url=job_url,
                        type="primary",
                        use_container_width=True,
                    )
                with act2:
                    autofill_key = f"af_{idx}_{abs(hash(job_url)) % 999999}"
                    if st.button(
                        "⚡ Auto-fill (beta)",
                        key=autofill_key,
                        use_container_width=True,
                    ):
                        with st.spinner("Attempting auto-fill…"):
                            try:
                                from utils.autofill import attempt_autofill
                                result = attempt_autofill(
                                    job_url,
                                    st.session_state.get("resume_profile"),
                                )
                            except Exception as exc:
                                result = {
                                    "success": False,
                                    "message": str(exc),
                                    "fallback_url": job_url,
                                }
                        if result.get("success"):
                            st.success(result["message"])
                        else:
                            st.warning(result["message"])
            else:
                st.caption("ℹ️ No direct application link available for this listing.")

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Ready to find your next role?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **How it works:**
            1. 📄 Upload your resume PDF (sidebar)
            2. 🔍 Choose job sites and location
            3. 🚀 Click **Search Jobs**
            4. 🤖 AI parses your resume and ranks every job
            5. ✅ Click **Apply** directly from the results
            """
        )
    with col2:
        st.markdown(
            """
            **Phase 2 features:**
            - ✉️ **Email alerts** — send top matches to any email
            - ⚡ **Auto-fill (beta)** — pre-fill application forms
            - 📈 **Pipeline metrics** — tokens, cost, latency
            - 🕐 **Daily scheduler** — `python run_email_scheduler.py`
            """
        )
