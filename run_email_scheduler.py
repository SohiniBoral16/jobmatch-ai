"""
Phase 2 — Daily Email Scheduler
==================================
Runs the full JobMatch AI pipeline on a cron schedule and emails
the top-5 matched jobs to a configured recipient.

Usage:
    # Run once immediately (good for testing):
    python run_email_scheduler.py --resume /abs/path/to/resume.pdf --run-once

    # Run every day at 07:00 IST:
    python run_email_scheduler.py --resume /abs/path/to/resume.pdf

Configure in .env:
    EMAIL_SENDER=you@gmail.com
    EMAIL_PASSWORD=xxxx xxxx xxxx xxxx      # Gmail App Password
    EMAIL_RECIPIENT=recipient@example.com
    ALERT_HOUR=7
    ALERT_MINUTE=0
    ALERT_TIMEZONE=Asia/Kolkata
    ALERT_LOCATION=Bangalore, India
    ALERT_SEARCH_TERM=                       # Optional override
    ALERT_WORK_MODE=                         # Optional: remote/onsite/hybrid
    ALERT_EXPERIENCE_LEVEL=                  # Optional: senior/lead/mid

Dependencies:
    pip install apscheduler
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from utils.email_alerts import run_daily_alert


# ── State builder ─────────────────────────────────────────────────────────────

def _build_state(resume_pdf_path: str) -> dict:
    """Build the initial GraphState for a scheduled pipeline run."""
    return {
        "resume_pdf_path": resume_pdf_path,
        "resume_text":     "",
        "search_sites":    ["linkedin", "indeed", "glassdoor"],
        "search_term":     os.environ.get("ALERT_SEARCH_TERM") or None,
        "location":        os.environ.get("ALERT_LOCATION") or "Bangalore, India",
        "work_mode":       (os.environ.get("ALERT_WORK_MODE") or "").lower() or None,
        "experience_level":(os.environ.get("ALERT_EXPERIENCE_LEVEL") or "").lower() or None,
        "resume_profile":  None,
        "linkedin_jobs":   [],
        "indeed_jobs":     [],
        "glassdoor_jobs":  [],
        "ranked_jobs":     [],
        "email_recipient": None,
        "error":           None,
    }


# ── Job runner ────────────────────────────────────────────────────────────────

def run_job(resume_pdf_path: str) -> None:
    """Called by the scheduler on each trigger."""
    print(f"[Scheduler] Starting daily alert run for: {resume_pdf_path}")
    result = run_daily_alert(_build_state(resume_pdf_path))

    if result.get("success"):
        print(f"[Scheduler] ✅ Email sent — {result.get('sent_jobs', 0)} jobs included")
        summary = result.get("summary", {})
        print(f"[Scheduler]    Latency: {summary.get('total_latency_ms', 0)} ms  "
              f"Cost: ${summary.get('estimated_total_cost_usd', 0.0)}")
    else:
        print(f"[Scheduler] ❌ Failed: {result.get('error', 'unknown error')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

    parser = argparse.ArgumentParser(
        description="JobMatch AI — Daily Email Alert Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test run (once, immediately):
    python run_email_scheduler.py --resume /path/to/resume.pdf --run-once

  Start daily cron (default 07:00 IST):
    python run_email_scheduler.py --resume /path/to/resume.pdf
        """,
    )
    parser.add_argument(
        "--resume", required=True,
        help="Absolute path to the resume PDF",
    )
    parser.add_argument(
        "--run-once", action="store_true",
        help="Run immediately once and exit (useful for testing)",
    )
    args = parser.parse_args()

    resume_path = Path(args.resume).expanduser().resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found: {resume_path}")

    if args.run_once:
        run_job(str(resume_path))
        return

    # Schedule recurring daily run
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        raise ImportError(
            "APScheduler is not installed.\n"
            "Run: pip install apscheduler"
        )

    tz     = os.environ.get("ALERT_TIMEZONE", "Asia/Kolkata")
    hour   = int(os.environ.get("ALERT_HOUR",   "7"))
    minute = int(os.environ.get("ALERT_MINUTE", "0"))

    scheduler = BlockingScheduler(timezone=tz)
    scheduler.add_job(
        run_job, "cron",
        hour=hour, minute=minute,
        args=[str(resume_path)],
    )

    print(f"[Scheduler] Daily alert scheduled at {hour:02d}:{minute:02d} {tz}")
    print(f"[Scheduler] Press Ctrl+C to stop")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("[Scheduler] Stopped.")


if __name__ == "__main__":
    main()
