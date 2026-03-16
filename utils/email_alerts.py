"""
Phase 2 — Tool 6: Email Alerts
================================
Builds HTML + plain-text email bodies from the top-ranked jobs
and sends them via Gmail SMTP (App Password required).

Setup in .env:
    EMAIL_SENDER=you@gmail.com
    EMAIL_PASSWORD=xxxx xxxx xxxx xxxx   # Gmail App Password
    EMAIL_RECIPIENT=recipient@example.com  # Default recipient

Gmail App Password: https://myaccount.google.com/apppasswords
(Enable 2FA first, then generate an App Password for "Mail")

Usage (from Streamlit UI):
    from utils.email_alerts import build_email_bodies, send_gmail_smtp_email
    text_body, html_body = build_email_bodies(top_jobs, obs_summary)
    send_gmail_smtp_email("JobMatch AI - Top Matches", text_body, html_body,
                          recipient="someone@example.com")

Usage (for daily scheduled runs):
    from utils.email_alerts import run_daily_alert
    run_daily_alert(initial_state, recipient="someone@example.com")
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from graph.pipeline import run_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _required_env(name: str) -> str:
    """Read a required env var; raise ValueError if missing or blank."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(
            f"Missing required env var: {name}\n"
            "Check your .env file."
        )
    return value


# ── Email body builder ────────────────────────────────────────────────────────

def build_email_bodies(
    top_jobs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[str, str]:
    """
    Build plain-text and HTML email bodies from top-ranked jobs.

    Args:
        top_jobs:  List of RankedJob dicts (usually the top 5).
        summary:   Observability summary dict from PipelineObserver.

    Returns:
        (text_body, html_body) — both ready to be sent as email parts.
    """
    success_rate = summary.get("success_rate_pct", 0)
    latency_ms   = summary.get("total_latency_ms", 0)
    cost_usd     = summary.get("estimated_total_cost_usd", 0.0)

    # Plain text
    text_lines = [
        "JobMatch AI — Daily Top Matches",
        "=" * 40,
        f"Pipeline success rate : {success_rate}%",
        f"Pipeline latency      : {latency_ms} ms",
        f"Estimated LLM cost    : ${cost_usd}",
        "",
        "TOP MATCHES:",
        "",
    ]
    for i, job in enumerate(top_jobs, start=1):
        line = (
            f"{i}. {job.get('title', 'N/A')} @ {job.get('company', 'N/A')}"
            f" — {job.get('location', 'N/A')} — Score: {job.get('match_score', 0)}%"
        )
        if job.get("job_url") and str(job["job_url"]) != "nan":
            line += f"\n   Apply: {job['job_url']}"
        if job.get("match_reasoning"):
            line += f"\n   {job['match_reasoning']}"
        text_lines.append(line)
        text_lines.append("")

    # HTML
    job_items_html = ""
    for i, job in enumerate(top_jobs, start=1):
        score      = job.get("match_score", 0)
        score_color = "#22c55e" if score >= 70 else ("#f59e0b" if score >= 40 else "#ef4444")
        url        = job.get("job_url", "")
        apply_link = (
            f'<a href="{url}" style="color:#2563eb;">Apply Now &rarr;</a>'
            if url and str(url) != "nan"
            else "<em>No link available</em>"
        )
        job_items_html += f"""
        <li style="margin-bottom:20px;padding:14px;border:1px solid #e5e7eb;
                   border-radius:8px;list-style:none;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:16px;">{job.get("title","N/A")}</strong>
            <span style="background:{score_color};color:#fff;padding:3px 10px;
                         border-radius:12px;font-size:13px;">{score}%</span>
          </div>
          <div style="color:#6b7280;margin-top:4px;">
            {job.get("company","N/A")} &bull; {job.get("location","N/A")}
          </div>
          <div style="margin-top:8px;font-size:14px;color:#374151;">
            {job.get("match_reasoning","")}
          </div>
          <div style="margin-top:10px;">{apply_link}</div>
        </li>
        """

    html_body = f"""
    <html>
    <body style="font-family:Arial,Helvetica,sans-serif;max-width:640px;margin:0 auto;padding:20px;">
      <h2 style="color:#1e40af;">🎯 JobMatch AI — Daily Top Matches</h2>

      <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
        <tr>
          <td style="padding:8px;background:#f3f4f6;border-radius:4px;">
            ✅ Success rate: <strong>{success_rate}%</strong>
          </td>
          <td style="padding:8px;background:#f3f4f6;border-radius:4px;">
            ⏱ Latency: <strong>{latency_ms} ms</strong>
          </td>
          <td style="padding:8px;background:#f3f4f6;border-radius:4px;">
            💵 Est. cost: <strong>${cost_usd}</strong>
          </td>
        </tr>
      </table>

      <h3 style="color:#374151;">Top {len(top_jobs)} Matches</h3>
      <ul style="padding:0;">
        {job_items_html}
      </ul>

      <p style="color:#9ca3af;font-size:12px;margin-top:30px;border-top:1px solid #e5e7eb;
                padding-top:10px;">
        Sent by JobMatch AI &bull; Powered by LangGraph + OpenAI
      </p>
    </body>
    </html>
    """

    return "\n".join(text_lines), html_body


# ── SMTP sender ───────────────────────────────────────────────────────────────

def send_gmail_smtp_email(
    subject: str,
    text_body: str,
    html_body: str,
    recipient: str | None = None,
) -> None:
    """
    Send a multipart HTML+text email via Gmail SMTP (SSL, port 465).

    Requires in .env:
        EMAIL_SENDER    — your Gmail address
        EMAIL_PASSWORD  — Gmail App Password (NOT your regular password)

    Args:
        subject:    Email subject line.
        text_body:  Plain-text fallback.
        html_body:  HTML version.
        recipient:  Destination address; falls back to EMAIL_RECIPIENT env var.
    """
    sender   = _required_env("EMAIL_SENDER")
    password = _required_env("EMAIL_PASSWORD")

    to_addr = (recipient or os.environ.get("EMAIL_RECIPIENT", "")).strip()
    if not to_addr:
        raise ValueError(
            "No recipient email address provided.\n"
            "Pass recipient= or set EMAIL_RECIPIENT in .env"
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = to_addr

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html",  "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, to_addr, msg.as_string())

    print(f"[Email] Sent to {to_addr}: {subject}")


# ── Daily alert runner (used by the scheduler) ────────────────────────────────

def run_daily_alert(
    initial_state: dict[str, Any],
    recipient: str | None = None,
) -> dict[str, Any]:
    """
    Run the full pipeline and email top-5 ranked jobs.

    Used by run_email_scheduler.py for scheduled daily alerts.

    Returns:
        {"success": bool, "sent_jobs": int, "summary": dict, "error": str|None}
    """
    final_state, summary = run_pipeline(initial_state)

    ranked = final_state.get("ranked_jobs", [])
    top5   = ranked[:5]

    if not top5:
        return {
            "success": False,
            "error":   final_state.get("error", "No jobs found for daily alert."),
            "summary": summary,
            "sent_jobs": 0,
        }

    text_body, html_body = build_email_bodies(top5, summary)
    send_gmail_smtp_email(
        subject="JobMatch AI — Daily Top 5 Matches",
        text_body=text_body,
        html_body=html_body,
        recipient=recipient,
    )

    return {
        "success":   True,
        "sent_jobs": len(top5),
        "summary":   summary,
        "error":     None,
    }
