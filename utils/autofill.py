"""
Phase 2 — Auto-fill (Beta)
============================
Best-effort Selenium-based form filler for LinkedIn and Indeed apply pages.

What it does:
  - Opens the job URL in a headless Chrome browser
  - Finds common input fields (name, email, phone, LinkedIn, location)
  - Fills them from CANDIDATE_* env vars or the parsed resume profile
  - Attempts to click a "Next / Continue / Review" button

What it does NOT do:
  - Handle multi-step application wizards reliably
  - Upload resumes or cover letters automatically
  - Submit the final form (intentionally — user must review first)

If Selenium is not installed, or the site is not supported, it falls back
gracefully by returning success=False with the original job URL so the user
can apply manually.

Setup (optional — only needed for auto-fill):
    pip install selenium webdriver-manager
    # Chrome must be installed on the system

Configure in .env:
    CANDIDATE_NAME=Sohini Boral
    CANDIDATE_EMAIL=your@email.com
    CANDIDATE_PHONE=+91 98765 43210
    CANDIDATE_LINKEDIN=https://linkedin.com/in/yourusername
    CANDIDATE_LOCATION=Bangalore, India
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse


# ── Candidate defaults ────────────────────────────────────────────────────────

def _candidate_defaults(resume_profile: dict[str, Any] | None = None) -> dict[str, str]:
    """
    Build a dict of candidate contact details from env vars (preferred)
    or from the parsed resume profile (fallback).
    """
    profile = resume_profile or {}
    return {
        "name":     os.environ.get("CANDIDATE_NAME")     or profile.get("name", ""),
        "email":    os.environ.get("CANDIDATE_EMAIL",    ""),
        "phone":    os.environ.get("CANDIDATE_PHONE",    ""),
        "linkedin": os.environ.get("CANDIDATE_LINKEDIN", ""),
        "location": os.environ.get("CANDIDATE_LOCATION") or
                    (profile.get("preferred_locations") or [""])[0],
    }


# ── Site support check ────────────────────────────────────────────────────────

def _is_supported(job_url: str) -> bool:
    """Return True only for sites where auto-fill has been tested."""
    try:
        netloc = urlparse(job_url).netloc.lower()
    except Exception:
        return False
    return "linkedin.com" in netloc or "indeed.com" in netloc


# ── Core auto-fill ────────────────────────────────────────────────────────────

def attempt_autofill(
    job_url: str,
    resume_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Attempt to auto-fill a job application form.

    Args:
        job_url:        The job application URL.
        resume_profile: Parsed resume profile dict (used as fallback for candidate info).

    Returns:
        {
            "success":      bool,
            "message":      str,   # User-facing status message
            "fallback_url": str,   # The job URL (always provided for manual fallback)
        }
    """
    if not job_url or str(job_url) == "nan":
        return {
            "success":      False,
            "message":      "Missing job URL.",
            "fallback_url": None,
        }

    if not _is_supported(job_url):
        return {
            "success":      False,
            "message":      "Auto-fill is not supported for this site yet. "
                            "Please click Apply to continue manually.",
            "fallback_url": job_url,
        }

    # Try to import Selenium — gracefully degrade if not installed
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        return {
            "success":      False,
            "message":      "Selenium is not installed. "
                            "Run: pip install selenium webdriver-manager\n"
                            "Falling back to manual apply.",
            "fallback_url": job_url,
        }

    candidate = _candidate_defaults(resume_profile)
    options   = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")

    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver  = webdriver.Chrome(service=service, options=options)
        driver.get(job_url)

        wait   = WebDriverWait(driver, 8)
        fields = driver.find_elements(By.CSS_SELECTOR, "input, textarea")

        for field in fields:
            ftype = (field.get_attribute("type") or "").lower()
            hint  = " ".join([
                field.get_attribute("name")        or "",
                field.get_attribute("id")          or "",
                field.get_attribute("aria-label")  or "",
                field.get_attribute("placeholder") or "",
            ]).lower()

            try:
                if "email" in hint or ftype == "email":
                    field.clear(); field.send_keys(candidate["email"])
                elif "phone" in hint or "mobile" in hint or ftype == "tel":
                    field.clear(); field.send_keys(candidate["phone"])
                elif ("first" in hint and "name" in hint) or hint.strip() == "name":
                    field.clear(); field.send_keys(candidate["name"])
                elif "linkedin" in hint:
                    field.clear(); field.send_keys(candidate["linkedin"])
                elif "location" in hint or "city" in hint:
                    field.clear(); field.send_keys(candidate["location"])
            except Exception:
                continue  # Skip fields we can't fill

        # Try clicking a "Next / Continue / Review" button
        for btn_text in ["next", "continue", "review", "submit"]:
            xpath   = (
                f"//button[contains(translate(., "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{btn_text}')]"
            )
            buttons = driver.find_elements(By.XPATH, xpath)
            if buttons:
                try:
                    wait.until(EC.element_to_be_clickable(buttons[0]))
                    buttons[0].click()
                    break
                except Exception:
                    continue

        return {
            "success":      True,
            "message":      "Auto-fill attempted! Please review the form before submitting.",
            "fallback_url": job_url,
        }

    except Exception as exc:
        return {
            "success":      False,
            "message":      f"Auto-fill encountered an error: {exc}\n"
                            "Please continue manually.",
            "fallback_url": job_url,
        }
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
