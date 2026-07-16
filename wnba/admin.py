"""Admin gate for the public deployment: unlocks a settings panel and the
Bet Builder tab (kept off the public site -- see betbuilder.py's cost note)
for whoever configured this deployment.

IMPORTANT: IP matching is a convenience, not real security. `st.context.
ip_address` reflects whatever address reached the app -- correct for a
visitor coming in over the public internet to a hosted deployment, but on a
LOCAL run (`streamlit run wnba/app.py` on your own machine) it's
`127.0.0.1`, not your public IP, so IP matching alone would never unlock
locally. A password fallback (`ADMIN_PASSWORD`, set via `.env` locally or
the host's secrets manager) covers that case and works everywhere. Treat
the password as a real credential -- anyone who has it gets admin access
regardless of IP.
"""
from __future__ import annotations

import requests
import streamlit as st

from wnba.config import ADMIN_ALLOWED_IPS, ADMIN_GITHUB_TOKEN, ADMIN_PASSWORD, GITHUB_REPO, GITHUB_WORKFLOW_FILE, GITHUB_WORKFLOW_REF


def _visitor_ip() -> str | None:
    try:
        return st.context.ip_address
    except Exception:
        return None


def is_admin() -> bool:
    if st.session_state.get("admin_unlocked"):
        return True

    ip = _visitor_ip()
    if ip and ip in ADMIN_ALLOWED_IPS:
        st.session_state["admin_unlocked"] = True
        return True

    return False


def render_admin_login() -> None:
    """A small, unobtrusive password entry for admin access when IP
    matching doesn't apply (e.g. running locally). Call this once, low on
    the page -- it does nothing visible for non-admins who don't open it.
    """
    if is_admin():
        return
    with st.expander("Admin login"):
        if not ADMIN_PASSWORD:
            st.caption("No ADMIN_PASSWORD configured for this deployment.")
            return
        entered = st.text_input("Password", type="password", key="admin_password_input")
        if entered and entered == ADMIN_PASSWORD:
            st.session_state["admin_unlocked"] = True
            st.rerun()
        elif entered:
            st.error("Incorrect password.")


def trigger_refresh_workflow() -> tuple[bool, str]:
    """Dispatches the same GitHub Actions workflow the daily cron runs,
    rather than running the (slow, one-request-per-game) refresh inside the
    web app's own process -- keeps this a thin trigger, not a duplicate
    implementation. Requires ADMIN_GITHUB_TOKEN (a GitHub personal access
    token with `repo` or `actions:write` scope) to be configured.
    """
    if not ADMIN_GITHUB_TOKEN:
        return False, "ADMIN_GITHUB_TOKEN is not configured on this deployment -- can't trigger GitHub Actions from here."

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW_FILE}/dispatches"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {ADMIN_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        json={"ref": GITHUB_WORKFLOW_REF},
        timeout=15,
    )
    if resp.status_code == 204:
        return True, "Refresh triggered. It runs on GitHub's servers (~15-25 min), then this site redeploys automatically once it's done."
    return False, f"GitHub API returned {resp.status_code}: {resp.text[:200]}"
