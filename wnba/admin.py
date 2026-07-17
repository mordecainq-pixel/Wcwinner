"""Admin gate for the public deployment: unlocks a settings panel and the
Bet Builder tab (kept off the public site -- see betbuilder.py's cost note)
for whoever knows the password.

IP-based gating was tried first and abandoned after confirming live that
Streamlit Community Cloud proxies requests through a pool of internal
servers -- the detected "visitor IP" changed on every single page reload,
not just between sessions, so there was never a stable address to check
against. Replaced with a hidden URL route (this deployment's
`/adminlogs`, wired up via st.Page/st.navigation in app.py) plus a
password -- doesn't depend on any network/hosting detail, so it can't
break the same way.

Unlock state lives in st.session_state, which is per-browser-session (one
WebSocket connection) -- a hard page reload or a new tab starts a fresh
session, so the password needs re-entering at /adminlogs after either of
those. That's a real tradeoff of not persisting anything (a cookie, a
long-lived token) -- traded deliberately for simplicity and for not
storing a credential anywhere longer than the current tab needs it.
"""
from __future__ import annotations

import requests
import streamlit as st

from wnba.config import ADMIN_GITHUB_TOKEN, ADMIN_PASSWORD, GITHUB_REPO, GITHUB_WORKFLOW_FILE, GITHUB_WORKFLOW_REF


def is_admin() -> bool:
    return bool(st.session_state.get("admin_unlocked"))


def render_admin_login_page(main_page) -> None:
    """The entire content of the hidden /adminlogs route. Deliberately
    shows nothing but a password box -- no hint of what's behind it -- so
    stumbling onto the URL without the password reveals nothing.
    """
    st.title("Admin")

    if is_admin():
        st.success("Already unlocked for this session.")
        if st.button("Go to site"):
            st.switch_page(main_page)
        return

    if not ADMIN_PASSWORD:
        st.error("ADMIN_PASSWORD is not configured on this deployment.")
        return

    password = st.text_input("Password", type="password", key="admin_login_password")
    if st.button("Unlock", type="primary"):
        if password and password == ADMIN_PASSWORD:
            st.session_state["admin_unlocked"] = True
            st.switch_page(main_page)
        else:
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
