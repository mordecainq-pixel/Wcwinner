"""Admin gate for the public deployment: unlocks a settings panel and the
Bet Builder tab (kept off the public site -- see betbuilder.py's cost note)
for whoever's IP is on the allowlist.

IP-only, by design choice: no password fallback. That means a local
`streamlit run wnba/app.py` will never show as admin -- `st.context.
ip_address` is `127.0.0.1` for any local connection, not your public IP --
and if your IP ever changes (ISP reassignment, different network), you'll
need to update ADMIN_ALLOWED_IPS on the deployment before the admin panel
unlocks again. That tradeoff was chosen deliberately over keeping a
password around as a second way in.
"""
from __future__ import annotations

import requests
import streamlit as st

from wnba.config import ADMIN_ALLOWED_IPS, ADMIN_GITHUB_TOKEN, GITHUB_REPO, GITHUB_WORKFLOW_FILE, GITHUB_WORKFLOW_REF


def _visitor_ip() -> str | None:
    try:
        return st.context.ip_address
    except Exception:
        return None


def is_admin() -> bool:
    ip = _visitor_ip()
    return bool(ip and ip in ADMIN_ALLOWED_IPS)


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
