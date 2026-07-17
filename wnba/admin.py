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

CONFIRMED LIVE: on Streamlit Community Cloud, `st.context.ip_address`
returns `::ffff:127.0.0.1` (IPv6 notation for localhost) for every
visitor, not their real address -- the app sits behind an internal reverse
proxy, and the raw connection Streamlit sees is that proxy's own loopback
connection, not the original client's. So `ip_address` alone is useless for
this deployment. The fix is the standard one for exactly this situation:
reverse proxies conventionally record the real client IP in the
`X-Forwarded-For` header (the first address if there's a chain of proxies),
so that's checked first, with `ip_address` only as a fallback for hosts
that don't sit behind a proxy at all (e.g. if this ever runs somewhere
else). Not verified against Streamlit Cloud's exact header behavior from
here (no way to reach the real deployment from this environment) -- if
admin access still doesn't unlock after this, the header may need a
different name or index, not necessarily "first entry."
"""
from __future__ import annotations

import requests
import streamlit as st

from wnba.config import ADMIN_ALLOWED_IPS, ADMIN_GITHUB_TOKEN, GITHUB_REPO, GITHUB_WORKFLOW_FILE, GITHUB_WORKFLOW_REF


def _visitor_ip() -> str | None:
    try:
        forwarded = st.context.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return st.context.ip_address
    except Exception:
        return None


def is_admin() -> bool:
    ip = _visitor_ip()
    return bool(ip and ip in ADMIN_ALLOWED_IPS)


def render_ip_diagnostic() -> None:
    """Debug view of the detected IP, gated behind a `?debug=1` URL param
    instead of always showing -- regular visitors never see this; you can
    pull it up on demand at yoursite.streamlit.app/?debug=1 if admin access
    isn't unlocking and you need to see what's actually being detected.
    """
    if st.query_params.get("debug") != "1":
        return
    st.caption(f"[debug] Detected visitor IP: `{_visitor_ip() or 'unavailable'}` -- admin: {is_admin()}")


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
