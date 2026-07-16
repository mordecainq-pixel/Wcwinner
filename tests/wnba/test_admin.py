import wnba.admin as admin


def test_trigger_refresh_workflow_fails_cleanly_without_token(monkeypatch):
    monkeypatch.setattr(admin, "ADMIN_GITHUB_TOKEN", "")
    ok, message = admin.trigger_refresh_workflow()
    assert ok is False
    assert "not configured" in message


def test_trigger_refresh_workflow_success(monkeypatch):
    monkeypatch.setattr(admin, "ADMIN_GITHUB_TOKEN", "fake-token")

    class _Resp:
        status_code = 204
        text = ""

    def fake_post(url, headers, json, timeout):
        assert "dispatches" in url
        assert headers["Authorization"] == "Bearer fake-token"
        assert json["ref"]
        return _Resp()

    monkeypatch.setattr(admin.requests, "post", fake_post)
    ok, message = admin.trigger_refresh_workflow()
    assert ok is True
    assert "triggered" in message.lower()


def test_trigger_refresh_workflow_reports_api_error(monkeypatch):
    monkeypatch.setattr(admin, "ADMIN_GITHUB_TOKEN", "fake-token")

    class _Resp:
        status_code = 404
        text = "Not Found"

    monkeypatch.setattr(admin.requests, "post", lambda *a, **k: _Resp())
    ok, message = admin.trigger_refresh_workflow()
    assert ok is False
    assert "404" in message
