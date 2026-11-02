from datetime import date

import pytest

from ticketpilot.config import Settings, get_settings


def test_defaults_match_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.AUTO_APPROVE_REFUND_LIMIT == 100
    assert s.CONFIDENCE_THRESHOLD == 0.80
    assert s.MAX_TOOL_STEPS == 4
    assert s.RETRIEVAL_TOP_K == 5
    assert s.RRF_K == 60
    assert s.RETRIEVAL_MODE == "hybrid"
    assert s.VALIDATION_ENABLED is True
    assert s.sensitive_categories == {"other_sensitive"}
    assert s.REFERENCE_DATE == date(2026, 10, 1)
    assert s.QDRANT_COLLECTION == "kestrel_kb"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("VALIDATION_ENABLED", "false")
    monkeypatch.setenv("SENSITIVE_CATEGORIES", "other_sensitive, account_billing")
    monkeypatch.setenv("REFERENCE_DATE", "2026-01-15")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.CONFIDENCE_THRESHOLD == 0.65
    assert s.RETRIEVAL_MODE == "bm25"
    assert s.VALIDATION_ENABLED is False
    assert s.sensitive_categories == {"other_sensitive", "account_billing"}
    assert s.REFERENCE_DATE == date(2026, 1, 15)


def test_empty_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PRICE_INPUT_PER_MTOK", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.AGENT_PRICE_INPUT_PER_MTOK is None
    assert s.ANTHROPIC_API_KEY is None


def test_invalid_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "magic")
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_relative_paths_resolve_to_project_root() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.business_db_path.is_absolute()
    assert s.business_db_path.name == "ticketpilot.db"
