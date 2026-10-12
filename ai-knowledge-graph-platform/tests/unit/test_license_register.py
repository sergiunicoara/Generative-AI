"""Model licence register: every configured model must be covered by an allowed-licence entry."""
from __future__ import annotations

from graphrag.core.license_register import check, find_model_ids, load_register

REG = {"allowed_licences": ["apache-2.0", "provider-terms"],
       "models": [{"pattern": "gpt-oss-*", "licence": "apache-2.0", "review_status": "reviewed"},
                  {"pattern": "bad-*", "licence": "proprietary", "review_status": "needs_review"},
                  {"pattern": "unused-*", "licence": "apache-2.0", "review_status": "reviewed"}]}


def test_repository_models_are_all_registered_with_allowed_licences():
    result = check(load_register(), find_model_ids())
    assert result["unregistered"] == [], "add a row to config/licenses.yml for each new model"
    assert result["disallowed"] == []
    assert result["unused_entries"] == [], "remove register rows for models no longer referenced"


def test_unregistered_model_fails():
    result = check(REG, {"mystery-7b": ["graphrag/x.py"]})
    assert result["ok"] is False and result["unregistered"][0]["model"] == "mystery-7b"


def test_disallowed_licence_fails_and_reviewed_entries_are_not_listed():
    result = check(REG, {"bad-1": ["a.py"], "gpt-oss-20b": ["b.py"]})
    assert result["ok"] is False and result["disallowed"] == [{"model": "bad-1", "licence": "proprietary"}]
    assert [n["model"] for n in result["needs_review"]] == ["bad-1"]


def test_unused_entry_is_reported_without_failing():
    result = check(REG, {"gpt-oss-20b": ["b.py"]})
    assert result["ok"] is True and result["unused_entries"] == ["bad-*", "unused-*"]


def test_scanner_finds_ids_with_trailing_punctuation_stripped(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "s.yml").write_text("model: deepseek-chat.\nother: text-embedding-3-large\n")
    (tmp_path / "config" / "licenses.yml").write_text("pattern: gpt-oss-*\n")
    found = find_model_ids(tmp_path, ("config",))
    assert set(found) == {"deepseek-chat", "text-embedding-3-large"}
