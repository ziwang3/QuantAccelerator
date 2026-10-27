import pytest

from quantaccelerator.paths import RAW
from quantaccelerator.tools import registry

# tests that read the downloaded raw data (Dataset/data/raw) or its ingested tables; skipped on a fresh clone
NEEDS_DATA = {
    "test_grounding_rejects_guessed_code_meanings",
    "test_identical_tool_call_is_not_rerun",
    "test_repeated_call_with_list_result_does_not_crash",
    "test_measure_separates_timing_and_value_leaks",
    "test_calendar_holidays",
    "test_asof_uses_only_published_records",
    "test_first_open_after",
    "test_grounding_checks_flags_and_code_list_hint",
    "test_grounding_names_the_page_when_meanings_are_right_but_miscited",
    "test_read_doc_finds_code_list_by_section_name",
    "test_read_doc_follows_appendix_cross_references",
    "test_tool_calls_are_registered",
    "test_utc_conversion_across_dst",
}


def pytest_collection_modifyitems(config, items):
    if RAW.exists() and any(RAW.iterdir()):
        return
    skip = pytest.mark.skip(reason="needs the downloaded data (see README: Data)")
    for item in items:
        if item.originalname in NEEDS_DATA or item.name in NEEDS_DATA:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Keep test tool calls out of the real ledger (runs/registry.sqlite)."""
    monkeypatch.setattr(registry, "DB", tmp_path / "registry.sqlite")
    yield
    registry.set_session(None)
