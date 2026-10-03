"""Canonical project paths. Raw data is read-only; everything derived goes to data/ or runs/."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("QA_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "Dataset" / "data" / "raw"
RAW_MANIFEST = RAW / "manifest.jsonl"
INTERIM = ROOT / "data" / "interim"
INTERIM_MANIFEST = INTERIM / "manifest.jsonl"
QUARTER = os.environ.get("QA_QUARTER", "2025q1")  # Form 4 quarter under audit; 2025q1 is the development quarter
BENCHMARK = ROOT / "data" / "benchmark" / ("demo" if QUARTER == "2025q1" else f"demo_{QUARTER}")
RUNS = ROOT / "runs"
NOTEBOOKS = ROOT / "notebooks"

INSIDER_ZIP = RAW / "sec_insider" / "{quarter}_form345.zip"
SUBMISSIONS_ZIP = RAW / "sec_bulk" / "submissions.zip"
SPY_CSV = RAW / "prices_tiingo" / "SPY.csv"
EXTERNAL_DOCS = ROOT / "data" / "external_docs"  # public web documentation fetched for grounding (see its manifest)
DOCS = {
    "sec_insider_readme": RAW / "docs" / "sec_insider_readme.pdf",
    "sec_forms_3_4_5_overview": RAW / "docs" / "sec_forms_3_4_5_overview.pdf",
}


def insider_dir(quarter: str = QUARTER) -> Path:
    return INTERIM / f"insider_{quarter}"


# researcher pipelines read their Form 4 data dir from $QA_DATA; every subprocess that runs one inherits this
os.environ.setdefault("QA_DATA", f"data/interim/insider_{QUARTER}")


def rel(path) -> str:
    """Path relative to the project root when inside it, absolute otherwise."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT.resolve())) if p.is_relative_to(ROOT.resolve()) else str(p)
