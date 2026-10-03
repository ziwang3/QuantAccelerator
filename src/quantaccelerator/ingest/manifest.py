"""Provenance for derived files: data/interim/manifest.jsonl (append-only)."""
import datetime as dt
import hashlib
import json
from pathlib import Path

from quantaccelerator.paths import INTERIM_MANIFEST, RAW_MANIFEST, ROOT, rel


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def source_sha256(path: Path) -> str:
    """sha256 of a source file, taken from the raw or interim manifest when listed (avoids re-hashing
    multi-GB raw files); computed otherwise."""
    p = Path(path).resolve()
    for mf, base in [(RAW_MANIFEST, ROOT / "Dataset"), (INTERIM_MANIFEST, ROOT)]:
        if mf.exists():
            for line in reversed(mf.read_text().splitlines()):
                rec = json.loads(line)
                if (base / rec["path"]).resolve() == p:
                    return rec["sha256"]
    return sha256_file(p)


def _relpath(p) -> str:
    """Project-relative path as written (data/ may be a symlink, e.g. in a worktree); absolute if outside."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return rel(p)


def record(output: Path, sources: list[Path], producer: str, **extra) -> dict:
    rec = {
        "path": _relpath(output),
        "sha256": sha256_file(Path(output)),
        "bytes": Path(output).stat().st_size,
        "sources": [{"path": _relpath(s), "sha256": source_sha256(s)} for s in sources],
        "producer": producer,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    INTERIM_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(INTERIM_MANIFEST, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
