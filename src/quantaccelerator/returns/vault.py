"""The returns vault: the only way to reach returns, and only for a frozen, pre-registered feature set.

`arm` opens the vault for one run. It needs the signed G3 CandidateFeatureSet, whose hash is recomputed and whose
frozen feature table must exist, and the signed G4 prediction plan with its split. After that:
  - discovery: available to the predictive tools (the agent), the only segment the agent ever sees;
  - validation: the orchestrator only, to re-run the agent's observations deterministically;
  - locked_test: the orchestrator only, after G5 released observations, exactly once.
Each segment's returns are cut at the segment's end before forward returns are computed, so a window cannot reach
into a later segment. Every access, granted or refused, is written to the vault_access table of the run registry, and
every distinct test is counted in the trials table (the trial count sets the discovery hurdle).
"""
import contextlib
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from quantaccelerator.returns import core
from quantaccelerator.tools import registry

SEGMENTS = ("discovery", "validation", "locked_test")
STATE: dict = {"run_dir": None, "set_hash": None, "study": None, "features": None, "frozen": [], "split": None,
               "plan": None, "source": None, "link": None, "conditioners": None, "segment": "discovery",
               "released": False, "cache": {}}


class VaultError(PermissionError):
    pass


def feature_set_hash(fs) -> str:
    """The G3 freeze hash: panel hash, every feature spec, and the sorted frozen names."""
    keep = (fs.signed_off.approved_features or []) if fs.signed_off and fs.signed_off.approved else []
    body = json.dumps({"panel": fs.panel_hash, "features": [s.model_dump(mode="json") for s in fs.features],
                       "frozen": sorted(keep)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def _db():
    c = registry._conn()
    c.execute("""CREATE TABLE IF NOT EXISTS vault_access (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
        study TEXT, segment TEXT, caller TEXT, granted INTEGER, reason TEXT, created_utc TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trials (id INTEGER PRIMARY KEY AUTOINCREMENT, study TEXT, session_id TEXT,
        run_id TEXT, tool TEXT, test_key TEXT, segment TEXT, created_utc TEXT)""")
    return c


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _log(segment: str, granted: bool, reason: str = "") -> None:
    with _db() as c:
        c.execute("INSERT INTO vault_access (session_id, study, segment, caller, granted, reason, created_utc) "
                  "VALUES (?,?,?,?,?,?,?)", (registry.CTX.session_id, STATE["study"], segment,
                                             registry.CTX.agent or "", int(granted), reason, _now()))


def accesses(study: str | None = None, session_id: str | None = None) -> pd.DataFrame:
    q, a = "SELECT * FROM vault_access WHERE 1=1", []
    if study:
        q, a = q + " AND study=?", a + [study]
    if session_id:
        q, a = q + " AND session_id=?", a + [session_id]
    with _db() as c:
        return pd.read_sql_query(q + " ORDER BY id", c, params=a)


def count_trial(tool: str, keys: list[str]) -> int:
    """Record the distinct tests a tool call ran (on discovery); returns the set's trial count afterwards."""
    with _db() as c:
        for k in keys:
            c.execute("INSERT INTO trials (study, session_id, run_id, tool, test_key, segment, created_utc) "
                      "VALUES (?,?,?,?,?,?,?)", (STATE["study"], registry.CTX.session_id,
                                                 registry.CTX.current_run_id, tool, k, STATE["segment"], _now()))
    return n_trials()


def n_trials(study: str | None = None) -> int:
    """Distinct tests run on discovery in this study (by default: this feature set, across every session; repeating a
    test is not a trial)."""
    with _db() as c:
        return c.execute("SELECT COUNT(DISTINCT test_key) FROM trials WHERE study=? AND segment='discovery'",
                         [study or STATE["study"]]).fetchone()[0]


# ---------------------------------------------------------------- arming
def arm(run_dir: Path, fs, plan_gate, source, conditioners: pd.DataFrame, study: str | None = None,
        link=None) -> None:
    """Open the vault for this run. `source()` returns the returns frame (core module docstring) for the panel's
    entities; `conditioners` has date, entity, sector and the panel's other conditioners. When the returns are
    keyed by another id (e.g. CRSP permno), `link(pairs)` gives date, entity, rid: the security each panel entity is on
    that date (returns and controls are computed per rid, so a window never spans two securities). The trial count
    and the one locked-test access belong to the study: the feature set itself by default (every session that ever
    tested it), or e.g. '<hash>:<run_id>' for independent benchmark repeats on a fixed oracle feature set."""
    STATE.update(run_dir=None, set_hash=None, study=None, features=None, split=None, plan=None, released=False,
                 segment="discovery", cache={})
    if fs.signed_off is None or not fs.signed_off.approved:
        raise VaultError("the feature set is not frozen: gate G3 must approve it before any return is seen")
    h = feature_set_hash(fs)
    if h != fs.set_hash:
        raise VaultError(f"the feature set's hash {fs.set_hash!r} does not match its content ({h!r}): it changed "
                         "after the freeze")
    path = Path(run_dir) / "features" / f"{h}.parquet"
    if not path.exists():
        raise VaultError(f"the frozen feature table {path} is missing")
    if plan_gate is None or plan_gate.gate != "G4" or not plan_gate.approved or plan_gate.approved_plan is None:
        raise VaultError("the prediction plan is not signed: gate G4 must approve it (and lock the split) first")
    plan = plan_gate.approved_plan
    if plan.set_hash != h:
        raise VaultError(f"the signed plan is for feature set {plan.set_hash!r}, not {h!r}")
    feats = pd.read_parquet(path)
    STATE.update(run_dir=Path(run_dir), set_hash=h, study=study or h, features=feats,
                 frozen=list(fs.signed_off.approved_features or []), split=plan.split, plan=plan, source=source,
                 link=link,
                 conditioners=conditioners)


def armed() -> bool:
    return STATE["set_hash"] is not None


@contextlib.contextmanager
def segment(name: str):
    """Switch the segment the predictive tools read (orchestrator only)."""
    if name not in SEGMENTS:
        raise ValueError(name)
    if registry.CTX.agent != "orchestrator":
        _log(name, False, f"segment switch by {registry.CTX.agent!r}")
        raise VaultError(f"only the orchestrator can use the {name} segment")
    if name == "locked_test":
        if not STATE["released"]:
            _log(name, False, "no G5 release")
            raise VaultError("the locked test is opened only after gate G5 releases observations")
        used = accesses(STATE["study"])
        if ((used.segment == "locked_test") & (used.granted == 1)).any():
            _log(name, False, "second access")
            raise VaultError(f"the locked test of study {STATE['study']} was already used; it is touched once")
    prev = STATE["segment"]
    STATE["segment"] = name
    _log(name, True, "orchestrator")
    try:
        yield
    finally:
        STATE["segment"] = prev


def release(note: str = "") -> None:
    """Gate G5 has released observations for the locked test."""
    if registry.CTX.agent != "orchestrator":
        raise VaultError("only the orchestrator records the G5 release")
    STATE["released"] = True


# ---------------------------------------------------------------- data
def bounds(seg: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = getattr(STATE["split"], seg)
    return pd.Timestamp(a), pd.Timestamp(b)


def frame(seg: str | None = None) -> pd.DataFrame:
    """Frozen features + conditioners + controls + forward returns, decision dates within the segment.
    The agent's tools always get the discovery segment."""
    if not armed():
        raise VaultError("the returns vault is not armed: freeze the feature set (G3) and sign the prediction plan "
                         "(G4) first")
    seg = seg or STATE["segment"]
    if seg != "discovery" and seg != STATE["segment"]:
        _log(seg, False, "direct read outside the active segment")
        raise VaultError(f"the {seg} segment is not open")
    if seg in STATE["cache"]:
        return STATE["cache"][seg]
    start, end = bounds(seg)
    rf = STATE["source"]()
    hist = rf[rf.date <= end]                         # nothing after the segment: windows stop at its end
    fwd = core.forward_returns(hist[hist.date >= start])
    ctl = core.controls(hist)
    f = STATE["features"]
    f = f[(f.date >= start) & (f.date <= end)]
    if STATE["link"] is not None:
        f = f.merge(STATE["link"](f[["date", "entity"]]), on=["date", "entity"], how="left")
    else:
        f = f.assign(rid=f.entity)
    on = lambda x: x.rename(columns={"entity": "rid"})
    d = f.merge(STATE["conditioners"], on=["date", "entity"], how="left") \
        .merge(on(ctl), on=["date", "rid"], how="left").merge(on(fwd), on=["date", "rid"], how="left")
    STATE["cache"][seg] = d
    if seg == "discovery":
        _log(seg, True, "tools")
    return d


def coverage(seg: str = "discovery") -> dict:
    d = frame(seg)
    return {"rows": int(len(d)), "dates": int(d.date.nunique()), "entities": int(d.entity.nunique()),
            "share_linked": round(float(d.rid.notna().mean()), 4),
            "share_with_return": round(float(d.fwd_1.notna().mean()), 4),
            "first_date": str(d.date.min().date()), "last_date": str(d.date.max().date())}
