"""Deterministic checks the orchestrator applies to agent outputs (limits error propagation between agents).

They verify claims against the data itself, never against gold labels.
"""
import json
import re

from quantaccelerator import build
from quantaccelerator.tools import registry
from quantaccelerator.tools.data import TABLES, load_table
from quantaccelerator.tools.docs import DOCS, passages

DOC_REF = re.compile(rf"({'|'.join(map(re.escape, DOCS))})\s*p\.?\s*(\d+)")
PAGE = re.compile(r"\bp(?:age)?\.?\s*(\d+)\b", re.I)


def doc_refs(evidence: str) -> set[tuple[str, int]]:
    """(doc, page) pairs cited in one evidence string. Canonical form is '<doc id> p<page>'; also accepts variants
    models produce, such as '<doc> p1 (fred_series_notes)' or 'fred_series_notes, page 1'."""
    refs = {(d, int(p)) for d, p in DOC_REF.findall(evidence)}
    if not refs:
        docs = [d for d in DOCS if d in evidence]
        pages = [int(p) for p in PAGE.findall(evidence)]
        if len(docs) == 1 and pages:
            refs = {(docs[0], p) for p in pages}
    return refs
STOP = {"with", "from", "that", "this", "which", "into", "other", "than", "code", "security", "securities",
        "transaction", "issuer", "insider", "pursuant", "rule", "their", "have", "been", "when", "where"}

FREQUENT_CODE_SHARE = 0.05
FLAG_VALUES = {"0", "1", "true", "false"}
STEM = 5  # prefix length for word matching ("acquisition" ~ "acquired")
DERIVED_TABLES = {"edgar_acceptance"}  # built by our ingest from the EDGAR index; no SEC doc describes its codes


def is_flag(values) -> bool:
    """Boolean columns (0/1/true/false) need no code list; their meaning is the column definition."""
    vals = {str(v).lower() for v in values if v is not None and str(v) != "nan"}
    return bool(vals) and vals <= FLAG_VALUES


def code_list_pages(codes: list[str]) -> list[str]:
    """Doc pages with a code-list line ('P Open market ...') for at least half of `codes` (hint for the agent)."""
    hits = []
    for doc in DOCS:  # the active case study's documentation
        for x in passages(doc):
            n = sum(bool(re.search(rf"(^|\|)\s*{re.escape(c)}\s+[A-Z][a-z]", x["text"], re.M)) for c in codes)
            if codes and n >= max(1, len(codes) / 2) and f"{doc} p{x['page']}" not in hits:
                hits.append(f"{doc} p{x['page']}")
    return hits[:3]


def _columns(table: str) -> list[str]:
    return list(load_table(table).columns)


def check_column_ref(ref: str, default_table: str | None = None) -> str | None:
    """'table.COLUMN' (or bare COLUMN with default_table / any table) must exist exactly."""
    table, _, col = ref.rpartition(".")
    table = table or default_table
    if table:
        if table not in TABLES:
            return f"'{ref}': unknown table {table!r} (tables: {sorted(TABLES)})"
        if col not in _columns(table):
            return f"'{ref}': no column {col!r} in {table}; actual columns: {_columns(table)}"
        return None
    if not any(col in _columns(t) for t in TABLES):
        return f"'{ref}': no such column in any table"
    return None


def opaque_code_columns(table: str) -> list[str]:
    import pandas as pd
    """Text columns holding a few short codes (<= 40 distinct values, median length <= 3): their meaning cannot be read
    off the values, so the card must explain them from the documentation."""
    if table in DERIVED_TABLES:
        return []
    df = load_table(table)
    out = []
    for c in df.columns:
        if df[c].dtype != object:
            continue
        vals = df[c].dropna()
        head = vals.head(200_000)
        uniq = pd.Series(head.unique()).astype(str)
        numbered = uniq.str.fullmatch(r"[A-Za-z]{1,2}\d+(\s*,\s*[A-Za-z]{1,2}\d+)*").mean() >= 0.8  # 'F1', 'F3, F1'
        if head.nunique() <= 40 and len(head) and head.astype(str).str.len().median() <= 3 and \
                not is_flag(head.unique()) and not numbered:
            out.append(c)
    return out


def check_overview(o) -> list[str]:
    if o is None:
        return ["overview: fill in the dataset overview first (what it is, publisher, market, entities, values, "
                "frequency, period, typical uses, caveats)"]
    runs = [r for r in _runs(o.evidence + [x for e in o.entities for x in e.evidence])]
    errs = []
    if not doc_refs(" ".join(o.evidence + [x for e in o.entities for x in e.evidence])):
        errs.append("overview.evidence: cite the documentation behind the overview ('<doc> p<page>')")
    if not runs:
        errs.append("overview.evidence: cite the tool runs behind the overview (e.g. profile_table, identifier_patterns)")
    ip = [r for r in runs if r["tool"] == "identifier_patterns"]
    for r in ip[:1]:
        note = (r.get("result") or {}).get("note", "")
        said = " ".join(f"{e.type} {e.share}" for e in o.entities).lower()
        if "cannot tell apart" in note and not any(w in said for w in ("etf", "fund", "exchange-traded")):
            errs.append("overview.entities: plain letter tickers cover common stocks and ETFs/funds alike (the "
                        "identifier_patterns note says so); describe that entity type accordingly, e.g. 'common stocks "
                        "and ETFs (plain tickers)'")
        big = (r.get("result") or {}).get("classes_above_1pct_of_identifiers") or []
        if len(o.entities) < min(len(big), 6):
            errs.append(f"overview.entities: the measured identifier mix has {len(big)} forms above 1% of identifiers "
                        f"({', '.join(big)}); give one entity type per form, with its share of identifiers and of rows "
                        "or volume, and say what the form likely is (from docs or market conventions)")
    if not any(r["tool"] in ("identifier_patterns", "profile_table") for r in
               _runs([x for e in o.entities for x in e.evidence] + o.evidence)):
        errs.append("overview.entities: measure which kinds of entities the data holds (identifier_patterns or "
                    "profile_table) and cite that run")
    return errs


def check_dataset_card(card) -> list[str]:
    errs = []
    described = {(f.table, f.name) for f in card.fields if f.value_meanings}
    for t in {t.table for t in card.tables} & set(TABLES):
        missing = [c for c in opaque_code_columns(t) if (t, c) not in described]
        if missing:
            errs.append(f"fields: describe the coded columns of {t} with value_meanings from the documentation (their "
                        f"codes cannot be read off the values): {missing}")
    for t in card.tables:
        if t.table not in TABLES:
            errs.append(f"tables: unknown table {t.table!r}")
            continue
        errs += [e for k in t.primary_key if (e := check_column_ref(k, t.table))]
        run = registry.get_run(t.n_rows_run_id)
        if not run or run["tool"] != "profile_table" or run["args"].get("table") != t.table:
            errs.append(f"tables.{t.table}.n_rows_run_id must be your profile_table run for {t.table}")
    errs += [f"time_fields: {e}" for r in card.time_fields if (e := check_column_ref(r))]
    errs += [f"entity_id_fields: {e}" for r in card.entity_id_fields if (e := check_column_ref(r))]
    for f in card.fields:
        if (e := check_column_ref(f.name, f.table)):
            errs.append(f"fields: {e}")
            continue
        if f.value_meanings:
            vc = load_table(f.table)[f.name].value_counts(normalize=True)
            if is_flag(vc.index) or f.table in DERIVED_TABLES:
                continue
            frequent = [str(k) for k, v in vc.items() if v >= FREQUENT_CODE_SHARE]
            # multi-valued cells ("Q,N") are covered when each of their codes is
            parts = lambda c: [p.strip() for p in c.split(",")] if "," in c else [c]
            missing = list(dict.fromkeys(p for c in frequent for p in parts(c) if p not in f.value_meanings))
            combos = [c for c in frequent if "," in c]
            frequent = list(dict.fromkeys(p for c in frequent for p in parts(c)))
            if missing:
                errs.append(f"fields: {f.table}.{f.name} value_meanings must cover every code with >= "
                            f"{FREQUENT_CODE_SHARE:.0%} of rows; missing {missing} (look them up in the docs)"
                            + (f"; cells like {combos[0]!r} combine several codes, so give one meaning per single code"
                               if combos else ""))
            # codes absent from this quarter are fine if documented; check_grounding checks every listed meaning
            if not build.prototype():
                errs += check_grounding(f, frequent)
    return errs[:12]


def check_pit_contract(contract) -> list[str]:
    errs = [f"fields: {e}" for f in contract.fields if (e := check_column_ref(f.field))]
    errs += check_same_day_use(contract)
    from quantaccelerator.datasets import active
    return (errs + ([] if build.prototype() else active().check_pit(contract)))[:12]


def cited_pages_text(evidence: list[str]) -> str:
    refs = {r for e in evidence for r in doc_refs(e)}
    return " ".join(x["text"] for d, p in refs for x in passages(d) if x["page"] == p).lower()


def check_grounding(f, frequent: list[str] | None = None) -> list[str]:
    """Code meanings must be supported by the documentation pages the field cites (not guessed from the letter)."""
    text = cited_pages_text(f.evidence)
    if not text:
        return [f"fields: {f.table}.{f.name} has value_meanings but cites no documentation page "
                f"(cite as '<doc> p<page>' the page that lists the codes)"]
    def supported(meaning: str, page_text: str) -> bool:
        words = [w for w in re.findall(r"[a-z]{4,}", meaning.lower()) if w not in STOP]
        return not words or sum(w[:STEM] in page_text for w in words) / len(words) >= 0.5

    unsupported = {c: m for c, m in f.value_meanings.items() if not supported(m, text)}
    if not unsupported:
        return []
    listed = [f"{c}={m!r}" for c, m in list(unsupported.items())[:6]]
    # meanings that are right but cite the wrong page(s): name the page(s) whose wording they match
    candidates = code_list_pages(frequent or list(f.value_meanings)) + \
        [f"{d} p{x['page']}" for d in DOCS for x in passages(d)]
    found = {}
    for c, m in unsupported.items():
        ref = next((r for r in dict.fromkeys(candidates) if supported(m, cited_pages_text([r]))), None)
        if ref is None:
            break
        found[c] = ref
    if len(found) == len(unsupported):
        pages = sorted(set(found.values()))
        return [f"fields: {f.table}.{f.name}: these meanings match the wording of {', '.join(pages)}, but your "
                f"evidence does not cite {'that page' if len(pages) == 1 else 'those pages'}: {listed}. Add "
                f"{', '.join(repr(x) for x in pages)} to this field's evidence."]
    pages = code_list_pages(frequent or list(f.value_meanings))
    hint = f" The code list appears on {', '.join(pages)}; read it and cite it." if pages else ""
    return [f"fields: {f.table}.{f.name} meanings not supported by the cited pages: {listed}; "
            f"use the wording of the documentation's code list.{hint}"]


def check_pit_evidence(contract) -> list[str]:
    """A rule built on acceptanceDateTime must cite a timezone_check run and agree with its result."""
    rule = contract.availability_rule
    if rule.source_field != "edgar_acceptance.acceptanceDateTime":
        return []
    runs = [r for r in (registry.get_run(x) for x in contract.evidence) if r and r["tool"] == "timezone_check"]
    if not runs:
        return ["evidence: a rule based on acceptanceDateTime must cite a timezone_check run (call it and cite it)"]
    best = runs[-1]["result"]["summary"]["more_consistent_reading"]
    implied = "UTC" if best.startswith("A") else "America/New_York"
    if rule.source_timezone != implied:
        return [f"availability_rule.source_timezone={rule.source_timezone!r} contradicts your cited timezone_check "
                f"run {runs[-1]['run_id']}, whose more consistent reading is {best!r} (source_timezone "
                f"{implied!r}). Reconcile the rule with the evidence."]
    return []


# ---- data research (profiling pass) ----------------------------------------------------------------
DATE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
CAPABILITY_TOOLS = {"research_breadth", "coverage_over_time", "update_dynamics", "category_mix_over_time",
                    "structural_breaks", "missingness", "profile_table", "profile_time"}
DATE_TOLERANCE_DAYS = 45


def _runs(refs: list[str]) -> list[dict]:
    from quantaccelerator.agents.base import RUN_ID
    return [r for r in (registry.get_run(x) for x in dict.fromkeys(RUN_ID.findall(" ".join(refs)))) if r]


def _dates(text: str) -> list:
    import pandas as pd
    out = []
    for y, m, d in DATE.findall(text):
        try:
            out.append(pd.Timestamp(f"{y}-{m}-{d or '15'}"))
        except ValueError:
            pass
    return out


def _data_tools() -> set:
    from quantaccelerator.tools.explore import EXPLORE_TOOLS
    return set(EXPLORE_TOOLS) - {"look_at_figure"} | {"profile_table", "profile_time", "preview_cleaning"}


def figure_of(refs: list[str]) -> str | None:
    """The first cited run that drew a figure."""
    for r in _runs(refs):
        if isinstance(r.get("result"), dict) and r["result"].get("figure"):
            return r["run_id"]
    return None


def check_observation(o, i: int | str = "") -> list[str]:
    """An observation rests on a data tool run of this session; its dates come from that run's numbers."""
    runs = _runs(o.evidence + ([o.figure_run_id] if o.figure_run_id else []))
    numeric = [r for r in runs if r["tool"] in _data_tools()]
    label = f"observations[{i}]" if i != "" else "observation"
    if not numeric:
        return [f"{label} ('{o.claim[:60]}'): cite the run_id of the data tool that measured it (a look_at_figure run "
                "alone is not evidence)"]
    known = [d for r in numeric for d in _dates(json.dumps(r.get("result"), default=str))]
    for d in _dates(o.claim):
        if not known or min(abs((d - k).days) for k in known) > DATE_TOLERANCE_DAYS:
            return [f"{label} mentions {d.date()} but no cited tool result has a date near it; take dates from the "
                    "tools' numbers, not from reading a chart"]
    return []


def check_capability(name: str, dim) -> list[str]:
    if dim is not None and not {r["tool"] for r in _runs(dim.evidence)} & CAPABILITY_TOOLS:
        yours = session_runs_of({"research_breadth", "coverage_over_time"})
        return [f"capability.{name}.evidence must include the run_id of the call that measured it"
                + (f": put {yours[:2]} (your research_breadth / coverage_over_time runs) in its evidence list"
                   if yours else ": call research_breadth first")]
    return []


def check_preprocessing(p, i: int | str = "") -> list[str]:
    return [f"preprocessing[{i}].field: {e}" for ref in p.field.split("/") if (e := check_column_ref(ref.strip()))]


def check_research(draft) -> list[str]:
    """Every rating, observation and cleaning step must rest on this run's tool results; dates in observations must
    come from tool numbers (a vision model's reading of a chart is not enough)."""
    from quantaccelerator.tools.clean import step_rows, validate_step
    errs = []
    for name in ("history_depth", "cross_sectional_breadth", "update_frequency", "coverage_stability", "event_breadth"):
        errs += check_capability(name, getattr(draft.capability, name))
    for i, o in enumerate(draft.observations):
        errs += check_observation(o, i)
        fig = registry.get_run(o.figure_run_id) if o.figure_run_id else None
        if o.figure_run_id and not (fig and isinstance(fig.get("result"), dict) and "figure" in fig["result"]):
            errs.append(f"observations[{i}].figure_run_id {o.figure_run_id!r} is not a figure made in this run")
    for i, s in enumerate(draft.cleaning.steps):
        bad = validate_step(s)
        errs += [f"cleaning.steps[{i}]: {e}" for e in bad]
        if not bad and step_rows(s) == 0:
            errs.append(f"cleaning.steps[{i}] ({s.op} on {s.table}) changes no rows; leave it out")
        if s.op in ("drop_duplicates", "drop_rows") and not _runs(s.evidence):
            match = matching_preview(s)
            errs.append(f"cleaning.steps[{i}] drops rows but its evidence cites no run: "
                        + (f"add your preview_cleaning run {match!r} to this step's evidence list" if match else
                           "call preview_cleaning with exactly this step and put its run_id in the step's evidence"))
    for i, p in enumerate(draft.preprocessing):
        errs += check_preprocessing(p, i)
    errs += unaddressed_flags(draft)
    return errs[:12]


def session_flags(agent: str = "data_research") -> list[tuple[str, dict]]:
    """(run_id, flag) for every flag raised by this agent's tool runs in the current session."""
    out = []
    for rid in registry.session_runs(registry.CTX.session_id).query("agent == @agent").run_id:
        r = registry.get_run(rid)
        for f in (r["result"].get("flags") or []) if r and isinstance(r.get("result"), dict) else []:
            out.append((rid, f))
    return out


def unaddressed_flags(draft) -> list[str]:
    """Material facts the tools flagged must appear in the findings (observations, warnings, open questions,
    preprocessing), or, for a redundant key column, in key_corrections."""
    items = [x.lower() for x in [o.claim for o in draft.observations] + draft.warnings + draft.open_questions +
             [" ".join([p.field, *p.observations, *p.risks]) for p in draft.preprocessing]]
    text = " ".join(items)
    missing, seen = [], set()
    for rid, f in session_flags():
        if f["id"] in seen:
            continue
        seen.add(f["id"])
        if f["id"] == "redundant_key":
            redundant = set(f["terms"][0])  # the redundant columns, lower-case
            if any(k and not redundant & {c.lower() for c in k} for k in draft.key_corrections.values()):
                continue
        # one finding must state the fact: words scattered over unrelated findings do not report it
        if not any(all(any(t in item for t in group) for group in f["terms"]) for item in items):
            missing.append(f"{f['fact']} ({rid})")
    errs = [f"your tools flagged facts your findings do not mention: {'; '.join(missing[:6])}. Report each in one "
            "observation or warning, or explain in warnings why it does not matter"] if missing else []
    return errs + unremedied(draft, text)


def _remedied(remedy: dict, steps) -> bool:
    ops = remedy["op"] if isinstance(remedy["op"], list) else [remedy["op"]]
    for s in steps:
        if s.op not in ops:
            continue
        w = remedy.get("where")
        same = lambda p: p.column == w["column"] and str(p.value) == str(w["value"]) and \
            p.op in (w["op"], w["op"].replace("col", ""))  # '>' against a column name is the same predicate
        if not w or any(same(p) for p in s.where):
            return True
    return False


def unremedied(draft, text: str) -> list[str]:
    """Data errors the tools flagged with a remedy need that cleaning step, or a warning explaining why not."""
    out = []
    for rid, f in session_flags():
        r = f.get("remedy")
        if not r or _remedied(r, draft.cleaning.steps):
            continue
        if any(w in text for w in ("no cleaning", "not clean", "leave", "keep them", "intentional", "do not remove",
                                   "should not be removed")) and f["terms"][0][0] in text:
            continue
        what = (f"{' or '.join(r['op']) if isinstance(r['op'], list) else r['op']}"
                + (f" where {r['where']['column']} {r['where']['op']} {r['where']['value']}" if r.get("where") else ""))
        out.append(f"{f['fact']} ({rid}) is a data error: propose_cleaning_step {what}, or explain in warnings why the "
                   "rows should stay as they are")
    return out


def check_answer(ans, given: str = "") -> list[str]:
    """A follow-up answer rests on this session's data tools, and its dates come from their numbers (or from the
    question itself, `given`)."""
    from quantaccelerator.tools.explore import EXPLORE_TOOLS
    data_tools = set(EXPLORE_TOOLS) - {"look_at_figure"} | {"profile_table", "profile_time", "preview_cleaning",
                                                            "read_doc"}
    runs = [r for r in _runs(ans.evidence + ans.figure_run_ids) if r["tool"] in data_tools]
    if not runs:
        return ["cite the run_id of at least one data tool (or read_doc) call behind the answer"]
    known = [d for r in runs for d in _dates(json.dumps(r.get("result"), default=str))] + _dates(given)
    for d in _dates(ans.answer):
        if not known or min(abs((d - k).days) for k in known) > DATE_TOLERANCE_DAYS:
            return [f"the answer mentions {d.date()} but no cited tool result has a date near it"]
    return []


def matching_preview(step) -> str | None:
    """run_id of this session's preview_cleaning call on the same operation, table, columns and predicates."""
    want = step.model_dump(include={"op", "table", "columns", "where"})
    for rid in registry.session_runs(registry.CTX.session_id).query("tool == 'preview_cleaning'").run_id[::-1]:
        r = registry.get_run(rid)
        got = (r or {}).get("args", {}).get("step", {})
        got = {k: got.get(k, [] if k in ("columns", "where") else None) for k in want}
        got["where"] = [{"value": None, **w} for w in got["where"]]
        if got == want:
            return rid
    return None


def session_runs_of(tools: set, agent: str = "data_research") -> list[str]:
    """run_ids of this agent's calls to the given tools in the current session, latest first."""
    runs = registry.session_runs(registry.CTX.session_id)
    return list(runs[runs.tool.isin(tools) & (runs.agent == agent)].run_id[::-1])


NUMERIC_KINDS = {"level", "flow", "stock", "event", "estimate", "revision", "score", "ratio"}


def missing_analyses(card, agent: str = "data_research") -> list[str]:
    """The research checklist, derived from the DatasetCard: what a careful researcher always checks on the main table
    (the table with the most rows among those with numeric fields). Returns what this agent has not run yet."""
    runs = registry.session_runs(registry.CTX.session_id)
    runs = runs[(runs.agent == agent) & (runs.ok == 1)]
    calls = [(t, json.loads(a)) for t, a in zip(runs.tool, runs.args_json)]
    has = lambda tool, **kw: any(t == tool and all(a.get(k) == v for k, v in kw.items()) for t, a in calls)
    num = [f for f in card.fields if f.kind in NUMERIC_KINDS]
    if not num:
        return []
    tables = {f.table for f in num}
    rows = {t.table: t for t in card.tables}
    main = max(tables, key=lambda t: len([f for f in num if f.table == t]))
    ents = [e.split(".")[-1] for e in card.entity_id_fields if e.split(".")[0] in (main, e)]
    times = [t.split(".")[-1] for t in card.time_fields if t.startswith(f"{main}.")]
    todo = []
    if not has("research_breadth", table=main):
        todo.append(f"research_breadth on {main}")
    if main in rows and not has("key_check", table=main):
        todo.append(f"key_check on {main} with its primary key {rows[main].primary_key}")
    if times and not any(t == "missingness" and a.get("table") == main and a.get("time_field") for t, a in calls):
        todo.append(f"missingness on {main} with time_field" + (" and entity_field" if ents else ""))
    for f in num:
        if f.table == main and not has("distribution", table=main, field=f.name, denominator=None):
            todo.append(f"distribution of {main}.{f.name}")
    units = {}
    for f in num:
        if f.table == main and f.unit:
            units.setdefault(f.unit.lower(), []).append(f.name)
    for unit, names in units.items():
        if len(names) > 1 and not any(t == "distribution" and a.get("table") == main and a.get("denominator") in names
                                      for t, a in calls):
            todo.append(f"distribution of a ratio of {main} fields measured in {unit} ({', '.join(names)}), with "
                        "denominator set")
    if times and not any(t == "structural_breaks" and a.get("table") == main and a.get("field") for t, a in calls):
        todo.append(f"structural_breaks on {main} for its main measure")
    for f in card.fields:
        if f.table == main and f.kind == "code" and not has("category_mix_over_time", table=main, field=f.name):
            todo.append(f"category_mix_over_time of {main}.{f.name}")
    return todo


def check_same_day_use(contract) -> list[str]:
    """A date-only field that describes what happened ON a date (event time or reporting period) cannot be traded at
    that date's own open: the day has not happened yet. Such a rule needs the next trading day or a lag."""
    if build.prototype():
        return []
    rule = contract.availability_rule
    if rule.tradable_at == "next_trading_day_open" or rule.extra_lag_trading_days > 0:
        return []
    role = next((f.role for f in contract.fields if f.field.split(".")[-1] == rule.source_field.split(".")[-1]), None)
    if role not in ("event_time", "reporting_period"):
        return []
    table, col = rule.source_field.split(".")
    if table not in TABLES or col not in load_table(table).columns:
        return []
    t = load_table(table)[col].dropna().head(100_000)
    import pandas as pd
    t = pd.to_datetime(t, errors="coerce").dropna()
    if len(t) and (t == t.dt.normalize()).all():  # date only
        return [f"availability_rule: {rule.source_field} is a date-only {role.replace('_', ' ')}, so tradable_at="
                f"{rule.tradable_at!r} would use each record at the open of the very day it describes, before that "
                "day's data exists. Use next_trading_day_open (or an extra lag) unless you have evidence the data is "
                "published before that open."]
    return []
