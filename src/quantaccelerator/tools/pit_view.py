"""PITResearchView: the one canonical, point-in-time research table built from a signed PIT contract.

Downstream research never rebuilds the timing joins itself: it reads this view, where every record carries its
decision_time (the first session whose open may use it, under the signed availability rule), and snapshots taken with
`asof`, which only ever return records whose decision_time is on or before the date. Cleaning flags from an approved
recipe travel with the records; nothing is dropped here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from quantaccelerator.datasets import active
from quantaccelerator.ingest.calendar import trading_days
from quantaccelerator.state.schemas import PITContract
from quantaccelerator.tools.data import load_table


class UnsignedContract(PermissionError):
    """The view needs a PIT contract signed off at gate G1."""


def signed_rule(contract: PITContract):
    g = contract.signed_off
    if g is None:
        raise UnsignedContract("the PIT contract has not been signed off at G1; no research view without it")
    rule = contract.availability_rule if g.approved else g.corrected_rule
    if rule is None:
        raise UnsignedContract("G1 rejected the contract without a replacement rule")
    return rule


def build_pit_view(contract: PITContract, derived: dict[str, str] | None = None) -> pd.DataFrame:
    p = active()
    if not p.view or not p.first_tradable:
        raise NotImplementedError(f"dataset profile {p.name!r} declares no PIT research view")
    rule = signed_rule(contract)
    v = p.view
    df = pd.read_parquet(Path(derived[v["table"]])) if derived and v["table"] in derived else load_table(v["table"])
    flags = [c for c in df.columns if df[c].dtype == bool and c not in v["values"]]  # added by the approved recipe
    out = pd.DataFrame({"entity": df[v["entity"]].values, "source_time": df[v["time"]].values,
                        "decision_time": p.first_tradable(rule, df).values})
    for c in v["values"] + flags:
        out[c] = df[c].values
    out["quality_flag"] = df[flags].any(axis=1).values if flags else False
    out = out.dropna(subset=["decision_time"]).sort_values(["decision_time", "entity"]).reset_index(drop=True)
    out.attrs.update({"rule": rule.model_dump(), "dataset_id": p.dataset_id, "flags": flags})
    return out


def asof(view: pd.DataFrame, dates, entities=None) -> pd.DataFrame:
    """For each date and entity, the latest record usable at that date's open (decision_time <= date), with
    feature_age = trading days since it became usable. Entities without a usable record are left out."""
    cal = trading_days()
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    ents = pd.Index(entities if entities is not None else view.entity.unique())
    left = pd.DataFrame({"date": np.repeat(dates.values, len(ents)), "entity": np.tile(ents.values, len(dates))})
    right = view[view.entity.isin(ents)].sort_values("decision_time")
    snap = pd.merge_asof(left.sort_values("date"), right, left_on="date", right_on="decision_time", by="entity",
                         direction="backward").dropna(subset=["decision_time"])
    snap["feature_age"] = cal.searchsorted(snap.date.values) - cal.searchsorted(snap.decision_time.values)
    return snap.reset_index(drop=True)
