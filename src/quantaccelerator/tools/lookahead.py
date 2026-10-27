"""measure_lookahead: how early an events table uses filings relative to a reference availability rule."""
import numpy as np
import pandas as pd

from quantaccelerator.ingest import calendar as cal
from quantaccelerator.state.schemas import AvailabilityRule
from quantaccelerator.tools.registry import CTX, tool
from quantaccelerator.tools.timing import apply_rule


def lookahead_stats(events: pd.DataFrame, rule: AvailabilityRule, key_col: str = "asof_date",
                    acc_col: str = "ACCESSION_NUMBER") -> dict:
    """Compare each event's as-of date with the first tradable date under `rule`, in trading days.

    lookahead > 0 means the event is used that many sessions before it could be traded.
    """
    for c in (key_col, acc_col):
        if c not in events:
            raise ValueError(f"events table lacks column {c!r}; columns: {list(events.columns)}")
    ref = apply_rule(rule)
    ev = events[[acc_col, key_col]].copy()
    ev["ref"] = ev[acc_col].map(ref)
    ev = ev[ev.ref.notna() & ev[key_col].notna()]
    days = cal.trading_days().values
    asof = cal.roll_forward(pd.to_datetime(ev[key_col]))
    la = np.searchsorted(days, ev.ref.values) - np.searchsorted(days, asof.values)
    early = la > 0
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else 0.0
    return {"n_events": len(events), "n_matched": len(ev),
            "share_before_tradable": round(float(early.mean()), 4) if len(ev) else 0.0,
            "share_exactly_first_tradable": round(float((la == 0).mean()), 4) if len(ev) else 0.0,
            "share_later_than_needed": round(float((la < 0).mean()), 4) if len(ev) else 0.0,
            "lookahead_trading_days_early_events": {"median": q(la[early], 0.5), "p90": q(la[early], 0.9),
                                                    "max": float(la.max()) if len(ev) else 0.0}}


@tool("measure_lookahead", "Measure look-ahead of a pipeline's output events table (its key columns plus asof_date, "
      "where asof_date = the session whose open is the first time the event is used) against the signed PIT "
      "contract's availability rule. share_before_tradable is the share of events that use information which was "
      "not yet public at asof_date; also returns the look-ahead distribution in trading days.",
      {"type": "object", "properties": {"output_path": {"type": "string",
                                                        "description": "output_path returned by run_pipeline"}},
       "required": ["output_path"]},
      inputs=lambda output_path: [output_path])
def measure_lookahead(output_path: str):
    if CTX.reference_rule is None:
        raise RuntimeError("no signed PIT contract in this session; cannot measure look-ahead")
    from quantaccelerator.datasets import active
    return {"reference_rule": CTX.reference_rule, **active().measure(pd.read_parquet(output_path), CTX.reference_rule)}
