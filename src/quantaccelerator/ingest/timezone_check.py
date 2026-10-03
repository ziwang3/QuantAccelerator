"""Empirical test of the timezone of EDGAR `acceptanceDateTime` (values end in 'Z').

Reading A: the 'Z' is honest (UTC) -> convert to America/New_York.
Reading B: the value is Eastern wall-clock time mislabeled as 'Z'.
EDGAR rule (Reg S-T 13(a)(4)): Forms 3/4/5 submitted by 22:00 ET get that day's filing date;
most other forms get the next business day's filing date after 17:30 ET.
Prints the evidence as markdown tables.
"""
import numpy as np
import pandas as pd

from quantaccelerator.paths import insider_dir

SEC16 = ["3", "4", "5", "3/A", "4/A", "5/A"]


def readings(raw: pd.Series) -> dict[str, pd.Series]:
    naive = pd.to_datetime(raw.str.replace("Z", "", regex=False))
    return {
        "A (UTC -> ET)": naive.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None),
        "B (face value = ET)": naive,
    }


def to_et(raw: pd.Series) -> pd.Series:
    """The adopted reading (A). Returns naive America/New_York wall-clock timestamps."""
    return readings(raw)["A (UTC -> ET)"]


def consistency_table(quarter: str = "2025q1") -> dict:
    """Consistency of each timezone reading with EDGAR filing-date cutoffs, plus Form 3/4/5 hour histograms."""
    w = pd.read_parquet(insider_dir(quarter) / "filings_window.parquet")
    fd = pd.to_datetime(w.filingDate)
    sec16 = w.form.isin(SEC16)
    rows, hist = [], {}
    for name, t in readings(w.acceptanceDateTime).items():
        mins = t.dt.hour * 60 + t.dt.minute
        day = t.dt.normalize()
        hist[name] = t[sec16].dt.hour.value_counts().reindex(range(24), fill_value=0)
        for grp, m, cut, rule in [("Form 3/4/5", sec16, 22 * 60, "22:00"), ("Form 3/4/5", sec16, 17 * 60 + 30, "17:30"),
                                  ("other forms", ~sec16, 17 * 60 + 30, "17:30")]:
            pred_same = (t.dt.weekday < 5) & (mins < cut) & m
            pred_next = ~pred_same & m
            c1 = ((fd == day) & pred_same).sum() / max(pred_same.sum(), 1)
            c2 = ((fd > day) & pred_next).sum() / max(pred_next.sum(), 1)
            night = (m & ((mins >= 22 * 60) | (mins < 6 * 60))).sum()
            rows.append({"reading": name, "form_group": grp, "cutoff_et": rule, "n_pred_same_day": int(pred_same.sum()),
                         "consistent_same_day": round(float(c1), 4), "n_pred_next_day": int(pred_next.sum()),
                         "consistent_next_day": round(float(c2), 4),
                         "n_filingdate_before_acceptance_date": int(((fd < day) & m).sum()),
                         "n_accepted_22h_to_06h_et": int(night)})
    sec = [r for r in rows if r["form_group"] == "Form 3/4/5" and r["cutoff_et"] == "22:00"]
    score = {r["reading"]: (r["n_pred_same_day"] * r["consistent_same_day"] + r["n_pred_next_day"] *
                            r["consistent_next_day"]) / max(r["n_pred_same_day"] + r["n_pred_next_day"], 1) for r in sec}
    best = max(score, key=score.get)
    summary = {"form345_consistency_by_reading": {k: round(v, 4) for k, v in score.items()},
               "more_consistent_reading": best,
               "impossible_under_reading": {r["reading"]: r["n_filingdate_before_acceptance_date"] for r in sec}}
    return {"summary": summary, "n_filings": len(w), "rows": rows,
            "form345_hour_hist_et": {k: v.to_dict() for k, v in hist.items()}}


def report(quarter: str = "2025q1") -> str:
    ct = consistency_table(quarter)
    lines = ["| reading | form group | rule | n predicted same-day | consistent | n predicted next-day | consistent | filingDate < acceptance date | acceptances 22:00-05:59 ET |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['reading']} | {r['form_group']} | {r['cutoff_et']} | {r['n_pred_same_day']} | {r['consistent_same_day']:.4f} | "
              f"{r['n_pred_next_day']} | {r['consistent_next_day']:.4f} | {r['n_filingdate_before_acceptance_date']} | "
              f"{r['n_accepted_22h_to_06h_et']} |" for r in ct["rows"]]
    h = pd.DataFrame(ct["form345_hour_hist_et"]).T
    hl = ["| reading | " + " | ".join(f"{i:02d}" for i in range(24)) + " |", "|---|" + "---|" * 24]
    hl += [f"| {k} | " + " | ".join(str(v) for v in row) + " |" for k, row in h.iterrows()]
    return "\n".join(lines) + "\n\nForm 3/4/5 acceptance-hour histogram (ET hour of day):\n\n" + "\n".join(hl)


if __name__ == "__main__":
    print(report())
