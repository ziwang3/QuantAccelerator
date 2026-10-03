"""pit_join: as-of join that refuses to run without an explicit availability column (guardrail 3)."""
import pandas as pd


def pit_join(left: pd.DataFrame, right: pd.DataFrame, key: str, availability_col: str | None, as_of_col: str,
             lag: pd.Timedelta | None = None) -> pd.DataFrame:
    """For each left row, attach the latest right row with right[availability_col] (+lag) <= left[as_of_col]."""
    if not availability_col:
        raise ValueError("pit_join requires an explicit availability_col (when the data became public); "
                         "refusing to join on event time. Escalate if the availability rule is unknown.")
    if availability_col not in right.columns:
        raise ValueError(f"availability_col {availability_col!r} not in right table columns")
    if right[availability_col].isna().any():
        raise ValueError(f"availability_col {availability_col!r} has {int(right[availability_col].isna().sum())} "
                         "nulls; resolve availability for every row before joining")
    r = right.copy()
    avail = "__available_at"
    r[avail] = r[availability_col] + (lag or pd.Timedelta(0))
    out = pd.merge_asof(left.sort_values(as_of_col), r.sort_values(avail), left_on=as_of_col, right_on=avail,
                        by=key, direction="backward", allow_exact_matches=True)
    return out.drop(columns=[avail])
