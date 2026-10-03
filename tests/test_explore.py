"""Pre-PIT analysis tools on small synthetic tables with known answers."""
import numpy as np
import pandas as pd

from quantaccelerator.tools.explore import (breadth_stats, break_stats, category_mix_stats, coverage_stats, distribution_stats,
                               key_stats, level_shifts, missingness_stats, update_stats, variance_split_stats)

DAYS = pd.bdate_range("2024-01-01", periods=200)


def panel(n_ent=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(d, f"E{e}", float(e + rng.normal(0, .1))) for d in DAYS for e in range(n_ent)]
    return pd.DataFrame(rows, columns=["t", "id", "x"])


def test_level_shift_finds_the_break_date():
    s = pd.Series(np.r_[np.full(100, 10.0), np.full(100, 20.0)] + np.random.default_rng(1).normal(0, .5, 200),
                  index=DAYS)
    b = level_shifts(s, 20)
    assert len(b) == 1 and b[0]["date"] == str(DAYS[100].date()) and b[0]["change_pct"] > 80
    assert level_shifts(pd.Series(np.random.default_rng(2).normal(0, 1, 200), index=DAYS), 20) == []


def test_coverage_counts_entries_and_exits():
    df = panel()
    df = df[~((df.id == "E0") & (df.t >= DAYS[150]))]            # E0 leaves
    df = pd.concat([df, pd.DataFrame({"t": DAYS[60:], "id": "NEW", "x": 1.0})])  # NEW enters
    out, per = coverage_stats(df, "t", "id", "M")
    assert out["n_entities"] == 21 and per.new.sum() == 21 and per.lost.sum() == 21
    assert out["entities_per_period"]["max"] == 21


def test_missingness_keeps_null_zero_and_absent_apart():
    df = panel(4)
    df.loc[df.index[:10], "x"] = np.nan
    df.loc[df.index[10:30], "x"] = 0.0
    df = df[~((df.id == "E1") & df.t.isin(DAYS[50:60]))]          # 10 absent days inside E1's history
    out, _, presence = missingness_stats(df, ["x"], "t", "id", "M")
    f = out["fields"]["x"]
    assert f["null_count"] == 10 and f["zero_count"] == 20
    assert out["panel"]["absent_inside_span_share"] == round(10 / 800, 4)
    assert presence["E1"] == 190 / 200 and presence["E0"] == 1.0


def test_distribution_flags_skew_zero_mass_and_caps():
    rng = np.random.default_rng(0)
    x = pd.Series(np.r_[rng.lognormal(0, 2, 9000), np.zeros(1000)])
    d = distribution_stats(x)
    assert d["zero_share"] == 0.1 and d["negative_share"] == 0 and d["skew"] > 2
    assert any("log1p" in h for h in d["hints"]) and any("zero" in h for h in d["hints"])
    capped = distribution_stats(pd.Series(np.r_[rng.uniform(0, 1, 900), np.ones(100)]))
    assert capped["share_at_max"] == 0.1 and any("maximum" in h for h in capped["hints"])


def test_update_dynamics_and_variance_split():
    df = panel(10)
    df["step"] = df.groupby("id").cumcount() // 10                 # changes every 10 days
    u = update_stats(df, "id", "t", "step")
    assert abs(u["share_unchanged_vs_previous_obs"] - 0.9) < 0.01 and u["autocorrelation"]["lag1"] > 0.9
    v = variance_split_stats(df, "id", "t", "x")                  # x = entity id + tiny noise: all cross-sectional
    assert v["share_between_entities"] > 0.99 and "across entities" in v["reading"]


def test_breaks_keys_breadth_and_categories():
    df = panel(10)
    df.loc[df.t >= DAYS[120], "x"] += 100.0                        # a level shift in the field's median
    out, _ = break_stats(df, "t", "x", "id", "D", ["rows", "median"], 20)
    assert [b["date"] for b in out["breaks"]["median"]] == [str(DAYS[120].date())] and "rows" not in out["breaks"]
    dup = pd.concat([df, df.head(3)])
    k = key_stats(dup, ["t", "id"])
    assert not k["key_is_unique"] and k["exact_duplicate_rows"] == 6 and k["rows_with_duplicated_key"] == 6
    b, _ = breadth_stats(df, "id", "t", "x")
    assert b["n_entities"] == 10 and b["distinct_dates"] == 200 and b["concentration"]["top10_share"] == 1.0
    m = pd.DataFrame({"t": np.repeat(DAYS, 2), "code": ["Q,N"] * 200 + ["Q,N,B"] * 200})
    c, _ = category_mix_stats(m, "t", "code", "D", ",")
    assert c["values_appearing_after_start"] == {"B": str(DAYS[100].date())} and c["values"]["Q"]["mean_share"] == 1.0
