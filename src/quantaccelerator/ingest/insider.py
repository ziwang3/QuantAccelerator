"""SEC Form 3/4/5 quarterly data set -> parquet (SUBMISSION, NONDERIV_TRANS, REPORTINGOWNER)."""
import zipfile

import pandas as pd

from quantaccelerator.ingest import manifest
from quantaccelerator.paths import INSIDER_ZIP, QUARTER, insider_dir

TABLES = ("SUBMISSION", "NONDERIV_TRANS", "REPORTINGOWNER")
DATE_COLS = {
    "SUBMISSION": ["FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB"],
    "NONDERIV_TRANS": ["TRANS_DATE", "DEEMED_EXECUTION_DATE"],
    "REPORTINGOWNER": [],
}
NUMERIC_COLS = {"NONDERIV_TRANS": ["TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"]}


def read_table(zip_path, table: str) -> pd.DataFrame:
    """Read one TSV from the quarterly zip; dates are DD-MON-YYYY, everything else stays text."""
    with zipfile.ZipFile(zip_path) as z, z.open(f"{table}.tsv") as f:
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=3)
    for c in DATE_COLS.get(table, []):
        df[c] = pd.to_datetime(df[c], format="%d-%b-%Y", errors="coerce")
    for c in NUMERIC_COLS.get(table, []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ingest(quarter: str = QUARTER) -> dict[str, int]:
    src = INSIDER_ZIP.with_name(INSIDER_ZIP.name.format(quarter=quarter))
    out_dir = insider_dir(quarter)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for t in TABLES:
        df = read_table(src, t)
        out = out_dir / f"{t.lower()}.parquet"
        df.to_parquet(out, index=False)
        manifest.record(out, [src], "quantaccelerator.ingest.insider", rows=len(df))
        counts[t] = len(df)
    return counts


if __name__ == "__main__":
    print(ingest())
