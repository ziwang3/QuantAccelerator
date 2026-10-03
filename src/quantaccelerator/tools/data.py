"""Named tables the agents may inspect (read-only views of data/interim), from the active dataset profile."""
from functools import lru_cache

import pandas as pd

from quantaccelerator.datasets import active

_P = active()
TABLES = {name: (_P.dataset_id, path) for name, path in _P.tables.items()}  # name -> (dataset id, parquet path)
KEY = dict(_P.keys)  # join key per table


def table_path(name: str):
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; available: {sorted(TABLES)}")
    return TABLES[name][1]


@lru_cache(maxsize=8)
def load_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(table_path(name))


def key_of(name: str) -> str:
    return KEY.get(name, "ACCESSION_NUMBER")
