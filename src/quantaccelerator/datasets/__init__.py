"""Dataset profiles: what makes one QuantAccelerator case study different from another.

A profile names the tables and documents the agents may read, the task text each agent receives, the researcher
pipelines under audit, the answer key, and the deterministic functions that measure look-ahead and apply an
availability rule. Tools, agents, the orchestrator and the scorer read the active profile, so adding a case study
means adding a profile, not editing the framework.

One process audits one dataset: the active profile comes from $QA_DATASET (default "form4").
"""
import dataclasses
import importlib
import os
from pathlib import Path
from typing import Callable


@dataclasses.dataclass
class Profile:
    name: str                                  # e.g. "form4"
    title: str                                 # human title of the case study
    dataset_id: str                            # id the agents see, e.g. "sec_form345_2025q1"
    tables: dict[str, Path]                    # table name -> parquet path
    keys: dict[str, str]                       # table name -> join key column
    docs: dict[str, Path]                      # doc id -> .pdf or .txt
    profiler_task: str
    pit_task: str
    audit_task: str                            # formatted with {pipeline}
    pit_tools: list[str]
    audit_tools: list[str]
    pipeline_glob: str                         # e.g. "insider_events_{variant}.py" under notebooks/
    variants: list[str]
    benchmark: Path                            # dir with gold.json (+ rule sample)
    gold_rule: Callable[[], object]            # -> AvailabilityRule
    measure: Callable                          # (events DataFrame, AvailabilityRule) -> look-ahead stats dict
    rule_agreement: Callable                   # (AvailabilityRule) -> share of the rule sample matching gold
    rule_by_stratum: Callable                  # (AvailabilityRule) -> {"overall": x, <stratum>: x, ...}
    check_pit: Callable = lambda contract: []  # dataset-specific evidence checks on a PITContract
    code_list_hint: bool = True                # doc checks may point to pages with code lists
    doc_help: str = ""                         # one line per doc for the read_doc tool description
    view: dict | None = None                   # PIT research view: {"table", "entity", "time", "values": [...]}
    first_tradable: Callable | None = None     # (AvailabilityRule, view-table DataFrame) -> first tradable session
    exchange_tickers: bool = False             # identifiers are exchange tickers (stocks and ETFs look alike)
    panel: dict | None = None                  # research panel (quantaccelerator.tools.panel): universe_n, formation, start, end,
                                               # scale_field, adv_window
    predict: dict | None = None                # Research EDA phase B: {"returns": () -> returns frame (returns.core),
                                               # "split": SplitSpec fields, "oracle_features": optional () -> specs,
                                               # candidates and ideas of a fixed feature set for benchmarks}

    def pipeline(self, variant: str) -> str:
        return f"notebooks/{self.pipeline_glob.format(variant=variant)}"

    def variant_of(self, pipeline: str) -> str:
        prefix, _, suffix = self.pipeline_glob.partition("{variant}")
        return Path(pipeline).name.removeprefix(prefix).removesuffix(suffix)


PROFILES = {"form4": "quantaccelerator.datasets.form4", "alfred": "quantaccelerator.datasets.alfred", "finra": "quantaccelerator.datasets.finra",
            "finra_defects": "quantaccelerator.datasets.finra_defects", "finra_small": "quantaccelerator.datasets.finra_small",
            "finra_eda": "quantaccelerator.datasets.finra_eda", "finra_pred": "quantaccelerator.datasets.finra_pred"}
_cache: dict[str, Profile] = {}


def get(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown dataset profile {name!r}; available: {sorted(PROFILES)}")
    if name not in _cache:
        _cache[name] = importlib.import_module(PROFILES[name]).PROFILE
    return _cache[name]


def active() -> Profile:
    return get(os.environ.get("QA_DATASET", "form4"))
