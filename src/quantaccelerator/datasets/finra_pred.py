"""Benchmark for Research EDA phase B: the finra_eda panel with an oracle-frozen feature set and synthetic returns
with planted structure (see quantaccelerator.eval.gold_pred)."""
import dataclasses

from quantaccelerator.datasets.finra_eda import PROFILE as EDA
from quantaccelerator.eval import gold_pred

PROFILE = dataclasses.replace(
    EDA, name="finra_pred", title="FINRA short volume · planted-returns benchmark (predictive EDA)",
    predict={"returns": gold_pred.returns_frame, "split": gold_pred.SPLIT, "oracle_features": gold_pred.oracle_features})
