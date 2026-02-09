from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .types import InformationReport


def _df_to_dict(df: pd.DataFrame) -> dict:
    return df.reset_index().to_dict(orient="list")


def save_report_json(report: InformationReport, path: str | Path) -> None:
    payload = {
        "ic": _df_to_dict(report.ic),
        "mi": _df_to_dict(report.mi),
        "te": _df_to_dict(report.te),
        "drift": _df_to_dict(report.drift),
        "redundancy": _df_to_dict(report.redundancy) if report.redundancy is not None else None,
        "metadata": report.metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_report_json(path: str | Path) -> InformationReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ic = pd.DataFrame(payload["ic"]).set_index("feature")
    mi = pd.DataFrame(payload["mi"]).set_index("feature")
    te = pd.DataFrame(payload["te"]).set_index("feature")
    drift = pd.DataFrame(payload["drift"]).set_index("feature")
    redundancy = None
    if payload.get("redundancy") is not None:
        redundancy = pd.DataFrame(payload["redundancy"]).set_index("index")
    return InformationReport(ic=ic, mi=mi, te=te, drift=drift, redundancy=redundancy, metadata=payload["metadata"])
