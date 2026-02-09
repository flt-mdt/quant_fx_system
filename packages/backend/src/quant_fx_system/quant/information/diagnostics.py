from __future__ import annotations

import pandas as pd

from .types import InformationReport


def summary_table(report: InformationReport) -> pd.DataFrame:
    ic = report.ic[["ic_mean"]].rename(columns={"ic_mean": "ic"})
    mi = report.mi[["mi"]]
    te = report.te[["te"]] if not report.te.empty else pd.DataFrame(index=ic.index)
    drift = report.drift[["psi"]] if not report.drift.empty else pd.DataFrame(index=ic.index)
    summary = ic.join(mi, how="outer").join(te, how="outer").join(drift, how="outer")
    return summary
