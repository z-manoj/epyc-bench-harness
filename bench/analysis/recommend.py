"""Recommendation table: best VALID configuration per scenario / precision / engine / backend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bench.core import status as S
from bench.core.config import ExperimentConfig


def _gmean(a: pd.Series) -> float:
    a = a.dropna()
    a = a[a > 0]
    return float(np.exp(np.log(a).mean())) if len(a) else float("nan")


def recommendations(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Best VALID config per {scenario} x {BF16, 8-bit} x {engine} x {ZenDNN on/off}.

    online_single: geometric mean of per-user tok/s (1/TPOT p50) across valid points.
    online_multi:  mean goodput output tok/s across (prompt, output) combinations.
    batch:         geometric mean of total tok/s across valid points.
    """
    scs = {sc.id: sc for sc in cfg.server_configs}
    rows = []
    valid = df[df["classification"].isin(list(S.VALID_CLASSES))]
    for (scen, pc, eng, accel), g in valid.groupby(["scenario", "precision_class", "engine", "accel"]):
        scored = []
        for sid, gg in g.groupby("server_config_id"):
            if scen == "online_single":
                score, head = _gmean(gg["per_user_tok_s_p50"]), (
                    f"per-user {_gmean(gg['per_user_tok_s_p50']):.1f} tok/s; TTFT p50 gmean "
                    f"{_gmean(gg['ttft_p50_ms']):.0f} ms")
            elif scen == "online_multi":
                gp = gg[gg["is_goodput"]]
                score = float(gp["output_tok_s"].mean()) if not gp.empty else 0.0
                head = (f"goodput mean {score:.1f} tok/s; conc "
                        f"{sorted(gp['concurrency'].astype(int).tolist())}" if not gp.empty else "no SLA-meeting point")
            else:
                score = _gmean(gg["total_tok_s"])
                head = f"total {score:.1f} tok/s (gmean); {_gmean(gg['output_tok_s_per_core']):.2f} out tok/s/core"
            scored.append((score, sid, head, gg["max_cv"].max(), len(gg)))
        scored = [s for s in scored if s[0] == s[0]]
        if not scored:
            continue
        score, sid, head, cv, n = max(scored)
        sc = scs[sid]
        rows.append({"scenario": scen, "precision": pc, "engine": eng, "accel": accel, "server_config": sid,
                     "topology": sc.topology, "cores": ";".join(i.cores for i in sc.instances),
                     "flags": " ".join(sc.args), "env": " ".join(f"{k}={v}" for k, v in sc.env.items()),
                     "headline": head, "valid_points": n, "max_cv": cv})
    return pd.DataFrame(rows)


