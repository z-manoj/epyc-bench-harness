"""End-to-end tests against tests/mock_server.py (acceptance criteria 1, 3 and 4)."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from bench.core import status as S
from bench.analysis.aggregate import analyze
from bench.core.io import load_events
from bench.runner.experiment import Experiment
from bench.runner.storage import rep_history
from bench.core.config import parse_config
from bench.selftest import MOCK_SERVER, REPO, check_outputs, mock_config, write_synthetic_dataset

pytestmark = pytest.mark.slow

SINGLE = [{"scenario": "online_single", "prompt_sizes": [64], "output_lens": [16], "concurrency": [1],
           "num_requests": 6, "min_measure_s": 1.0}]


@pytest.fixture(autouse=True)
def restore_affinity():
    aff = os.sched_getaffinity(0)
    yield
    os.sched_setaffinity(0, aff)


@pytest.fixture
def root(tmp_path):
    write_synthetic_dataset(tmp_path / "data", [32, 64])
    return tmp_path


def statuses(exp_dir, cfg):
    sc = cfg.server_configs[0]
    return {p.name: [h["status"] for h in rep_history(p)] for p in sorted((exp_dir / sc.id).glob("*_p*"))}


def test_full_lifecycle_three_reps(root):
    cfg = mock_config(root, "e2e_full", port_base=18610)
    exp_dir = Path(cfg.results_dir) / cfg.experiment_id
    asyncio.run(Experiment(cfg, exp_dir).run())
    analyze(exp_dir)
    assert check_outputs(exp_dir, cfg) == []
    st = statuses(exp_dir, cfg)
    assert st and all(len(v) >= 3 for v in st.values()), st
    # phase markers present for every rep, in order
    ev = load_events(exp_dir / "mock_2c" / "events.jsonl")
    for pid, reps in st.items():
        for rep in range(1, len(reps) + 1):
            phases = [e["phase"] for e in ev if e.get("point_id") == pid and e.get("rep") == rep
                      and e["event"] == "start"]
            if reps[rep - 1] == S.SUCCESS:
                assert phases == ["rep", "gate", "warmup", "measure", "cooldown"], (pid, rep, phases)


def test_crash_yields_failed_crash_and_relaunch(root):
    marker = root / "crashed.marker"
    # 4 one-time warmup + 3 per-rep warmup + 2 measured requests -> crash in rep 1 measurement
    cfg = mock_config(root, "e2e_crash", port_base=18630, workloads=SINGLE,
                      mock_args=["--crash-after", "9", "--crash-once-file", str(marker)])
    exp_dir = Path(cfg.results_dir) / cfg.experiment_id
    asyncio.run(Experiment(cfg, exp_dir).run())
    assert marker.exists()
    st = statuses(exp_dir, cfg)
    (reps,) = st.values()
    assert reps[0] == S.FAILED_CRASH, reps
    assert reps[1:].count(S.SUCCESS) >= 3, reps
    launch = json.loads((exp_dir / "mock_2c" / "launch.json").read_text())
    assert len(launch["history"]) == 2
    df = analyze(exp_dir)
    assert df.iloc[0]["classification"] in S.VALID_CLASSES
    assert df.iloc[0]["n_failed"] == 1


def _accel_server_configs(cfg, specs):
    base = cfg.server_configs[0].model_dump(mode="json")
    out = []
    for i, (sid, accel, backend_args) in enumerate(specs):
        out.append({**base, "id": sid, "accel": accel, "accel_group": "mock", "port_base": base["port_base"] + 10 * i,
                    "args": [*base["args"], *backend_args]})
    return out


def test_accel_pair_speedup_and_reports(root):
    from bench.reporting.report import report
    cfg = mock_config(root, "e2e_accel", port_base=18670, workloads=SINGLE)
    raw = cfg.model_dump(mode="json")
    raw["server_configs"] = _accel_server_configs(cfg, [
        ("mock_zendnn", "zendnn", ["--backend", "zendnn", "--accel-speedup", "2"]),
        ("mock_cpu", "none", ["--backend", "cpu"])])
    cfg = type(cfg).model_validate(raw)
    exp_dir = Path(cfg.results_dir) / cfg.experiment_id
    asyncio.run(Experiment(cfg, exp_dir).run())
    md = report(exp_dir, html=True, timelines="none")
    csv = md.parent / "csv"
    for name in ("points", "reps", "deviation", "recommendations", "acceleration", "acceleration_summary"):
        assert (csv / f"{name}.csv").exists(), name
    assert (md.parent / "report.html").exists()
    import pandas as pd
    sp = pd.read_csv(csv / "acceleration.csv")
    assert len(sp) == 1 and bool(sp.iloc[0]["both_valid"])
    # TPOT is halved by the mock; loose bounds absorb harness overhead on a busy machine
    assert 1.5 < sp.iloc[0]["tpot_p50_ms_speedup"] < 2.5, sp.iloc[0].to_dict()
    assert sp.iloc[0]["output_tok_s_speedup"] > 1.3
    rec = pd.read_csv(csv / "recommendations.csv")
    assert set(rec["accel"]) == {"zendnn", "none"}


def test_attach_benchmarks_running_server(root):
    port = 18720
    proc = subprocess.Popen(
        [sys.executable, str(MOCK_SERVER), "--host", "127.0.0.1", "--port", str(port),
         "--parallel", "4", "--tpot-ms", "8", "--ttft-ms", "15", "--close-after-stream", "--backend", "cpu"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"mock server exited {proc.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("mock server did not listen")
        cfg = mock_config(root, "e2e_attach", workloads=SINGLE, reps=1)
        raw = cfg.model_dump(mode="json")
        raw["run_policy"]["max_extra_reps"] = 0
        # Host-wide /proc sampling can miss a 1s window on a large machine; this test checks the client.
        raw["thresholds"]["gate"]["max_missed_frac"] = 1.0
        raw["thresholds"]["validation"]["max_missed_frac"] = 1.0
        raw["server_configs"] = [{
            "id": "remote_mock", "mode": "attach", "engine": "mock", "accel": "none",
            "model": "mock-model", "precision": "bf16", "host": "127.0.0.1", "port": port,
            "slots_per_instance": 4,
        }]
        cfg = parse_config(raw)
        exp_dir = Path(cfg.results_dir) / cfg.experiment_id
        asyncio.run(Experiment(cfg, exp_dir).run())
        assert proc.poll() is None
        st = json.loads((exp_dir / "remote_mock" / "config_status.json").read_text())
        assert st["status"] == S.CONFIG_DONE, st
        reps = statuses(exp_dir, cfg)
        assert reps and all(v and all(s == S.SUCCESS for s in v) for v in reps.values()), reps
        launch = json.loads((exp_dir / "remote_mock" / "launch.json").read_text())
        assert launch["mode"] == "attach" and launch["instances"][0]["pid"] is None
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)


def test_backend_mismatch_is_invalid_backend(root):
    cfg = mock_config(root, "e2e_mismatch", port_base=18690, workloads=SINGLE)
    raw = cfg.model_dump(mode="json")
    raw["server_configs"] = _accel_server_configs(cfg, [("mock_claims_none", "none", ["--backend", "zendnn"])])
    cfg = type(cfg).model_validate(raw)
    exp_dir = Path(cfg.results_dir) / cfg.experiment_id
    asyncio.run(Experiment(cfg, exp_dir).run())
    st = json.loads((exp_dir / "mock_claims_none" / "config_status.json").read_text())
    assert st["status"] == S.INVALID_BACKEND, st
    assert not list((exp_dir / "mock_claims_none").glob("*/rep_*/summary.json"))


def _write_yaml(cfg, path):
    d = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(d))


def test_interrupt_then_resume_no_duplicates(root):
    cfg = mock_config(root, "e2e_resume", port_base=18650)
    exp_dir = Path(cfg.results_dir) / cfg.experiment_id
    cfg_path = root / "cfg.yaml"
    _write_yaml(cfg, cfg_path)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([sys.executable, "-m", "bench", "run", "--config", str(cfg_path)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sc_dir = exp_dir / "mock_2c"
    deadline = time.time() + 120
    # interrupt in the middle of the second rep of the first point
    while time.time() < deadline:
        done = list(sc_dir.glob("*/rep_*/summary.json"))
        if done and list(sc_dir.glob("*/rep_2/gate.json")):
            break
        time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("experiment did not progress")
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 130, out.decode()[-2000:]
    before = {str(p.parent.relative_to(sc_dir)) for p in sc_dir.glob("*/rep_*/summary.json")}

    r = subprocess.run([sys.executable, "-m", "bench", "resume", "--experiment", str(exp_dir)], env=env,
                       capture_output=True, timeout=300)
    assert r.returncode == 0, r.stdout.decode()[-3000:]
    after = {str(p.parent.relative_to(sc_dir)) for p in sc_dir.glob("*/rep_*/summary.json")}
    assert before <= after
    for pdir in sorted(p for p in sc_dir.glob("*_p*") if p.is_dir()):
        reps = sorted(int(d.name.split("_")[1]) for d in pdir.glob("rep_*"))
        assert reps == list(range(1, len(reps) + 1)), (pdir.name, reps)      # contiguous, no duplicates
        hist = rep_history(pdir)
        assert len(hist) == len(reps)
        assert sum(h["status"] == S.SUCCESS for h in hist) >= 3
    # rerunning resume is a no-op
    r2 = subprocess.run([sys.executable, "-m", "bench", "resume", "--experiment", str(exp_dir)], env=env,
                        capture_output=True, timeout=120)
    assert r2.returncode == 0
    assert {str(p.parent.relative_to(sc_dir)) for p in sc_dir.glob("*/rep_*/summary.json")} == after
