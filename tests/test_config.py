from pathlib import Path

import pytest

from bench.core.config import (ConfigError, eval_expr, expand, format_cores, load_config, parse_config, parse_cores,
                          resolved_dict)

REPO = Path(__file__).resolve().parent.parent


def base(**over):
    raw = {
        "experiment_id": "t",
        "server_configs": [{
            "id": "v", "engine": "vllm", "model": "/m", "precision": "bf16", "port_base": 8000,
            "instances": [{"cores": "2-33", "numa_node": 0}, {"cores": "34-65", "numa_node": 0}],
            "args": ["--max-num-seqs", "64"],
        }],
        "workloads": [{"scenario": "online_multi", "prompt_sizes": [128], "output_lens": [128],
                       "concurrency": [2, "max"], "num_requests": "max(50, 5*concurrency)",
                       "min_measure_s": 60}],
    }
    raw.update(over)
    return raw


def test_parse_cores():
    assert parse_cores("2-4,7,9-10") == [2, 3, 4, 7, 9, 10]
    assert parse_cores("5") == [5]
    assert format_cores([2, 3, 4, 7, 9, 10]) == "2-4,7,9-10"
    with pytest.raises(ConfigError):
        parse_cores("4-2")
    with pytest.raises(ConfigError):
        parse_cores("a-b")


def test_eval_expr():
    assert eval_expr("max(50, 5*concurrency)", {"concurrency": 4}) == 50
    assert eval_expr("max(50, 5*concurrency)", {"concurrency": 64}) == 320
    assert eval_expr(20, {}) == 20
    for bad in ("__import__('os')", "open('x')", "concurrency.real", "foo + 1"):
        with pytest.raises(ConfigError):
            eval_expr(bad, {"concurrency": 1})


def test_expand_max_and_expression():
    cfg = parse_config(base())
    pts = expand(cfg).points["v"]
    by_c = {p.concurrency: p for p in pts}
    assert set(by_c) == {2, 128}                 # max = 64 slots x 2 instances
    assert by_c[2].num_requests == 50
    assert by_c[128].num_requests == 640
    assert by_c[2].point_id == "online_multi_p128_o128_c2"


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError):
        parse_config(base(bogus=1))
    raw = base()
    raw["run_policy"] = {"reps": 3, "typo_key": 1}
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = base()
    raw["server_configs"][0]["instances"][0]["corez"] = "1"
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_harness_core_overlap_rejected():
    raw = base(platform={"harness_cores": [0, 1, 2]})
    with pytest.raises(ConfigError, match="overlap"):
        parse_config(raw)


def test_instance_core_overlap_rejected():
    raw = base()
    raw["server_configs"][0]["instances"][1]["cores"] = "30-65"
    with pytest.raises(ConfigError, match="overlap"):
        parse_config(raw)


def test_cooldown_range():
    with pytest.raises(ConfigError):
        parse_config(base(run_policy={"cooldown_s": 5}))
    parse_config(base(run_policy={"cooldown_s": 5}, dev_mode=True))
    parse_config(base(run_policy={"cooldown_s": 12}))


def test_online_single_requires_conc_1():
    raw = base()
    raw["workloads"] = [{"scenario": "online_single", "prompt_sizes": [128], "output_lens": [128],
                         "concurrency": [2], "num_requests": 20}]
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_threshold_defaults():
    cfg = parse_config(base())
    t = cfg.thresholds
    assert t.gate.max_server_util_mean_pct == 3 and t.gate.max_server_util_std_pp == 2
    assert t.gate.max_rss_drift_mb == 128 and t.gate.max_sys_mem_drift_mb == 256
    assert t.validation.max_drift_frac == 0.05 and t.validation.max_outside_util_pct == 5
    assert t.consistency.ttft_p95_cv == 0.10 and t.consistency.peak_rss_cv == 0.03
    assert cfg.run_policy.reps == 3 and cfg.run_policy.cooldown_s == 15


def test_llamacpp_slots_and_resolved():
    raw = base()
    raw["server_configs"] = [{
        "id": "l", "engine": "llamacpp", "binary": "/bin/true", "model": "/m.gguf", "precision": "q8_0",
        "port_base": 9000, "instances": [{"cores": "2-65"}], "args": ["-t", "64", "-np", "16"]}]
    cfg = parse_config(raw)
    assert cfg.server_configs[0].total_slots == 16
    d = resolved_dict(cfg)
    assert d["resolved"]["l"]["points"][-1]["concurrency"] == 16


def test_example_config_loads():
    cfg = load_config(REPO / "examples" / "experiment.yaml")
    assert cfg.experiment_id
    assert expand(cfg).points
    attach = load_config(REPO / "examples" / "attach.yaml")
    assert {sc.mode for sc in attach.server_configs} == {"attach"}
    assert all(sc.binary is None and sc.endpoints for sc in attach.server_configs)


def test_attach_needs_only_host_port():
    from bench.analysis.acceleration import accel_pairs
    raw = base(dev_mode=True)
    raw["server_configs"] = [
        {"id": "llama_zen", "mode": "attach", "engine": "llamacpp", "accel": "zendnn",
         "model": "llama-3.1-8b", "precision": "q8_0", "host": "10.0.0.5", "port": 8080,
         "slots_per_instance": 8, "accel_group": "q8"},
        {"id": "llama_cpu", "mode": "attach", "engine": "llamacpp", "accel": "none",
         "model": "llama-3.1-8b", "precision": "q8_0", "host": "10.0.0.6", "port": 8080,
         "slots_per_instance": 8, "accel_group": "q8"},
    ]
    cfg = parse_config(raw)
    zen = cfg.server_configs[0]
    assert zen.endpoints[0].host == "10.0.0.5" and zen.port(0) == 8080
    assert zen.total_slots == 8 and zen.topology == "external" and zen.all_cores == []
    assert [(z.id, n.id) for z, n in accel_pairs(cfg)] == [("llama_zen", "llama_cpu")]


def test_launch_still_requires_binary_and_instances():
    raw = base()
    raw["server_configs"] = [{
        "id": "l", "engine": "llamacpp", "model": "/m.gguf", "precision": "q8_0", "port_base": 9000}]
    with pytest.raises(ConfigError, match="instances"):
        parse_config(raw)
    raw["server_configs"][0]["instances"] = [{"cores": "2-3"}]
    with pytest.raises(ConfigError, match="binary"):
        parse_config(raw)
