"""YAML config loading, schema validation and expansion into the run matrix."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigError(ValueError):
    pass


def parse_cores(spec: str | list[int] | int) -> list[int]:
    """Parse a cpulist like "2-33,40,42-43" into a sorted list of core ids."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return sorted({int(c) for c in spec})
    cores: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ConfigError(f"invalid core spec {spec!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise ConfigError(f"invalid core range {part!r} in {spec!r}")
        cores.update(range(lo, hi + 1))
    if not cores:
        raise ConfigError(f"empty core spec {spec!r}")
    return sorted(cores)


def format_cores(cores: list[int]) -> str:
    """Inverse of parse_cores: [2,3,4,7] -> "2-4,7"."""
    cores = sorted(set(cores))
    out, i = [], 0
    while i < len(cores):
        j = i
        while j + 1 < len(cores) and cores[j + 1] == cores[j] + 1:
            j += 1
        out.append(str(cores[i]) if i == j else f"{cores[i]}-{cores[j]}")
        i = j + 1
    return ",".join(out)


_ALLOWED_FUNCS = {"max": max, "min": min, "int": int, "round": round}


def eval_expr(expr: str | int, variables: dict[str, Any]) -> int:
    """Safely evaluate a small arithmetic expression such as "max(50, 5*concurrency)"."""
    if isinstance(expr, (int, float)):
        return int(expr)
    tree = ast.parse(str(expr), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                raise ConfigError(f"function not allowed in expression {expr!r}")
        elif isinstance(node, ast.Name):
            if node.id not in _ALLOWED_FUNCS and node.id not in variables:
                raise ConfigError(f"unknown name {node.id!r} in expression {expr!r}")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
                                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                                   ast.Pow, ast.USub, ast.UAdd)):
            raise ConfigError(f"unsupported syntax {type(node).__name__} in expression {expr!r}")
    value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {**_ALLOWED_FUNCS, **variables})
    return int(value)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Platform(Strict):
    harness_cores: list[int] = Field(default_factory=lambda: [0, 1])
    expected_smt: Optional[Literal["on", "off"]] = "on"
    expected_governor: Optional[str] = "performance"
    require_numactl: bool = True

    @field_validator("expected_smt", mode="before")
    @classmethod
    def _yaml_bool(cls, v: Any) -> Any:
        # YAML 1.1 parses bare on/off as booleans.
        return {True: "on", False: "off"}.get(v, v) if isinstance(v, bool) else v


class RunPolicy(Strict):
    reps: int = Field(3, ge=1)
    max_extra_reps: int = Field(2, ge=0)
    max_failed_reps: int = Field(3, ge=0)
    stability_window_s: float = Field(30, gt=0)
    stability_max_retries: int = Field(3, ge=0)
    cooldown_s: float = 15
    sample_interval_ms: float = Field(50, gt=0)
    server_ready_timeout_s: float = Field(900, gt=0)
    request_timeout_s: float = Field(1800, gt=0)
    restart_server_between_reps: bool = False
    system_check_window_s: Optional[float] = None   # defaults to stability_window_s


class Warmup(Strict):
    server_warmup_requests: int = Field(20, ge=0)
    per_rep_warmup_requests: int = Field(5, ge=0)
    per_rep_warmup_min_s: float = Field(10, ge=0)


class SLA(Strict):
    ttft_p95_ms: float = 2000
    tpot_p95_ms: float = 200


class SystemCheckThresholds(Strict):
    max_mean_util_pct: float = 2.0
    max_foreign_proc_cpu_pct: float = 1.0


class GateThresholds(Strict):
    max_server_util_mean_pct: float = 3.0
    max_server_util_std_pp: float = 2.0
    max_other_util_mean_pct: float = 2.0
    max_sys_mem_drift_mb: float = 256.0
    max_rss_drift_mb: float = 128.0
    max_missed_frac: float = 0.01
    max_freq_dev_frac: float = 0.05


class ValidationThresholds(Strict):
    max_error_frac: float = 0.01
    min_complete_frac: float = 0.99
    prompt_warn_frac: float = 0.02
    prompt_fail_frac: float = 0.05
    max_drift_frac: float = 0.05
    max_outside_util_pct: float = 5.0
    max_freq_drop_frac: float = 0.10
    max_missed_frac: float = 0.01
    mem_growth_warn_mb: float = 256.0
    recovery_margin_pp: float = 2.0
    max_cooldown_factor: float = 2.0
    oom_min_available_frac: float = 0.01


class ConsistencyThresholds(Strict):
    output_tok_s_cv: float = 0.05
    total_tok_s_cv: float = 0.05
    ttft_p50_cv: float = 0.05
    tpot_p50_cv: float = 0.05
    itl_p50_cv: float = 0.05
    ttft_p95_cv: float = 0.10
    tpot_p95_cv: float = 0.10
    e2e_p99_cv: float = 0.15
    cpu_util_cv: float = 0.05
    peak_rss_cv: float = 0.03
    outlier_std: float = 2.0
    outlier_frac: float = 0.07

    def metric_thresholds(self) -> dict[str, tuple[float, bool]]:
        """metric key in rep summary -> (cv threshold, gating)."""
        return {
            "output_tok_s": (self.output_tok_s_cv, True),
            "total_tok_s": (self.total_tok_s_cv, True),
            "ttft_p50_ms": (self.ttft_p50_cv, True),
            "tpot_p50_ms": (self.tpot_p50_cv, True),
            "itl_p50_ms": (self.itl_p50_cv, True),
            "ttft_p95_ms": (self.ttft_p95_cv, True),
            "tpot_p95_ms": (self.tpot_p95_cv, True),
            "e2e_p99_ms": (self.e2e_p99_cv, False),
            "cpu_server_util_mean": (self.cpu_util_cv, True),
            "rss_peak_mb": (self.peak_rss_cv, True),
        }


class Thresholds(Strict):
    system_check: SystemCheckThresholds = Field(default_factory=SystemCheckThresholds)
    gate: GateThresholds = Field(default_factory=GateThresholds)
    validation: ValidationThresholds = Field(default_factory=ValidationThresholds)
    consistency: ConsistencyThresholds = Field(default_factory=ConsistencyThresholds)


class PlanSettings(Strict):
    budget_hours: float = 72.0
    launch_estimate_s: dict[str, float] = Field(
        default_factory=lambda: {"vllm": 300.0, "llamacpp": 60.0, "mock": 3.0})
    server_warmup_request_s: float = 10.0


class Instance(Strict):
    cores: str
    numa_node: int = Field(0, ge=0)

    @field_validator("cores", mode="before")
    @classmethod
    def _cores_str(cls, v: Any) -> str:
        if isinstance(v, list):
            return format_cores([int(x) for x in v])
        return str(v)

    @property
    def core_list(self) -> list[int]:
        return parse_cores(self.cores)


Engine = Literal["vllm", "llamacpp", "mock"]
ServeMode = Literal["launch", "attach"]


class EndpointAddr(Strict):
    """An already-running OpenAI-compatible server. Used only when mode is attach."""
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)


class ServerConfig(Strict):
    id: str
    engine: Engine
    model: str
    precision: str
    # "launch": the harness starts and stops the server. "attach": connect to one that is already up;
    # the harness never execs or signals it. engine/accel are labels for the client and the report.
    mode: ServeMode = "launch"
    # "zendnn": llama.cpp ggml-zendnn / vLLM zentorch. "none": plain CPU.
    # In launch mode, llama.cpp "none" needs a binary built with -DGGML_ZENDNN=OFF (the device cannot
    # be turned off at runtime). In attach mode the label is not checked against the process.
    accel: Literal["zendnn", "none"] = "zendnn"
    # Configs sharing an accel_group are paired for speedup analysis. Without it, configs are paired
    # when engine, precision, model, cores, args and env all match (see analysis.acceleration).
    accel_group: Optional[str] = None
    binary: Optional[str] = None
    served_model_name: Optional[str] = None
    host: str = "127.0.0.1"
    port_base: int = Field(ge=1, le=65000)
    # attach: explicit targets. Empty means host:port_base (and port_base+i when several instances).
    endpoints: list[EndpointAddr] = Field(default_factory=list)
    # launch: required (pinning). attach: optional; when set, CPU gates attribute these cores.
    instances: list[Instance] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    slots_per_instance: Optional[int] = Field(None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _port_alias(cls, data: Any) -> Any:
        # `port` is the attach-mode name; the field stays `port_base` so ServerConfig.port(idx) remains.
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "port" in data:
            port = data.pop("port")
            if data.get("port_base") not in (None, port):
                raise ValueError(f"port {port} and port_base {data.get('port_base')} disagree")
            data["port_base"] = port
        if "port_base" not in data:
            eps = data.get("endpoints") or []
            if eps and isinstance(eps[0], dict) and "port" in eps[0]:
                data["port_base"] = eps[0]["port"]
        return data

    @model_validator(mode="after")
    def _mode_requirements(self) -> "ServerConfig":
        if self.mode == "attach":
            if self.endpoints and self.instances and len(self.endpoints) != len(self.instances):
                raise ValueError(
                    f"server config {self.id!r}: attach instances ({len(self.instances)}) and "
                    f"endpoints ({len(self.endpoints)}) must be the same length")
            if not self.endpoints:
                n = len(self.instances) or 1
                self.endpoints = [EndpointAddr(host=self.host, port=self.port_base + i) for i in range(n)]
        else:
            if self.endpoints:
                raise ValueError(f"server config {self.id!r}: endpoints require mode: attach")
            if not self.instances:
                raise ValueError(f"server config {self.id!r}: launch mode requires instances")
        return self

    @field_validator("id")
    @classmethod
    def _id_safe(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", v):
            raise ValueError(f"server config id {v!r} must be filesystem-safe")
        return v

    @field_validator("args", mode="before")
    @classmethod
    def _args_str(cls, v: Any) -> Any:
        return [str(a) for a in v] if isinstance(v, list) else v

    @field_validator("env", mode="before")
    @classmethod
    def _env_str(cls, v: Any) -> Any:
        return {str(k): str(val) for k, val in v.items()} if isinstance(v, dict) else v

    def arg_value(self, *names: str) -> Optional[str]:
        for i, a in enumerate(self.args):
            for n in names:
                if a == n and i + 1 < len(self.args):
                    return self.args[i + 1]
                if a.startswith(n + "="):
                    return a.split("=", 1)[1]
        return None

    @property
    def slots(self) -> int:
        """Concurrent request slots per instance."""
        if self.slots_per_instance:
            return self.slots_per_instance
        if self.engine == "llamacpp":
            v = self.arg_value("-np", "--parallel")
            return int(v) if v else 1
        if self.engine == "vllm":
            v = self.arg_value("--max-num-seqs")
            return int(v) if v else 256
        v = self.arg_value("--parallel")
        return int(v) if v else 8

    @property
    def n_endpoints(self) -> int:
        return len(self.endpoints) if self.mode == "attach" else len(self.instances)

    @property
    def total_slots(self) -> int:
        return self.slots * self.n_endpoints

    @property
    def all_cores(self) -> list[int]:
        return sorted({c for inst in self.instances for c in inst.core_list})

    @property
    def topology(self) -> str:
        if not self.instances:
            return "external" if self.n_endpoints == 1 else f"{self.n_endpoints}xexternal"
        sizes = {len(i.core_list) for i in self.instances}
        per = f"{sizes.pop()}c" if len(sizes) == 1 else "mixed"
        return f"{len(self.instances)}x{per}"

    def port(self, idx: int) -> int:
        if self.mode == "attach" and self.endpoints:
            return self.endpoints[idx].port
        return self.port_base + idx


Scenario = Literal["online_single", "online_multi", "batch"]


class Workload(Strict):
    scenario: Scenario
    prompt_sizes: list[Union[int, Literal["mixed"]]]
    output_lens: list[int]
    concurrency: list[Union[int, Literal["max"]]]
    num_requests: Union[int, str]
    min_measure_s: float = Field(0, ge=0)

    @field_validator("output_lens")
    @classmethod
    def _pos(cls, v: list[int]) -> list[int]:
        if any(x < 1 for x in v):
            raise ValueError("output_lens must be >= 1")
        return v

    @model_validator(mode="after")
    def _check(self) -> "Workload":
        if self.scenario == "online_single" and any(c != 1 for c in self.concurrency):
            raise ValueError("online_single requires concurrency [1]")
        # Validate the expression early with a representative value.
        eval_expr(self.num_requests, {"concurrency": 1})
        return self


class ExperimentConfig(Strict):
    experiment_id: str
    results_dir: str = "./results"
    dataset_dir: str = "./data"
    dev_mode: bool = False          # relaxes range checks (short windows) for tests / selftest
    platform: Platform = Field(default_factory=Platform)
    run_policy: RunPolicy = Field(default_factory=RunPolicy)
    warmup: Warmup = Field(default_factory=Warmup)
    sla: SLA = Field(default_factory=SLA)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    plan: PlanSettings = Field(default_factory=PlanSettings)
    server_configs: list[ServerConfig] = Field(min_length=1)
    workloads: list[Workload] = Field(min_length=1)

    @field_validator("experiment_id")
    @classmethod
    def _id_safe(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", v):
            raise ValueError(f"experiment_id {v!r} must be filesystem-safe")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> "ExperimentConfig":
        rp = self.run_policy
        if not self.dev_mode:
            if not (10 <= rp.cooldown_s <= 15):
                raise ValueError("run_policy.cooldown_s must be within 10-15 s")
        elif rp.cooldown_s <= 0:
            raise ValueError("run_policy.cooldown_s must be > 0")

        harness = set(self.platform.harness_cores)
        if not harness:
            raise ValueError("platform.harness_cores must not be empty")
        ids: set[str] = set()
        for sc in self.server_configs:
            if sc.id in ids:
                raise ValueError(f"duplicate server config id {sc.id!r}")
            ids.add(sc.id)
            seen: set[int] = set()
            for i, inst in enumerate(sc.instances):
                cores = set(inst.core_list)
                if cores & harness:
                    raise ValueError(
                        f"server config {sc.id!r} instance {i} cores {inst.cores} overlap "
                        f"harness_cores {sorted(harness)}")
                if cores & seen:
                    raise ValueError(f"server config {sc.id!r}: instance {i} cores overlap another instance")
                seen |= cores
            if sc.mode == "launch" and sc.engine == "llamacpp" and not sc.binary:
                raise ValueError(f"server config {sc.id!r}: launch mode llamacpp requires `binary`")
            if sc.mode == "launch" and sc.engine == "mock" and not sc.binary:
                raise ValueError(f"server config {sc.id!r}: launch mode mock requires `binary`")
        return self


# ---------------------------------------------------------------------------
# Expansion


@dataclass(frozen=True)
class WorkloadPoint:
    scenario: str
    prompt_size: Union[int, str]
    output_len: int
    concurrency: int
    num_requests: int
    min_measure_s: float

    @property
    def point_id(self) -> str:
        return f"{self.scenario}_p{self.prompt_size}_o{self.output_len}_c{self.concurrency}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id, "scenario": self.scenario, "prompt_size": self.prompt_size,
            "output_len": self.output_len, "concurrency": self.concurrency,
            "num_requests": self.num_requests, "min_measure_s": self.min_measure_s,
        }


@dataclass
class RunMatrix:
    config: ExperimentConfig
    points: dict[str, list[WorkloadPoint]] = field(default_factory=dict)   # server_config_id -> points


def expand_points(cfg: ExperimentConfig, sc: ServerConfig) -> list[WorkloadPoint]:
    points: list[WorkloadPoint] = []
    seen: set[str] = set()
    for wl in cfg.workloads:
        for conc_spec in wl.concurrency:
            conc = sc.total_slots if conc_spec == "max" else int(conc_spec)
            if conc < 1:
                raise ConfigError(f"concurrency must be >= 1 (got {conc_spec})")
            nreq = eval_expr(wl.num_requests, {"concurrency": conc})
            if nreq < 1:
                raise ConfigError(f"num_requests evaluated to {nreq} for concurrency {conc}")
            for ps in wl.prompt_sizes:
                for ol in wl.output_lens:
                    p = WorkloadPoint(wl.scenario, ps, ol, conc, nreq, wl.min_measure_s)
                    if p.point_id in seen:
                        continue
                    seen.add(p.point_id)
                    points.append(p)
    return points


def expand(cfg: ExperimentConfig) -> RunMatrix:
    m = RunMatrix(cfg)
    for sc in cfg.server_configs:
        m.points[sc.id] = expand_points(cfg, sc)
    return m


def load_config(path: str | os.PathLike) -> ExperimentConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    cfg = parse_config(raw)
    # Relative paths are resolved against the config file location.
    base = path.resolve().parent
    upd = {}
    for key in ("results_dir", "dataset_dir"):
        p = Path(getattr(cfg, key))
        if not p.is_absolute():
            upd[key] = str((base / p).resolve())
    return cfg.model_copy(update=upd)


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    try:
        cfg = ExperimentConfig.model_validate(raw)
    except Exception as e:  # pydantic.ValidationError
        raise ConfigError(str(e)) from e
    expand(cfg)  # surface expansion errors at validation time
    return cfg


def resolved_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Fully resolved config (defaults applied, expressions and 'max' expanded)."""
    d = cfg.model_dump(mode="json")
    m = expand(cfg)
    d["resolved"] = {
        sc.id: {
            "mode": sc.mode,
            "total_slots": sc.total_slots,
            "topology": sc.topology,
            "accel": sc.accel,
            "endpoints": [f"http://{e.host}:{e.port}" for e in sc.endpoints],
            "instances": [{"cores": inst.cores, "core_list": inst.core_list,
                           "numa_node": inst.numa_node, "port": sc.port(i)}
                          for i, inst in enumerate(sc.instances)],
            "points": [p.as_dict() for p in m.points[sc.id]],
        }
        for sc in cfg.server_configs
    }
    return d


def save_resolved(cfg: ExperimentConfig, path: str | os.PathLike) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(resolved_dict(cfg), f, sort_keys=False)


def load_resolved(path: str | os.PathLike) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw.pop("resolved", None)
    return parse_config(raw)
