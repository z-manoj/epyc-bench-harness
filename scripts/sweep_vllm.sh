#!/usr/bin/env bash
# vLLM with vs without zentorch: long single-stream prompts and concurrent bursts. One server per (env, model);
# every scenario = 1 warmup request + RUNS measured bursts of <concurrency> simultaneous requests.
#
# Usage: scripts/sweep_vllm.sh [config ...]   (default: vllm-zentorch-bf16 vllm-cpu-bf16 vllm-zentorch-w8a8 vllm-cpu-w8a8)
# Env:   SCENARIOS ("prompt:gen:concurrency ...")  SERVER_CPUS (0-95)  SERVER_SMT (192-287)  MEM_NODE (0)
#        CLIENT_CPUS (96-99)  RUNS (3)  OUT
set -uo pipefail
unset LD_PRELOAD CONDA_PREFIX VLLM_CPU_OMP_THREADS_BIND VLLM_CPU_KVCACHE_SPACE TORCHINDUCTOR_FREEZING VLLM_USE_AOT_COMPILE

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SERVER_CPUS=${SERVER_CPUS:-0-95}
SERVER_SMT=${SERVER_SMT:-192-287}
MEM_NODE=${MEM_NODE:-0}
CLIENT_CPUS=${CLIENT_CPUS:-96-99}
RUNS=${RUNS:-3}
SCENARIOS=${SCENARIOS:-"4096:128:1 8192:128:1 1024:128:4 1024:128:16 1024:128:64"}
OUT=${OUT:-$ROOT/results/sweep_vllm_$(date +%Y%m%d_%H%M)}
VZ=${VLLM_BASE:-$HOME/vllm-zen}
MODELS=${MODELS_DIR:-$HOME/mkumar/models}
PY=$VZ/tools/bin/python
PORT=8320
CONFIGS=(vllm-zentorch-bf16 vllm-cpu-bf16 vllm-zentorch-w8a8 vllm-cpu-w8a8)
[ $# -gt 0 ] && CONFIGS=("$@")

mkdir -p "$OUT/logs"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$OUT/sweep.log"; }

stop() {
  [ -n "${SRV_PID:-}" ] || return 0
  kill -TERM -- -$SRV_PID 2>/dev/null
  for _ in $(seq 30); do kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$SRV_PID 2>/dev/null
  SRV_PID=
  sleep 3
}
trap '[ -n "${CLIENT_PID:-}" ] && kill $CLIENT_PID 2>/dev/null; stop; exit 130' INT TERM

{
  echo "date: $(date -Is)"; echo "host: $(hostname)  cpu: $(lscpu | sed -nE 's/Model name: *//p')"
  echo "server cpus: $SERVER_CPUS (+SMT $SERVER_SMT for vLLM frontend)  mem node: $MEM_NODE  client cpus: $CLIENT_CPUS"
  echo "scenarios (prompt:gen:concurrency): $SCENARIOS  runs: $RUNS"
  echo "vllm-zentorch: $($VZ/env/bin/python -m pip list 2>/dev/null | grep -E '^(vllm|torch|zentorch) ' | tr -s ' ' | paste -sd, -)"
  echo "vllm-cpu: $($VZ/env-cpu/bin/python -m pip list 2>/dev/null | grep -E '^(vllm|torch) ' | tr -s ' ' | paste -sd, -)"
} > "$OUT/meta.txt"

max_len=0
for s in $SCENARIOS; do IFS=: read -r p g _ <<<"$s"; (( p + g > max_len )) && max_len=$((p + g)); done
max_len=$(( (max_len + 1023) / 1024 * 1024 ))

for c in "${CONFIGS[@]}"; do
  log "=== $c"
  [[ $c == *zentorch* ]] && env=$VZ/env-zentorch.sh || env=$VZ/env-cpu.sh
  [[ $c == *w8a8 ]] && hfm=$MODELS/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 || hfm=$MODELS/Llama-3.1-8B-Instruct
  CMD="source $env && export VLLM_CPU_OMP_THREADS_BIND=$SERVER_CPUS VLLM_CPU_KVCACHE_SPACE=32 && cd /tmp && exec numactl --physcpubind=$SERVER_CPUS,$SERVER_SMT --membind=$MEM_NODE vllm serve $hfm --served-model-name llama --dtype bfloat16 --max-model-len $max_len --no-enable-prefix-caching --host 127.0.0.1 --port $PORT"
  echo "$CMD" > "$OUT/logs/$c.cmd"
  setsid bash -c "$CMD" > "$OUT/logs/$c.log" 2>&1 &
  SRV_PID=$!
  t0=$SECONDS ok=0
  while (( SECONDS - t0 < 1200 )); do
    kill -0 $SRV_PID 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && { ok=1; break; }
    sleep 3
  done
  if [ $ok = 0 ]; then
    log "$c: server failed to start"; tail -n 15 "$OUT/logs/$c.log" | tee -a "$OUT/sweep.log"; stop; continue
  fi
  log "$c: ready after $((SECONDS - t0)) s (max_model_len $max_len)"
  grep -m2 -E "Selected .*Kernel for|max_num_batched_tokens|max_num_seqs" "$OUT/logs/$c.log" | cut -c1-200 >> "$OUT/backend_check.txt"
  for s in $SCENARIOS; do
    IFS=: read -r p g conc <<<"$s"
    taskset -c "$CLIENT_CPUS" "$PY" "$ROOT/scripts/smoke_client.py" --url http://127.0.0.1:$PORT --engine vllm \
      --label "$c" --scenario "p${p}_g${g}_c${conc}" --prompt-len "$p" --gen-len "$g" --concurrency "$conc" \
      --runs "$RUNS" --warmup 1 --out "$OUT/raw.jsonl" >> "$OUT/sweep.log" 2>&1 &
    CLIENT_PID=$!
    wait $CLIENT_PID || log "$c p${p}_g${g}_c${conc}: client failed"
    CLIENT_PID=
    kill -0 $SRV_PID 2>/dev/null || { log "$c: server died"; break; }
  done
  stop
done
log "done: $OUT"
