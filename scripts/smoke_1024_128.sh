#!/usr/bin/env bash
# Smoke test: prompt 1024 / generate 128, single stream, 1 warmup + RUNS measured requests per config.
# llama.cpp (ZenDNN / plain CPU builds) and vLLM (zentorch / stock CPU envs), one config at a time,
# pinned with numactl to SERVER_CPUS (one thread per physical core) and memory bound to MEM_NODE.
#
# Usage: scripts/smoke_1024_128.sh [config ...]   (default: all; see CONFIGS below)
# Env:   SERVER_CPUS (0-95 = socket 0 cores)  SERVER_SMT (192-287)  MEM_NODE (0)  CLIENT_CPUS (96-97)  RUNS (3)  OUT
# Single CCD: SERVER_CPUS=8-15 SERVER_SMT=200-207 CLIENT_CPUS=16-17
set -uo pipefail
# Each server gets its runtime env explicitly (vLLM via its env file, llama.cpp via LLAMA_PRELOAD), not from the caller.
unset LD_PRELOAD CONDA_PREFIX VLLM_CPU_OMP_THREADS_BIND VLLM_CPU_KVCACHE_SPACE TORCHINDUCTOR_FREEZING VLLM_USE_AOT_COMPILE

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SERVER_CPUS=${SERVER_CPUS:-0-95}
SERVER_SMT=${SERVER_SMT:-192-287}
MEM_NODE=${MEM_NODE:-0}
CLIENT_CPUS=${CLIENT_CPUS:-96-97}
RUNS=${RUNS:-3}
PROMPT_LEN=1024 GEN_LEN=128
OUT=${OUT:-$ROOT/results/smoke_1024_128_$(date +%Y%m%d_%H%M)}
LLAMA=${LLAMA_DIR:-$HOME/llama.cpp}
VZ=${VLLM_BASE:-$HOME/vllm-zen}
MODELS=${MODELS_DIR:-$HOME/mkumar/models}
PY=$VZ/tools/bin/python
# Same OpenMP runtime + allocator as the vLLM envs; LLAMA_PRELOAD="" runs llama.cpp on its own libgomp/glibc malloc
LLAMA_PRELOAD=${LLAMA_PRELOAD-$VZ/env/lib/libiomp5.so:$VZ/env/lib/libtcmalloc_minimal.so.4}
NTHREADS=$(( $(sed -E 's/([0-9]+)-([0-9]+)/\2-\1+1/' <<<"$SERVER_CPUS") ))
PORT=8300

CONFIGS=(llamacpp-zendnn-bf16 llamacpp-cpu-bf16 vllm-zentorch-bf16 vllm-cpu-bf16 llamacpp-zendnn-q8 llamacpp-cpu-q8
         vllm-zentorch-w8a8 vllm-cpu-w8a8)
[ $# -gt 0 ] && CONFIGS=("$@")

mkdir -p "$OUT/logs"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$OUT/smoke.log"; }

launch() {  # launch <config> ; sets ENGINE, starts server in its own session
  local c=$1 bin gguf env
  case $c in
    llamacpp-*)
      ENGINE=llamacpp
      [[ $c == *zendnn* ]] && bin=$LLAMA/build/bin/llama-server || bin=$LLAMA/build-nozendnn/bin/llama-server
      [[ $c == *bf16 ]] && gguf=$MODELS/gguf/Llama-3.1-8B-Instruct-BF16.gguf || gguf=$MODELS/gguf/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
      CMD="LD_PRELOAD=$LLAMA_PRELOAD numactl --physcpubind=$SERVER_CPUS --membind=$MEM_NODE $bin -m $gguf -a llama -t $NTHREADS -tb $NTHREADS -c 4096 -np 1 --host 127.0.0.1 --port $PORT"
      ;;
    vllm-*)
      ENGINE=vllm
      [[ $c == *zentorch* ]] && env=$VZ/env-zentorch.sh || env=$VZ/env-cpu.sh
      # w8a8: compressed-tensors INT8 weights + dynamic per-token INT8 activations; bf16 is the activation/KV dtype
      [[ $c == *w8a8 ]] && hfm=$MODELS/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 || hfm=$MODELS/Llama-3.1-8B-Instruct
      CMD="source $env && export VLLM_CPU_OMP_THREADS_BIND=$SERVER_CPUS VLLM_CPU_KVCACHE_SPACE=16 && cd /tmp && exec numactl --physcpubind=$SERVER_CPUS,$SERVER_SMT --membind=$MEM_NODE vllm serve $hfm --served-model-name llama --dtype bfloat16 --max-model-len 4096 --no-enable-prefix-caching --host 127.0.0.1 --port $PORT"
      ;;
    *) log "unknown config $c"; return 1 ;;
  esac
  echo "$CMD" > "$OUT/logs/$c.cmd"
  setsid bash -c "$CMD" > "$OUT/logs/$c.log" 2>&1 &
  SRV_PID=$!
}

wait_ready() {  # wait_ready <timeout_s>
  local t0=$SECONDS
  while (( SECONDS - t0 < $1 )); do
    kill -0 $SRV_PID 2>/dev/null || return 1
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0
    sleep 3
  done
  return 1
}

stop() {
  kill -TERM -- -$SRV_PID 2>/dev/null
  for _ in $(seq 30); do kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$SRV_PID 2>/dev/null
  sleep 3
}
trap '[ -n "${CLIENT_PID:-}" ] && kill $CLIENT_PID 2>/dev/null; stop; exit 130' INT TERM

{
  echo "date: $(date -Is)"; echo "host: $(hostname)  cpu: $(lscpu | sed -nE 's/Model name: *//p')"
  echo "server cpus: $SERVER_CPUS (+SMT $SERVER_SMT for vLLM frontend)  mem node: $MEM_NODE  client cpus: $CLIENT_CPUS"
  echo "llama.cpp: $(git -C "$LLAMA" log --oneline -1)  LD_PRELOAD=${LLAMA_PRELOAD:-none}"
  echo "vllm-zentorch: $($VZ/env/bin/python -m pip list 2>/dev/null | grep -E '^(vllm|torch|zentorch) ' | tr -s ' ' | paste -sd, -)"
  echo "vllm-cpu: $($VZ/env-cpu/bin/python -m pip list 2>/dev/null | grep -E '^(vllm|torch) ' | tr -s ' ' | paste -sd, -)"
} > "$OUT/meta.txt"

for c in "${CONFIGS[@]}"; do
  log "=== $c"
  launch "$c" || continue
  t0=$SECONDS
  if ! wait_ready 1200; then
    log "$c: server failed to become ready; tail of log:"; tail -n 20 "$OUT/logs/$c.log" | tee -a "$OUT/smoke.log"
    stop; continue
  fi
  log "$c: ready after $((SECONDS - t0)) s"
  echo "$c $((SECONDS - t0))" >> "$OUT/startup_s.txt"
  taskset -c "$CLIENT_CPUS" "$PY" "$ROOT/scripts/smoke_client.py" --url http://127.0.0.1:$PORT --engine $ENGINE \
    --label "$c" --runs "$RUNS" --warmup 1 --prompt-len $PROMPT_LEN --gen-len $GEN_LEN --out "$OUT/raw.jsonl" \
    >> "$OUT/smoke.log" 2>&1 &
  CLIENT_PID=$!
  wait $CLIENT_PID
  CLIENT_PID=
  grep -m3 -iE "zentorch is activated|using .*Platform|ZenDNN" "$OUT/logs/$c.log" | cut -c1-160 >> "$OUT/backend_check.txt"
  stop
done
log "done: $OUT"
