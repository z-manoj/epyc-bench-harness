#!/usr/bin/env bash
# Reproduce AMD's ZenDNN 5.2 vLLM blog methodology (footnote ZD-059) on this host:
#   vllm bench throughput, random 128 in / 128 out, 1024 prompts, max-num-seqs 128, BF16,
#   single instance on all socket cores vs 2 instances on interleaved (even / odd) cores, stock vs zentorch.
# Usage: scripts/amd_repro_throughput.sh [cpu-single cpu-multi zentorch-single zentorch-multi ...]
# Env:   CORES_LO/CORES_HI (0/95)  MEM_NODE (0)  REPS (1)  OUT
set -uo pipefail
unset LD_PRELOAD CONDA_PREFIX VLLM_CPU_OMP_THREADS_BIND VLLM_CPU_KVCACHE_SPACE TORCHINDUCTOR_FREEZING VLLM_USE_AOT_COMPILE

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CORES_LO=${CORES_LO:-0} CORES_HI=${CORES_HI:-95} MEM_NODE=${MEM_NODE:-0} REPS=${REPS:-1}
OUT=${OUT:-$ROOT/results/amd_repro_$(date +%Y%m%d_%H%M)}
VZ=${VLLM_BASE:-$HOME/vllm-zen}
MODEL=${MODEL:-$HOME/mkumar/models/Llama-3.1-8B-Instruct}
CASES=(cpu-single zentorch-single cpu-multi zentorch-multi)
[ $# -gt 0 ] && CASES=("$@")
mkdir -p "$OUT/logs"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$OUT/repro.log"; }

ALL=$(seq -s, $CORES_LO $CORES_HI)
EVEN=$(seq -s, $CORES_LO 2 $CORES_HI)
ODD=$(seq -s, $((CORES_LO + 1)) 2 $CORES_HI)

run_instance() {  # run_instance <env> <cpus> <tag>
  local env=$1 cpus=$2 tag=$3
  bash -c "source $VZ/env-$env.sh && export VLLM_CPU_OMP_THREADS_BIND=$cpus VLLM_CPU_KVCACHE_SPACE=16 && cd /tmp && \
    exec numactl --physcpubind=$cpus --membind=$MEM_NODE vllm bench throughput --model $MODEL --dtype bfloat16 \
    --dataset-name random --random-input-len 128 --random-output-len 128 --num-prompts 1024 --max-num-seqs 128 \
    --max-model-len 1024 --output-json $OUT/$tag.json" > "$OUT/logs/$tag.log" 2>&1
}

{
  echo "date: $(date -Is)  host: $(hostname)  cpu: $(lscpu | sed -nE 's/Model name: *//p')"
  echo "single: cores $CORES_LO-$CORES_HI; multi: 2 instances on even / odd cores; membind $MEM_NODE"
  echo "vllm-zentorch: $($VZ/env/bin/python -m pip list 2>/dev/null | grep -E '^(vllm|torch|zentorch) ' | tr -s ' ' | paste -sd, -)"
  echo "vllm-cpu: $($VZ/env-cpu/bin/python -m pip list 2>/dev/null | grep -E '^(vllm|torch) ' | tr -s ' ' | paste -sd, -)"
} > "$OUT/meta.txt"

for rep in $(seq 1 $REPS); do
  for c in "${CASES[@]}"; do
    env=${c%%-*} mode=${c#*-}
    log "=== $c rep $rep"
    if [ "$mode" = single ]; then
      run_instance $env "$ALL" "${c}_r${rep}" || log "$c failed (see logs/${c}_r${rep}.log)"
    else
      run_instance $env "$EVEN" "${c}_r${rep}_i0" & p0=$!
      run_instance $env "$ODD" "${c}_r${rep}_i1" & p1=$!
      wait $p0 || log "$c instance 0 failed"; wait $p1 || log "$c instance 1 failed"
    fi
    grep -h "^Throughput:" "$OUT"/logs/${c}_r${rep}*.log | tee -a "$OUT/repro.log"
  done
done
log "done: $OUT"
