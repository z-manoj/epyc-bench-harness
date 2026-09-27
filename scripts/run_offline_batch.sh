#!/usr/bin/env bash
# Offline LLM.generate batch (N prompts x PROMPT_LEN tokens, 128 out), vLLM with and without zentorch, same pinning.
# Usage: scripts/run_offline_batch.sh [zentorch cpu]
# Env:   CPUS (100-191)  MEM_NODE (1)  N (20)  PROMPT_LEN (1024)  RUNS (3)  MODEL  OUT
#        FREEZING (unset = env default 1; applies to both envs)  CHUNKED (1; 0 = --no-chunked-prefill)
set -uo pipefail
unset LD_PRELOAD CONDA_PREFIX VLLM_CPU_OMP_THREADS_BIND VLLM_CPU_KVCACHE_SPACE VLLM_USE_AOT_COMPILE
FREEZING=${FREEZING:-}
unset TORCHINDUCTOR_FREEZING

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CPUS=${CPUS:-100-191} MEM_NODE=${MEM_NODE:-1} N=${N:-20} PROMPT_LEN=${PROMPT_LEN:-1024} RUNS=${RUNS:-3} CHUNKED=${CHUNKED:-1}
MODEL=${MODEL:-$HOME/mkumar/models/Llama-3.1-8B-Instruct}
OUT=$(realpath -m "${OUT:-$ROOT/results/offline_batch_$(date +%Y%m%d_%H%M)}")
VZ=${VLLM_BASE:-$HOME/vllm-zen}
ENVS=(zentorch cpu)
[ $# -gt 0 ] && ENVS=("$@")
mkdir -p "$OUT/logs"
echo "date: $(date -Is)  cpus: $CPUS  mem node: $MEM_NODE  batch: $N x $PROMPT_LEN in / 128 out  freezing: ${FREEZING:-env default (1)}  chunked prefill: $CHUNKED" >> "$OUT/meta.txt"

for e in "${ENVS[@]}"; do
  tag="vllm-$e${FREEZING:+-freeze$FREEZING}"
  chunk_arg=""
  [ "$CHUNKED" = 0 ] && { tag="$tag-nochunk"; chunk_arg="--no-chunked-prefill"; }
  echo "[$(date +%T)] === $tag" | tee -a "$OUT/offline.log"
  bash -c "${FREEZING:+export TORCHINDUCTOR_FREEZING=$FREEZING; }source $VZ/env-$e.sh && \
    export VLLM_CPU_OMP_THREADS_BIND=$CPUS VLLM_CPU_KVCACHE_SPACE=16 && cd /tmp && \
    exec numactl --physcpubind=$CPUS --membind=$MEM_NODE python $ROOT/scripts/offline_batch.py --model $MODEL \
    --prompts $ROOT/prompts/prompts_${PROMPT_LEN}.txt --label $tag --n $N --prompt-len $PROMPT_LEN --runs $RUNS \
    --out $OUT/raw.jsonl $chunk_arg" > "$OUT/logs/$tag.log" 2>&1
  rc=$?
  grep -E "^\[$tag\]" "$OUT/logs/$tag.log" | tee -a "$OUT/offline.log"
  [ $rc = 0 ] || { echo "$tag failed rc=$rc:" | tee -a "$OUT/offline.log"; grep -m3 -E "Segfault|dnnl_helper|Error" "$OUT/logs/$tag.log" | cut -c1-160 | tee -a "$OUT/offline.log"; }
done
echo "[$(date +%T)] done: $OUT" | tee -a "$OUT/offline.log"
