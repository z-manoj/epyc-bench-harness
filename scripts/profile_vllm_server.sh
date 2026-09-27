#!/usr/bin/env bash
# Torch-profile one 1024-token / 16-token request on a running-as-benchmarked vLLM server (zentorch vs stock env).
# Usage: scripts/profile_vllm_server.sh <zentorch|cpu> <model_dir> <out_dir>
set -uo pipefail
ENVN=$1 MODEL=$2 OUTD=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=8310 CPUS=${SERVER_CPUS:-0-95} SMT=${SERVER_SMT:-192-287}
mkdir -p "$OUTD/trace_$ENVN"
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUTD/trace_$ENVN\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"active_iterations\":17}"
setsid bash -c "source $HOME/vllm-zen/env-$ENVN.sh && export VLLM_CPU_OMP_THREADS_BIND=$CPUS VLLM_CPU_KVCACHE_SPACE=16 && cd /tmp && \
  exec numactl --physcpubind=$CPUS,$SMT --membind=0 vllm serve $MODEL --served-model-name llama --dtype bfloat16 \
  --max-model-len 4096 --no-enable-prefix-caching --host 127.0.0.1 --port $PORT --profiler-config '$PROF'" \
  > "$OUTD/server_$ENVN.log" 2>&1 &
PID=$!
for _ in $(seq 400); do
  kill -0 $PID 2>/dev/null || { echo "server died; see $OUTD/server_$ENVN.log"; exit 1; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && break
  sleep 3
done
C="$HOME/vllm-zen/tools/bin/python $ROOT/scripts/smoke_client.py --url http://127.0.0.1:$PORT --engine vllm --gen-len 16 --out /dev/null"
$C --label warm --runs 1 --warmup 1
curl -s -X POST http://127.0.0.1:$PORT/start_profile
$C --label prof --runs 1 --warmup 0
curl -s -X POST http://127.0.0.1:$PORT/stop_profile
sleep 20
kill -TERM -- -$PID; sleep 10; kill -KILL -- -$PID 2>/dev/null
ls -la "$OUTD/trace_$ENVN"
