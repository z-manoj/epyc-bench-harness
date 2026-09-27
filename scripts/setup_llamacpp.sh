#!/usr/bin/env bash
# Build llama.cpp twice (ggml-zendnn ON -> build/, OFF -> build-nozendnn/) and download the GGUF models.
# ggml-zendnn cannot be disabled at runtime, hence two builds. Idempotent: re-running only rebuilds what changed.
#
# Usage: scripts/setup_llamacpp.sh [--no-models] [--bf16] [--serial]
# Env:   LLAMA_DIR (~/llama.cpp)  LLAMA_COMMIT (9adc7f4)  MODELS_DIR (~/mkumar/models)  JOBS (nproc/4)
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
LLAMA_COMMIT=${LLAMA_COMMIT:-9adc7f4}
MODELS_DIR=${MODELS_DIR:-$HOME/mkumar/models}
JOBS=${JOBS:-$(( $(nproc) / 4 > 1 ? $(nproc) / 4 : 1 ))}
TARGETS=(llama-server llama-bench llama-cli)

GET_MODELS=1 GET_BF16=0 PARALLEL=1
for a in "$@"; do
  case $a in
    --no-models) GET_MODELS=0 ;;
    --bf16) GET_BF16=1 ;;
    --serial) PARALLEL=0 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }
die() { log "ERROR: $*"; exit 1; }

for t in git cmake gcc g++; do command -v $t >/dev/null || die "$t not found"; done

log "llama.cpp: $LLAMA_DIR @ $LLAMA_COMMIT"
if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone -q https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" cat-file -e "$LLAMA_COMMIT^{commit}" 2>/dev/null || git -C "$LLAMA_DIR" fetch -q origin
git -C "$LLAMA_DIR" checkout -q "$LLAMA_COMMIT"
log "checked out $(git -C "$LLAMA_DIR" log --oneline -1)"

build() {  # build <dir> <ON|OFF>
  local dir=$1 zendnn=$2 logf="$LLAMA_DIR/setup_$1.log"
  log "building $dir (GGML_ZENDNN=$zendnn, -j$JOBS), log: $logf"
  {
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/$dir" -DCMAKE_BUILD_TYPE=Release -DGGML_ZENDNN="$zendnn" &&
    cmake --build "$LLAMA_DIR/$dir" --config Release -j "$JOBS" --target "${TARGETS[@]}"
  } > "$logf" 2>&1 || { tail -n 20 "$logf"; die "$dir build failed"; }
  log "$dir done"
}

if [ $PARALLEL = 1 ]; then
  build build ON & p1=$!
  build build-nozendnn OFF & p2=$!
  wait $p1 && wait $p2 || die "a build failed"
else
  build build ON
  build build-nozendnn OFF
fi

# The harness checks backends the same way; build/ must list ZenDNN, build-nozendnn/ must list nothing.
devs_on=$("$LLAMA_DIR/build/bin/llama-server" --list-devices 2>&1 | sed -n '/Available devices/,$p')
devs_off=$("$LLAMA_DIR/build-nozendnn/bin/llama-server" --list-devices 2>&1 | sed -n '/Available devices/,$p')
grep -q "ZenDNN" <<<"$devs_on" || die "build/ does not list the ZenDNN device:"$'\n'"$devs_on"
grep -q "(none)" <<<"$devs_off" || die "build-nozendnn/ lists devices:"$'\n'"$devs_off"
log "verified: build/ -> $(grep ZenDNN <<<"$devs_on" | sed 's/^ *//' | cut -c1-60)..., build-nozendnn/ -> no devices"

if [ $GET_MODELS = 1 ]; then
  command -v hf >/dev/null || die "hf CLI not found (pip install -U huggingface_hub)"
  [ -z "${HF_TOKEN:-}" ] && [ -s "$HOME/.cache/huggingface/token" ] && export HF_TOKEN=$(<"$HOME/.cache/huggingface/token")
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  mkdir -p "$MODELS_DIR/gguf"
  fetch() {  # fetch <repo> <file>
    if [ -s "$MODELS_DIR/gguf/$2" ] && [ ! -e "$MODELS_DIR/gguf/.cache/huggingface/download/$2.incomplete" ]; then
      log "have $2"; return
    fi
    log "downloading $1/$2"
    hf download "$1" "$2" --local-dir "$MODELS_DIR/gguf" >/dev/null
  }
  fetch bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
  [ $GET_BF16 = 1 ] && fetch unsloth/Llama-3.1-8B-Instruct-GGUF Llama-3.1-8B-Instruct-BF16.gguf
  ls -lh "$MODELS_DIR"/gguf/*.gguf
fi

cat <<EOF

llama.cpp ready:
  ZenDNN:  $LLAMA_DIR/build/bin/llama-server
  CPU:     $LLAMA_DIR/build-nozendnn/bin/llama-server
  models:  $MODELS_DIR/gguf/
EOF
