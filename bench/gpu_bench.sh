#!/usr/bin/env bash
# Runs ON the Colab VM after colab-llm.py has downloaded the models. Detach it, then poll the log:
#   setsid nohup bash /content/gpu_bench.sh > /content/bench/run.log 2>&1 < /dev/null &
# Stops llama-server first: the benchmark needs the whole T4.
#
# 1. speed: llama-bench pp512 / tg128 at increasing context depth, per KV config
# 2. fit:   the largest --ctx-size llama-server loads, and the VRAM it then uses
set -u
B=/content/llama/llama-b11696
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/content/llama/cudart-llama-b11696-bin-ubuntu-cuda-12.8-x64:$B
OUT=/content/bench
mkdir -p $OUT
pkill -f '^/content/llama/.*/llama-server'; sleep 3

M9=/content/models/Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf
M14=/content/models/Huihui-Qwen3-14B-abliterated-v2.Q4_K_M.gguf

bench() {  # name model depths reps fa ctk ctv
  echo "== speed $1 fa=$5 kv=$6/$7 depths=$3 $(date +%T)"
  $B/llama-bench -m "$2" -ngl 999 -t 2 -fa "$5" -ctk "$6" -ctv "$7" -p 512 -n 128 -d "$3" -r "$4" -o jsonl \
    >> $OUT/speed-$1.jsonl 2>> $OUT/speed-$1.err || echo "   failed (see speed-$1.err)"
}

fit() {  # name model kv ctx...
  local name=$1 model=$2 kv=$3; shift 3
  for ctx in "$@"; do
    $B/llama-server -m "$model" --host 127.0.0.1 --port 8100 -c "$ctx" -ngl 999 -t 2 --jinja \
      -fa on -ctk "$kv" -ctv "$kv" > $OUT/fit.log 2>&1 &
    local pid=$! ok=0
    for _ in $(seq 240); do
      sleep 1
      curl -sf localhost:8100/health > /dev/null && { ok=1; break; }
      kill -0 $pid 2> /dev/null || break
    done
    local mem; mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
    echo "{\"model\":\"$name\",\"kv\":\"$kv\",\"ctx\":$ctx,\"ok\":$ok,\"mem\":\"$mem\"}" | tee -a $OUT/fit.jsonl
    kill $pid 2> /dev/null; wait $pid 2> /dev/null; sleep 2
    [ $ok = 1 ] || break  # larger contexts will not fit either
  done
}

# Most decision-relevant runs first, so a reclaimed VM still leaves usable data.
fit 9b  $M9  q8_0 65536 131072 204800 262144
fit 9b  $M9  f16  65536 131072 204800 262144
bench 9b $M9 0,16384,65536 2 on q8_0 q8_0
bench 9b $M9 0,16384,65536 2 on f16 f16
bench 9b $M9 0,16384 2 off f16 f16
bench 9b $M9 131072,196608 1 on q8_0 q8_0
fit 14b $M14 q8_0 16384 32768 40960 65536
fit 14b $M14 f16  16384 32768 40960
bench 14b $M14 0,8192,24576 2 on q8_0 q8_0
bench 14b $M14 0,8192,24576 2 on f16 f16
bench 14b $M14 0,8192 2 off f16 f16
echo "== done $(date +%T)"
