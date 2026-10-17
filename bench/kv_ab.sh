#!/usr/bin/env bash
# Runs ON the VM after gpu_bench.sh. Interleaves f16 and q8_0 KV so T4 thermal throttling
# hits both equally (gpu_bench.sh runs each config back to back, and the first run of the
# day gets a cool GPU), and logs SM clock + temperature next to every result.
#   setsid nohup bash /content/kv_ab.sh 9b,14b > /content/bench/ab.log 2>&1 < /dev/null &
set -u
B=/content/llama/llama-b11696
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/content/llama/cudart-llama-b11696-bin-ubuntu-cuda-12.8-x64:$B
OUT=/content/bench
pkill -f '^/content/llama/.*/llama-server'; sleep 3
declare -A M=([9b]=/content/models/Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf
              [14b]=/content/models/Huihui-Qwen3-14B-abliterated-v2.Q4_K_M.gguf)
declare -A D=([9b]=0,32768,98304 [14b]=0,16384)
for model in ${1//,/ }; do
  for round in 1; do
    for kv in f16 q8_0 q8_0 f16; do  # ABBA order cancels a steady drift
      echo "== $model round $round kv=$kv $(date +%T) $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)"
      $B/llama-bench -m "${M[$model]}" -ngl 999 -t 2 -fa on -ctk $kv -ctv $kv -p 512 -n 128 \
        -d "${D[$model]}" -r 1 -o jsonl >> $OUT/ab-$model.jsonl 2>> $OUT/ab-$model.err
    done
  done
done
echo "== done $(date +%T)"
