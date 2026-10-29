#!/usr/bin/env bash
# Runs the experiment for one model. Safe to stop (Ctrl+C) and restart: finished runs are skipped.
#
#   bash run_all.sh                                    # main model: qwen3:8b (reasoning), full design
#   MODEL=llama3.1:8b bash run_all.sh                  # second model: reduced design
#
# Main model  (qwen3:8b):     7 defences x (15 benign + 4 x 90) = 2,625 runs
#                             + framing check A_provider, B1_user without defence = 180 runs  -> 2,805
# Second model (llama3.1:8b): 3 defences x 375 = 1,125 runs (no framing check)
# (Channel B4 exists in the code but is not part of the experiment.)
#
# Sampling: Qwen3 in thinking mode must not use greedy decoding (Qwen3 model card), so it runs at
# temperature 0.6 and top-p 0.95; top-k (20) is Ollama's default for the model. Llama runs at
# temperature 0. The seed is fixed at 0 for all runs.
set -u
MODEL="${MODEL:-qwen3:8b}"
BASE_URL="${BASE_URL:-http://localhost:11434/v1}"
OUT="${OUT:-results}"

if [[ "$MODEL" == qwen3* ]]; then
  SAMPLING="--temperature ${TEMPERATURE:-0.6} --top-p ${TOP_P:-0.95}"
  DEFAULT_DEFENCES="none repeat_prompt pi_detector tool_filter melon melon_firststep desc_scan"
  DEFAULT_FRAMING="A_provider B1_user"
else
  SAMPLING="--temperature ${TEMPERATURE:-0}"
  DEFAULT_DEFENCES="none melon melon_firststep"
  DEFAULT_FRAMING=""
fi
DEFENCES="${DEFENCES:-$DEFAULT_DEFENCES}"
CHANNELS="${CHANNELS:-benign A B1 B2 B3}"
FRAMING_CHANNELS="${FRAMING_CHANNELS:-$DEFAULT_FRAMING}"
mkdir -p "$OUT/logs"

run() {  # $1 = channel, $2 = defence
  echo "=== $MODEL | $1 | $2 ==="
  python -m mcpsec.run --model "$MODEL" --base-url "$BASE_URL" $SAMPLING --channel "$1" --defence "$2" --out "$OUT" \
    --save-messages 2>&1 | tee -a "$OUT/logs/${MODEL//:/-}__${1}__${2}.log"
}

for defence in $DEFENCES; do
  for channel in $CHANNELS; do run "$channel" "$defence"; done
done
for channel in $FRAMING_CHANNELS; do run "$channel" none; done   # framing check: no defence only

python -m mcpsec.analyze --results "$OUT" --out analysis
