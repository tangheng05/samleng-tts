#!/bin/zsh
# Score every 2000th checkpoint of the newest run under OUT_DIR while it trains, with phone updates.
#   ops/watch_run.sh outputs/train/distill_v1_B     (started by the training job, in the background)
export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
export LC_ALL=en_US.UTF-8
cd ~/code/samleng-tts
OUT=${1:?usage: watch_run.sh OUT_DIR}
name=${OUT:t}

score_new() {
  local run=$(ls -td $OUT/*/ 2>/dev/null | head -1)
  run=${run%/}
  [[ -n $run ]] || return
  for c in $run/checkpoint_*.pth(N); do
    s=${${c:t:r}#checkpoint_}
    (( s % 2000 == 0 )) || continue
    [[ -f outputs/train/eval/$name/$s/results.csv ]] && continue
    echo "== step $s $(date)"
    nice -n 10 uv run python scripts/eval_checkpoint.py $run --step $s --notify
  done
}

sleep 300  # let training create its run folder
while pgrep -f "train.py .*${name}" >/dev/null; do
  score_new
  sleep 120
done
score_new  # the last checkpoint can land just before training exits
echo "training stopped $(date)"
