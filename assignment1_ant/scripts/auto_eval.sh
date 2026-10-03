#!/usr/bin/env bash
# Evaluate the next finished queue run that has not been evaluated yet, then exit (so the caller is notified per run).
# Waits while no new run has finished. For each run: model_1000 (budget-matched to the 1000-iter baseline) and the
# final model, on the 12 evaluation terrains, eval seeds 24 25, 256 envs -> results/eval_round2.csv.
# Axis-2 runs (v12/v13/v14) are evaluated in their deploy task with each terrain swapped in.
set -u
cd "$(dirname "$0")/.."
Q=logs/queue
touch "$Q/evaluated.txt"
source /home/cai/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
while true; do
  run=""
  for r in $(grep " END " "$Q/done.txt" | grep "exit=0" | awk '{print $5}'); do
    grep -qx "$r" "$Q/evaluated.txt" || { run=$r; break; }
  done
  if [ -n "$run" ]; then break; fi
  # nothing new: stop if training is over (both workers exited), else wait
  [ "$(grep -c 'worker exits' "$Q/done.txt")" -ge 2 ] && { echo "[AUTO] all runs evaluated"; exit 0; }
  sleep 60
done
dir=$(ls -d logs/rsl_rl/ant_rough/*_"$run" logs/rsl_rl/ant/*_"$run" 2>/dev/null | tail -1)
final=$(ls "$dir"/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
deploy=()
case "$run" in
  v12_*|v12r_*) deploy=(--deploy_task Isaac-Ant-Deploy-V12-v0) ;;
  v13_*) deploy=(--deploy_task Isaac-Ant-Deploy-V13-v0) ;;
  v14_*) deploy=(--deploy_task Isaac-Ant-Deploy-V14-v0) ;;
  v1214rmu02_*) deploy=(--deploy_task Isaac-Ant-Deploy-V1214-Mu02-v0) ;;
  v1214nohist_*) deploy=(--deploy_task Isaac-Ant-Deploy-V1214NoHist-v0) ;;
  v1314_*) deploy=(--deploy_task Isaac-Ant-Deploy-V1314-v0) ;;
  v1214mu03_*) deploy=(--deploy_task Isaac-Ant-Deploy-V1214-Mu03-v0) ;;
  v1214rmu03_*) deploy=(--deploy_task Isaac-Ant-Deploy-V1214-Mu03-v0) ;;
  v1214rmu06_*) deploy=(--deploy_task Isaac-Ant-Deploy-V1214-Mu06-v0) ;;
  v1214_*|v1214r_*) deploy=(--deploy_task Isaac-Ant-Deploy-V1214-v0) ;;
esac
echo "[AUTO] evaluating $run ($dir, final $final) ${deploy[*]}"
python scripts/eval_suite.py --out results/eval_round2.csv --seeds 24 25 --num_envs 256 --log_dir logs/eval_round2/"$run" \
  "${deploy[@]}" --ckpt "${run}@1000=$dir/model_1000.pt" --ckpt "${run}@${final}=$dir/model_${final}.pt" 2>&1 | tail -16
echo "$run" >> "$Q/evaluated.txt"
