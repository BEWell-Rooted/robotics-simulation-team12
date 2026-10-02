#!/usr/bin/env bash
# Training queue worker: pops one line at a time from logs/queue/jobs.txt (under flock) and trains it; exits when the
# queue is empty. Start two workers for two parallel trainings. Lines: <run_name> <task> <seed> [extra train.py args]
# Append lines at any time; completed / failed runs go to logs/queue/done.txt.
#
#   bash scripts/queue_runner.sh A &   bash scripts/queue_runner.sh B &
set -u
cd "$(dirname "$0")/.."
Q=logs/queue
mkdir -p "$Q"
touch "$Q/jobs.txt" "$Q/done.txt"
WORKER=${1:-A}
source /home/cai/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
while true; do
  job=$(flock "$Q/lock" bash -c "head -n1 '$Q/jobs.txt'; sed -i '1d' '$Q/jobs.txt'")
  [ -z "$job" ] && break
  read -r run task seed extra <<< "$job"
  start=$(date +%s)
  echo "$(date '+%m-%d %H:%M') [$WORKER] START $run" >> "$Q/done.txt"
  ~/IsaacLab_RS/isaaclab.sh -p scripts/rsl_rl/train.py --task "$task" --headless --seed "$seed" --run_name "$run" \
    ${extra:-} < /dev/null > "$Q/$run.log" 2>&1
  code=$?
  last=$(grep -a "Learning iteration" "$Q/$run.log" | tail -1 | grep -o "[0-9]*/[0-9]*")
  echo "$(date '+%m-%d %H:%M') [$WORKER] END   $run exit=$code iter=$last $(( ($(date +%s) - start) / 60 ))min" >> "$Q/done.txt"
done
echo "$(date '+%m-%d %H:%M') [$WORKER] queue empty, worker exits" >> "$Q/done.txt"
