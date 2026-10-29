#!/usr/bin/env bash
# 트랙 C: C3 · 위약을 Spark 두 대에서 (prereg §5.4, plan_C §7). 학습 seed 별 같은 Spark: s101 → gx10-3, s111 → gx10-2.
# 같은 학습 seed 는 c3spark 를 먼저 띄우고 'ckpt md5' 줄이 찍힌 뒤 placebo 를 띄운다(원격 체크포인트 복사가 겹치지 않게).
#   tmux new -s rs_a2_c3 'bash ~/robotics-simulation-team12/assignment2_toy2box/tools/plan_C/run_c3_all.sh'
set -uo pipefail
A2=$HOME/robotics-simulation-team12/assignment2_toy2box
O=$HOME/IsaacLab_RS/logs/rs_a2_runs/plan_C; mkdir -p "$O"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$O/c3_all.status.log"; }
pids=()
for s in 101 111; do
  bash $A2/tools/plan_C/run_plan_C.sh c3spark $s > "$O/c3spark_s${s}.out" 2>&1 & pids+=($!)
  log "c3spark s$s 시작 (pid $!)"
  for i in $(seq 1 120); do grep -q "ckpt md5" "$O/c3spark_s${s}.out" 2>/dev/null && break; sleep 5; done
  grep -q "ckpt md5" "$O/c3spark_s${s}.out" || log "c3spark s$s: 10분 안에 md5 줄이 없다 — 그래도 placebo 를 띄운다"
  bash $A2/tools/plan_C/run_plan_C.sh placebo $s > "$O/placebo_s${s}.out" 2>&1 & pids+=($!)
  log "placebo s$s 시작 (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; log "pid $p 끝 rc=$?"; done
log "C3 · 위약 끝"
