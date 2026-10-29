#!/usr/bin/env bash
# 사전등록 §7 변경 기록 "처치 시작 뒤" 1 의 보정 경로 (A6000, 메인 큐 run_queue_1029.sh 와 병행).
# 층화 칸 부족으로 데이터 게이트를 못 넘은 팔(A-T, 필요하면 A-R)에 같은 명령으로 datagen seed 6, 7 샤드(각 130)를
# 하나씩 더 만들어(최대 2개) 같은 seed 0 으로 다시 층화한다. 학습 데이터 500 · 게이트 기준은 그대로다.
# 게이트를 넘으면 메인 큐가 건너뛴 그 팔의 학습·평가(seed 42) → 평가 seed 43 → (A-T 만) 체크포인트 민감도를 돌린다.
#   tmux new -s rs_a2_fix 'bash ~/robotics-simulation-team12/assignment2_toy2box/tools/run_queue_1029_followup.sh'
set -uo pipefail
A2=$HOME/robotics-simulation-team12/assignment2_toy2box
cd "$HOME/IsaacLab_RS"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate lerobot-arena
export PYTHONUNBUFFERED=1
Q=$HOME/IsaacLab_RS/logs/rs_a2_runs/queue_1029
REF=./datasets/team12_kgh/kgh_mimic_500.hdf5
HUMAN=./datasets/team12_kgh/toy2box_kgh_annotated.hdf5
P=./datasets/team12_kgh/plan_A; L=$HOME/IsaacLab_RS/logs/rs_a2_runs/plan_A
RUNS=$A2/results/runs
MIMIC_TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0

log() { echo "[$(date '+%F %T')] [보정] $*" | tee -a "$Q/status.log"; }

gen_shard() {  # tag flag name seed (끝까지 간 샤드는 건너뜀)
  local tag=$1 flag=$2 name=$3 s=$4
  local out=$P/${name}_shards/${name}_s${s}_130.hdf5 lg=$L/${tag}_s${s}.log
  if [ -f "$out" ] && grep -q "^RC=0" "$lg" 2>/dev/null; then return 0; fi
  if pgrep -f "[o]utput_file $out" >/dev/null; then   # 다른 창에서 이미 돌고 있으면 기다린다
    while pgrep -f "[o]utput_file $out" >/dev/null; do sleep 30; done
    grep -q "^RC=0" "$lg" && return 0
  fi
  rm -f "$out" "${out%.hdf5}_failed.hdf5" "$L/${tag}_s${s}_src.jsonl"
  mkdir -p "/tmp/rs_a2_tmp_${tag}_s${s}" "$P/${name}_shards"
  TMPDIR=/tmp/rs_a2_tmp_${tag}_s${s} timeout 10800 ./isaaclab.sh -p $A2/tools/generate_yawhold.py $flag --datagen_seed $s \
    --src_log $L/${tag}_s${s}_src.jsonl --task $MIMIC_TASK --headless --num_envs 50 \
    --generation_num_trials 130 --input_file $HUMAN --output_file $out > "$lg" 2>&1
  echo "RC=$?" >> "$lg"
}

# 층화 + 게이트. 칸이 모자라면(rc 2) 샤드를 하나 더 (seed 6, 7). 성공이면 0
fix_arm() {  # tag flag name garm
  local tag=$1 flag=$2 name=$3 garm=$4 last=5 rc
  local strat=$P/${name}_strat500.hdf5
  while :; do
    local seeds; seeds=$(seq -s, 1 $last)
    if [ ! -f "$strat" ]; then
      python -I $A2/tools/plan_A_stratify.py --ref $REF --n 500 --seed 0 --out $strat \
        $(for s in $(seq 1 $last); do echo $P/${name}_shards/${name}_s${s}_130.hdf5; done) \
        > $A2/results/plan_A/stratify_${tag}500.txt 2>&1
      rc=$?
    else rc=0; fi
    if [ $rc -eq 0 ]; then log "$tag 층화 통과 (샤드 seed 1–$last)"; break; fi
    if [ $rc -ne 2 ] || [ $last -ge 7 ]; then log "$tag 층화 실패 rc=$rc (샤드 seed 1–$last) — 멈춤"; return 2; fi
    last=$((last+1)); log "$tag 층화 칸 부족 → datagen seed $last 샤드 추가 생성"
    gen_shard "$tag" "$flag" "$name" "$last"
    grep -q "^RC=0" "$L/${tag}_s${last}.log" || { log "$tag seed $last 생성 실패"; return 1; }
    log "$tag s$last: $(grep -a 'successful demos generated' "$L/${tag}_s${last}.log" | tail -1 | tr -d '\033' | sed 's/\[K//g')"
  done
  local extra=""
  if [ "$garm" = AR ]; then
    local aty; aty=$(grep -oP 'yaw_go = \K[0-9.]+' $A2/results/plan_A/gate_AT500.txt 2>/dev/null)
    [ -n "$aty" ] || { log "AR 게이트: A-T 의 yaw_go 가 없다"; return 3; }
    extra="--at_yaw $aty"
  fi
  local n; n=$(ls $P/${name}_shards/${name}_s?_130.hdf5 | wc -l)
  python -I $A2/tools/plan_A_smoke_check.py check $garm $strat --n_logs $n \
    --log $(for s in $(seq 1 $n); do echo $L/${tag}_s${s}.log; done) $extra > $A2/results/plan_A/gate_${tag}500.txt 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then log "$tag 데이터 게이트 통과"; else log "$tag 데이터 게이트 미통과 — results/plan_A/gate_${tag}500.txt"; fi
  return $rc
}

train_eval() {  # tag data ep eseed outdir
  local tag=$1 data=$2 ep=$3 es=$4 od=$5 s pids=() rcs=() k ok=0
  log "$tag e$ep 평가 seed $es 시작"
  for s in 101 111; do
    ( for try in 1 2; do
        OUTDIR=$od bash $A2/tools/run_baseline.sh $tag $data $s $ep $es >> "$Q/${tag}_s${s}_e${ep}_seed${es}.out" 2>&1
        r=$?; [ $r -eq 5 ] && [ $try -eq 1 ] && { echo "retry after rc=5" >> "$Q/${tag}_s${s}_e${ep}_seed${es}.out"; continue; }
        exit $r
      done ) & pids+=($!); sleep 20
  done
  k=0; for p in "${pids[@]}"; do wait "$p"; rcs[$k]=$?; k=$((k+1)); done
  for k in 0 1; do
    s=$([ $k -eq 0 ] && echo 101 || echo 111)
    if [ "${rcs[$k]}" -eq 0 ]; then
      log "$tag s$s e$ep seed$es 완료: $(grep -h 'Successful trials' $HOME/IsaacLab_RS/logs/rs_a2_runs/b3_eval_base_${tag}_s${s}_e${ep}_seed${es}.log | tail -1)"
    else log "$tag s$s e$ep seed$es 실패 rc=${rcs[$k]}"; ok=1; fi
  done
  return $ok
}

log "===== 보정 경로 시작 ====="
if fix_arm AT "--translate 1" translate AT; then
  train_eval translate500 $P/translate_strat500.hdf5 2000 42 $RUNS
  train_eval translate500 $P/translate_strat500.hdf5 2000 43 $RUNS
  touch "$Q/AT_MAIN_DONE"
else log "A-T 보정 실패 — A-T 는 게이트 미통과로 보고"; fi
# A-R: 메인 큐의 판단을 기다린다
while [ ! -f "$Q/AR_DATA_OK" ] && [ ! -f "$Q/AR_DATA_FAIL" ]; do sleep 60; done
if [ -f "$Q/AR_DATA_FAIL" ]; then
  log "A-R 메인 큐 게이트 실패 → 보정 경로"
  rm -f "$P/rotworld_strat500.hdf5"   # 층화 실패면 파일이 없다; 게이트만 실패했으면 같은 층화본을 다시 만든다(결정적)
  if fix_arm AR "--rotworld 1" rotworld AR; then
    train_eval rotworld500 $P/rotworld_strat500.hdf5 2000 42 $RUNS
    train_eval rotworld500 $P/rotworld_strat500.hdf5 2000 43 $RUNS
  else log "A-R 보정 실패 — A-R 은 게이트 미통과로 보고"; fi
fi
# A-T 체크포인트 민감도 (보고만)
if [ -f "$Q/AT_MAIN_DONE" ]; then
  train_eval translate500 $P/translate_strat500.hdf5 1800 42 $RUNS
  train_eval translate500 $P/translate_strat500.hdf5 1900 42 $RUNS
fi
touch "$Q/FOLLOWUP_DONE"; log "===== 보정 경로 끝 ====="
