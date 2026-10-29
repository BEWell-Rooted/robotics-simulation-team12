#!/usr/bin/env bash
# 사전등록 docs/prereg_1029.md §7 의 A6000 직렬 큐. 팔마다 트랙 기획 문서의 명령을 그대로 쓰고,
# 다음 단계로 가기 전에 종료 코드와 데이터 게이트(§3)를 확인한다.
#
#   tmux new -s rs_a2_queue 'bash ~/robotics-simulation-team12/assignment2_toy2box/tools/run_queue_1029.sh'
#   진행 상황: ~/IsaacLab_RS/logs/rs_a2_runs/queue_1029/status.log
#
# 순서: A-T 생성·층화·게이트 → (A-R 생성은 A-T 학습과 병행, CPU 작업) → A-T 학습·평가 → B2 → A-R → B3'
#       → 평가 seed 43 (baseline · A-T · B2 · A-R · B3') → 체크포인트 민감도(보고만). 판정(paired_compare)은 큐 밖에서 한다.
# 팔마다 학습 seed 101 · 111 을 동시에(TMPDIR 따로) 돌린다. 이미 끝난 단계는 건너뛴다(다시 실행해도 안전).
# 게이트 미통과는 기록하고 그 팔을 학습하지 않고 다음 팔로 간다(§3). 평가 실패(종료 코드 5)는 한 번 다시 돌린다.
set -uo pipefail
A2=$HOME/robotics-simulation-team12/assignment2_toy2box
cd "$HOME/IsaacLab_RS"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate lerobot-arena
export PYTHONUNBUFFERED=1
Q=$HOME/IsaacLab_RS/logs/rs_a2_runs/queue_1029; mkdir -p "$Q"
REF=./datasets/team12_kgh/kgh_mimic_500.hdf5
HUMAN=./datasets/team12_kgh/toy2box_kgh_annotated.hdf5
P=./datasets/team12_kgh/plan_A; PB=./datasets/team12_kgh/plan_B
L=$HOME/IsaacLab_RS/logs/rs_a2_runs/plan_A; mkdir -p "$L"
RUNS=$A2/results/runs; BASE=$A2/results/baseline
MIMIC_TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/status.log"; }

# ── Mimic 샤드 하나 생성 (끝까지 갔으면 건너뜀) ──
gen_shard() {  # tag flag name seed
  local tag=$1 flag=$2 name=$3 s=$4
  local out=$P/${name}_shards/${name}_s${s}_130.hdf5 lg=$L/${tag}_s${s}.log
  if [ -f "$out" ] && grep -q "^RC=0" "$lg" 2>/dev/null; then return 0; fi
  rm -f "$out" "${out%.hdf5}_failed.hdf5" "$L/${tag}_s${s}_src.jsonl"
  mkdir -p "/tmp/rs_a2_tmp_${tag}_s${s}" "$P/${name}_shards"
  TMPDIR=/tmp/rs_a2_tmp_${tag}_s${s} timeout 10800 ./isaaclab.sh -p $A2/tools/generate_yawhold.py $flag --datagen_seed $s \
    --src_log $L/${tag}_s${s}_src.jsonl --task $MIMIC_TASK --headless --num_envs 50 \
    --generation_num_trials 130 --input_file $HUMAN --output_file $out > "$lg" 2>&1
  echo "RC=$?" >> "$lg"
}

# ── 팔 하나의 데이터: 샤드 5개 → 층화 500 → 게이트. 성공이면 0 ──
gen_arm() {  # tag flag name gatearm
  local tag=$1 flag=$2 name=$3 garm=$4 s pids=()
  log "$tag 생성 시작 ($flag, seed 1–5 × 130)"
  for s in 1 2 3 4 5; do gen_shard "$tag" "$flag" "$name" "$s" & pids+=($!); sleep 20; done
  for p in "${pids[@]}"; do wait "$p"; done
  for s in 1 2 3 4 5; do
    grep -q "^RC=0" "$L/${tag}_s${s}.log" || { log "$tag 샤드 s$s 생성 실패 — $L/${tag}_s${s}.log"; return 1; }
    log "$tag s$s: $(grep -a 'successful demos generated' "$L/${tag}_s${s}.log" | tail -1 | tr -d '\033' | sed 's/\[K//g')"
  done
  local strat=$P/${name}_strat500.hdf5
  if [ ! -f "$strat" ]; then
    python -I $A2/tools/plan_A_stratify.py --ref $REF --n 500 --seed 0 --out $strat \
      $P/${name}_shards/${name}_s{1,2,3,4,5}_130.hdf5 > $A2/results/plan_A/stratify_${tag}500.txt 2>&1
    local rc=$?; [ $rc -eq 0 ] || { log "$tag 층화 게이트 미통과(rc=$rc) — results/plan_A/stratify_${tag}500.txt"; return 2; }
  fi
  local extra=""
  if [ "$garm" = AR ]; then
    local aty; aty=$(grep -oP 'yaw_go = \K[0-9.]+' $A2/results/plan_A/gate_AT500.txt 2>/dev/null)
    [ -n "$aty" ] || { log "AR 게이트: A-T 의 yaw_go 를 찾을 수 없다"; return 3; }
    extra="--at_yaw $aty"
  fi
  python -I $A2/tools/plan_A_smoke_check.py check $garm $strat --log $L/${tag}_s{1,2,3,4,5}.log $extra \
    > $A2/results/plan_A/gate_${tag}500.txt 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "$tag 데이터 게이트 통과"; else log "$tag 데이터 게이트 미통과 — results/plan_A/gate_${tag}500.txt"; return 4; fi
}

# ── 학습 seed 101 · 111 을 동시에: 학습(체크포인트 없을 때만) → 평가 → 분류 ──
train_eval() {  # tag data ep eseed outdir
  local tag=$1 data=$2 ep=$3 es=$4 od=$5 s pids=() rcs=() k
  log "$tag e$ep 평가 seed $es 시작 (OUTDIR=${od#$A2/})"
  for s in 101 111; do
    ( for try in 1 2; do
        OUTDIR=$od bash $A2/tools/run_baseline.sh $tag $data $s $ep $es >> "$Q/${tag}_s${s}_e${ep}_seed${es}.out" 2>&1
        r=$?; [ $r -eq 5 ] && [ $try -eq 1 ] && { echo "retry after rc=5" >> "$Q/${tag}_s${s}_e${ep}_seed${es}.out"; continue; }
        exit $r
      done ) & pids+=($!); sleep 20
  done
  k=0; for p in "${pids[@]}"; do wait "$p"; rcs[$k]=$?; k=$((k+1)); done
  local ok=0
  for k in 0 1; do
    s=$([ $k -eq 0 ] && echo 101 || echo 111)
    if [ "${rcs[$k]}" -eq 0 ]; then
      log "$tag s$s e$ep seed$es 완료: $(grep -h 'Successful trials' $HOME/IsaacLab_RS/logs/rs_a2_runs/b3_eval_base_${tag}_s${s}_e${ep}_seed${es}.log | tail -1)"
    else log "$tag s$s e$ep seed$es 실패 rc=${rcs[$k]} — $Q/${tag}_s${s}_e${ep}_seed${es}.out"; ok=1; fi
  done
  return $ok
}

md5_ok() {  # file prefix
  local m; m=$(md5sum "$1" | cut -c1-8); [ "$m" = "$2" ] || { log "md5 불일치 $1: $m (기대 $2…)"; return 1; }
}

log "===== 큐 시작 (prereg_1029 §7) ====="
AT_OK=0; AR_OK=0
# 1. A-T 데이터
if gen_arm AT "--translate 1" translate AT; then AT_OK=1; fi
# 2. A-R 데이터는 A-T 학습과 병행 (A-R 게이트는 A-T 게이트 값을 쓰므로 A-T 게이트 뒤에 시작)
( if gen_arm AR "--rotworld 1" rotworld AR; then touch "$Q/AR_DATA_OK"; else touch "$Q/AR_DATA_FAIL"; fi ) &
AR_GEN_PID=$!
# 3. A-T 학습·평가
if [ $AT_OK -eq 1 ]; then train_eval translate500 $P/translate_strat500.hdf5 2000 42 $RUNS; else log "A-T 학습 건너뜀 (데이터 게이트 미통과)"; fi
# 4. B2
if md5_ok $PB/kgh_mimic_500_reach4.hdf5 32f3c488; then train_eval reach4 $PB/kgh_mimic_500_reach4.hdf5 2000 42 $RUNS; fi
# 5. A-R
wait $AR_GEN_PID
if [ -f "$Q/AR_DATA_OK" ]; then train_eval rotworld500 $P/rotworld_strat500.hdf5 2000 42 $RUNS; AR_OK=1
else log "A-R 학습 건너뜀 (데이터 게이트 미통과 또는 생성 실패)"; fi
# 6. B3'
if md5_ok $PB/kgh_mimic_500_dirorig_magb2.hdf5 a025e391; then train_eval dirorig_magb2 $PB/kgh_mimic_500_dirorig_magb2.hdf5 2000 42 $RUNS; fi
# 7. 평가 seed 43
log "----- 평가 seed 43 -----"
train_eval kgh500 $REF 2000 43 $BASE
[ $AT_OK -eq 1 ] && train_eval translate500 $P/translate_strat500.hdf5 2000 43 $RUNS
train_eval reach4 $PB/kgh_mimic_500_reach4.hdf5 2000 43 $RUNS
[ $AR_OK -eq 1 ] && train_eval rotworld500 $P/rotworld_strat500.hdf5 2000 43 $RUNS
train_eval dirorig_magb2 $PB/kgh_mimic_500_dirorig_magb2.hdf5 2000 43 $RUNS
touch "$Q/MAIN_DONE"; log "===== 판정용 평가 끝 ====="
# 8. 체크포인트 민감도 (보고만, prereg §5.2)
log "----- 체크포인트 민감도 (보고만) -----"
train_eval kgh500 $REF 1800 42 $BASE
[ $AT_OK -eq 1 ] && { train_eval translate500 $P/translate_strat500.hdf5 1800 42 $RUNS; train_eval translate500 $P/translate_strat500.hdf5 1900 42 $RUNS; }
train_eval reach4 $PB/kgh_mimic_500_reach4.hdf5 1800 42 $RUNS
train_eval reach4 $PB/kgh_mimic_500_reach4.hdf5 1900 42 $RUNS
touch "$Q/ALL_DONE"; log "===== 큐 끝 ====="
