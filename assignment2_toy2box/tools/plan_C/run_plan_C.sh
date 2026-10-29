#!/usr/bin/env bash
# 트랙 C 실험 실행.
#
#   bash run_plan_C.sh c3spark <학습 seed 101|111> [평가 seed=42]  # C3: baseline 체크포인트 그대로, 그리퍼 규칙만 majority (평가만, Spark)
#   bash run_plan_C.sh placebo <학습 seed 101|111> [평가 seed=42]  # 위약: 같은 체크포인트, 그리퍼 부호를 Bernoulli(P(open)) 로 다시 뽑음 (Spark)
#       → C3 판정은 'C3 대 위약' 200쌍. 같은 Spark 에서 짝짓는다: s101 은 gx10-3, s111 은 gx10-2 (교차 점검 1029 §1 #4).
#         두 팔은 같은 Spark 에서 동시에 돌려도 된다(고유 TMPDIR). A6000 에서 실행하면 ssh 로 Spark 에 보내고 기록을 가져와 분류한다.
#   bash run_plan_C.sh 1mode <학습 seed 101|111> [평가 seed=42]  # C-1mode(제출 가능한 근사 팔, C3 가 prereg 5절의 '지지'일 때만): A6000 에서 학습 → 평가 → 분류
#       (확증 팔이라 평가 seed 42 와 43 을 모두 돌린다: 두 번째 호출은 학습을 건너뛰고 평가만)
#   같은 학습 seed 의 c3spark 와 placebo 를 함께 돌릴 때는 c3spark 를 먼저 띄우고 'ckpt md5' 줄이 찍힌 뒤 placebo 를 띄운다
#   (원격 체크포인트 복사가 겹치지 않게).
#   종료 코드: 0 성공 · 3 기록이 이미 있음 · 4 체크포인트·config 확인 실패 · 5 rollout ≠ 100 (기록은 .failed 로) · 6 분류·지표 실패
#   bash run_plan_C.sh c3 <학습 seed>               # (옛 모드) A6000 에서 C3 — 판정에는 쓰지 않는다
#
# 평가는 seed ESEED(기본 42) · 100회 · horizon 800, 기록은 play_record.py 형식, 분류는 classify_failures.py v3.
# 출력 이름 <NAME>_seed<ESEED>.csv 옆에 <NAME>_seed<ESEED>.carry.csv (paired_compare.py 가 찾는 이름).
# 두 seed 를 동시에 돌려도 되도록 프로세스마다 TMPDIR 을 따로 준다(RMPFlow 가 reset 마다 /tmp/agibot.urdf 를 다시 씀).
set -uo pipefail
ARM=$1; SEED=$2; ESEED=${3:-42}
A2=$HOME/robotics-simulation-team12/assignment2_toy2box
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
DATA=./datasets/team12_kgh/kgh_mimic_500.hdf5
FULL=$HOME/IsaacLab_RS/logs/rs_a2_runs/plan_C
REC_DIR=$HOME/IsaacLab_RS/logs/rollout_records/plan_C
OUT_DIR=$A2/results/plan_C
mkdir -p "$FULL" "$REC_DIR" "$OUT_DIR"
export PYTHONUNBUFFERED=1
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate lerobot-arena
cd "$HOME/IsaacLab_RS"
ROOT=logs/robomimic/$TASK

case $ARM in
  c3)
    NAME=C3maj_s${SEED}_e2000
    CK=$(ls $ROOT/base_kgh500_s${SEED}/*/models/model_epoch_2000.pth | tail -1)
    export TMPDIR=/tmp/rs_a2_tmp_C_${NAME}_seed${ESEED}; mkdir -p "$TMPDIR"
    REC=$REC_DIR/${NAME}_seed${ESEED}.hdf5
    [ -f "$REC" ] && { echo "기록 $REC 가 이미 있다 — 중단"; exit 3; }
    echo "[run_plan_C] $(date '+%F %T') eval $NAME ckpt=$CK"
    timeout 10800 ./isaaclab.sh -p $A2/tools/plan_C/play_gripmix.py --grip_rule majority --task $TASK --checkpoint $CK \
      --num_rollouts 100 --horizon 800 --seed $ESEED --headless --record_file $REC > $FULL/eval_${NAME}_seed${ESEED}.log 2>&1
    EVAL_LOG=$FULL/eval_${NAME}_seed${ESEED}.log
    ;;
  1mode)
    NAME=C1mode_kgh500_s${SEED}
    RUN_ROOT=$ROOT/$NAME
    export TMPDIR=/tmp/rs_a2_tmp_C_${NAME}_seed${ESEED}; mkdir -p "$TMPDIR"
    if ! ls $RUN_ROOT/*/models/model_epoch_2000.pth >/dev/null 2>&1; then
      [ -d "$RUN_ROOT" ] && { echo "$RUN_ROOT 가 있는데 체크포인트가 없다 — 확인 필요, 중단"; exit 3; }
      echo "[run_plan_C] $(date '+%F %T') train $NAME (gmm.num_modes=1)"
      timeout 14400 ./isaaclab.sh -p $A2/tools/plan_C/train_override.py --seed $SEED --set algo.gmm.num_modes=1 \
        --task $TASK --algo bc --dataset $DATA --name $NAME > $FULL/train_${NAME}.log 2>&1
      grep -E "train_override|finished run|run failed|Traceback" $FULL/train_${NAME}.log
    fi
    CK=$(ls $RUN_ROOT/*/models/model_epoch_2000.pth | tail -1)
    python -I -c "import json,sys; c=json.load(open('$(dirname $(dirname $CK))/config.json')); assert c['algo']['gmm']['num_modes']==1 and c['train']['seed']==$SEED, c['algo']['gmm']; print('[run_plan_C] config 확인: num_modes 1, seed $SEED')" || exit 4
    REC=$REC_DIR/${NAME}_e2000_seed${ESEED}.hdf5
    [ -f "$REC" ] && { echo "기록 $REC 가 이미 있다 — 중단"; exit 3; }
    echo "[run_plan_C] $(date '+%F %T') eval $NAME ckpt=$CK"
    # 제출과 같은 경로: 조교 play.py 그대로 (play_record.py 는 play.py 를 바꾸지 않고 상태만 기록)
    timeout 10800 ./isaaclab.sh -p $A2/tools/play_record.py --task $TASK --checkpoint $CK \
      --num_rollouts 100 --horizon 800 --seed $ESEED --headless --record_file $REC > $FULL/eval_${NAME}_seed${ESEED}.log 2>&1
    EVAL_LOG=$FULL/eval_${NAME}_seed${ESEED}.log
    NAME=${NAME}_e2000
    ;;
  c3spark|placebo)
    case $SEED in 101) HOST=gx10-3;; 111) HOST=gx10-2;; *) echo "seed 는 101 | 111"; exit 2;; esac
    RULE=majority; [ "$ARM" = placebo ] && RULE=resample
    TAGA=C3maj; [ "$ARM" = placebo ] && TAGA=C3placebo
    NAME=${TAGA}_s${SEED}_e2000_${HOST}
    CK=$(ls $ROOT/base_kgh500_s${SEED}/*/models/model_epoch_2000.pth | tail -1)
    RCK=IsaacLab_RS/logs/ckpt/base_kgh500_s${SEED}/model_epoch_2000.pth
    RREC=IsaacLab_RS/logs/rollout_records/plan_C/${NAME}_seed${ESEED}.hdf5
    REC=$REC_DIR/${NAME}_seed${ESEED}.hdf5
    [ -f "$REC" ] && { echo "기록 $REC 가 이미 있다 — 중단"; exit 3; }
    ssh $HOST "mkdir -p ~/rs_a2_tools/plan_C ~/IsaacLab_RS/logs/ckpt/base_kgh500_s${SEED} ~/IsaacLab_RS/logs/rollout_records/plan_C ~/rs_a2_logs/plan_C"
    scp -q $A2/tools/plan_C/play_gripmix.py $HOST:~/rs_a2_tools/plan_C/
    scp -q $A2/tools/play_record.py $A2/tools/classify_failures.py $HOST:~/rs_a2_tools/
    # 체크포인트가 A6000 것과 같은지 md5 로 확인 (없으면 복사)
    L_MD5=$(md5sum $CK | cut -d' ' -f1); R_MD5=$(ssh $HOST "md5sum ~/$RCK 2>/dev/null | cut -d' ' -f1")
    [ "$L_MD5" = "$R_MD5" ] || { scp -q $CK $HOST:~/$RCK; R_MD5=$(ssh $HOST "md5sum ~/$RCK | cut -d' ' -f1"); }
    [ "$L_MD5" = "$R_MD5" ] || { echo "체크포인트 md5 불일치 — 중단"; exit 4; }
    ssh $HOST "test -f ~/$RREC" && { echo "$HOST:~/$RREC 가 이미 있다 — 중단"; exit 3; }
    echo "[run_plan_C] $(date '+%F %T') eval $NAME on $HOST rule=$RULE ckpt md5 $L_MD5"
    ssh $HOST "mkdir -p /tmp/rs_a2_tmp_C_${NAME}_seed${ESEED} && source ~/miniforge3/etc/profile.d/conda.sh && conda activate rs_toy2box && cd ~/IsaacLab_RS && \
      TMPDIR=/tmp/rs_a2_tmp_C_${NAME}_seed${ESEED} PYTHONUNBUFFERED=1 timeout 14400 ./isaaclab.sh -p ~/rs_a2_tools/plan_C/play_gripmix.py --grip_rule $RULE \
      --task $TASK --checkpoint ./logs/ckpt/base_kgh500_s${SEED}/model_epoch_2000.pth --num_rollouts 100 --horizon 800 --seed $ESEED \
      --headless --record_file ~/$RREC > ~/rs_a2_logs/plan_C/eval_${NAME}_seed${ESEED}.log 2>&1; echo RC=\$? >> ~/rs_a2_logs/plan_C/eval_${NAME}_seed${ESEED}.log"
    scp -q $HOST:~/$RREC $REC
    scp -q $HOST:~/$RREC.gripchanges.csv $OUT_DIR/${NAME}_seed${ESEED}_gripchanges.csv
    scp -q $HOST:~/rs_a2_logs/plan_C/eval_${NAME}_seed${ESEED}.log $FULL/eval_${NAME}_seed${ESEED}.log
    EVAL_LOG=$FULL/eval_${NAME}_seed${ESEED}.log
    # 귀속 (ii) 용 P(open): 같은 체크포인트를 이 기록 상태에 다시 넣어 계산 (CPU)
    python -I $A2/tools/plan_C/gmm_offline.py $CK $REC $FULL/gmm_offline_${NAME}_seed${ESEED}.npz > $FULL/gmm_offline_${NAME}_seed${ESEED}.log 2>&1
    python -I $A2/tools/plan_C/attribute_c3.py --rec $REC --gmm $FULL/gmm_offline_${NAME}_seed${ESEED}.npz \
      --csv $OUT_DIR/${NAME}_seed${ESEED}_attr.csv > $OUT_DIR/${NAME}_seed${ESEED}_attr.txt 2>&1
    ;;
  *) echo "arm 은 c3spark | placebo | 1mode | c3"; exit 2;;
esac
grep -E "Successful trials|play_gripmix\] rule" $EVAL_LOG
N_ROLL=$(python -I -c "import h5py; print(len(h5py.File('$REC','r')['rollouts']))" 2>/dev/null || echo 0)
if [ "$N_ROLL" != "100" ]; then
  # 부분 기록은 치운다 (다시 돌릴 때 "기록이 이미 있다" 에 막히지 않게; Spark 쪽 기록도)
  TS=$(date +%H%M%S); mv "$REC" "$REC.failed_$TS" 2>/dev/null
  [ -n "${HOST:-}" ] && ssh $HOST "mv ~/$RREC ~/$RREC.failed_$TS 2>/dev/null"
  echo "[run_plan_C] 기록 rollout 수 $N_ROLL ≠ 100 — 실패, 기록을 .failed 로 옮김"; exit 5
fi
python -I $A2/tools/classify_failures.py $REC --csv $OUT_DIR/${NAME}_seed${ESEED}.csv --json $OUT_DIR/${NAME}_seed${ESEED}.json > $OUT_DIR/${NAME}_seed${ESEED}.txt 2>&1 \
  || { echo "[run_plan_C] 분류 실패"; exit 6; }
python -I $A2/tools/carry_metrics.py $REC --csv $OUT_DIR/${NAME}_seed${ESEED}.carry.csv > $OUT_DIR/${NAME}_seed${ESEED}.carry.txt 2>&1 \
  || { echo "[run_plan_C] carry_metrics 실패"; exit 6; }
echo "[run_plan_C] $(date '+%F %T') done $NAME"
