#!/usr/bin/env bash
# 과제 2 baseline 한 줄 실행: BC-RNN 학습(조교 config, 2000 epoch) → play.py 평가(검증 seed 42, 100회, horizon 800)
#   → rollout 상태 기록(play_record.py) → 실패 분류(classify_failures.py).
# 학습 seed 는 train_seed.py 래퍼로 준다 (train.py 에는 --seed 가 없음).
#
#   bash run_baseline.sh <tag> <mimic.hdf5 (IsaacLab_RS 기준 경로)> <학습 seed> [eval epoch=2000] [평가 seed=42]
#   예) bash run_baseline.sh kgh500 ./datasets/team12_kgh/kgh_mimic_500.hdf5 101
#       bash run_baseline.sh kgh500 ./datasets/team12_kgh/kgh_mimic_500.hdf5 101 2000 43   # 이미 학습된 run 을 seed 43 으로 평가만
#   환경 변수 OUTDIR: 분류 결과 폴더 (기본 A2/results/baseline). 처치 팔은 OUTDIR=A2/results/runs 를 권함
#   종료 코드: 0 성공 · 3/4 run 폴더·체크포인트 문제 · 5 평가 실패(rc ≠ 0 또는 rollout ≠ 100, 부분 기록은 .failed 로) · 6 분류·지표 실패
#
# 산출물
#   run 폴더   ~/IsaacLab_RS/logs/robomimic/<task>/base_<tag>_s<seed>/<시각>/models/model_epoch_<N>.pth
#   기록       ~/IsaacLab_RS/logs/rollout_records/base_<tag>_s<seed>_e<N>_seed42.hdf5
#   요약 로그   A2/docs/run_logs/b2_train_*.log · b3_eval_*.log (전체 로그는 ~/IsaacLab_RS/logs/rs_a2_runs/)
#   분류 결과   A2/results/baseline/base_<tag>_s<seed>_e<N>_seed42.{csv,json,txt}, plots/
#
# 주의: RMPFlow 는 env reset 마다 URDF 를 tempfile.gettempdir() 에 강제로 다시 받는다.
#       여러 프로세스가 같은 /tmp/agibot.urdf 를 쓰면 깨지므로 프로세스마다 TMPDIR 을 따로 준다.
set -uo pipefail
TAG=$1; DATA=$2; SEED=$3; EP=${4:-2000}; ESEED=${5:-42}
A2=$HOME/robotics-simulation-team12/assignment2_toy2box
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
NAME=base_${TAG}_s${SEED}
FULL=$HOME/IsaacLab_RS/logs/rs_a2_runs
OUTDIR=${OUTDIR:-$A2/results/baseline}
mkdir -p "$FULL" "$OUTDIR/plots"
export TMPDIR=/tmp/rs_a2_tmp_${NAME}_seed${ESEED}; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate lerobot-arena
cd "$HOME/IsaacLab_RS"

RUN_ROOT=logs/robomimic/$TASK/$NAME
if ls $RUN_ROOT/*/models/model_epoch_${EP}.pth >/dev/null 2>&1; then
  echo "[run_baseline] $(date +%T) 체크포인트가 이미 있다 — 학습 생략"
else
  if [ -d "$RUN_ROOT" ]; then echo "[run_baseline] $RUN_ROOT 가 이미 있는데 체크포인트가 없다. 확인 필요 — 중단"; exit 3; fi
  echo "[run_baseline] $(date '+%F %T') train $NAME data=$DATA seed=$SEED"
  timeout 14400 ./isaaclab.sh -p $A2/tools/train_seed.py --seed $SEED --task $TASK --algo bc --dataset $DATA --name $NAME \
    > $FULL/b2_train_${NAME}.log 2>&1
  echo "train rc=$?" >> $FULL/b2_train_${NAME}.log
  grep -E "train_seed|Saving|finished run|run failed|Error|Traceback|Epoch (1|100|500|1000|1500|2000)$" $FULL/b2_train_${NAME}.log \
    > $A2/docs/run_logs/b2_train_${NAME}.log
  echo "[run_baseline] $(date '+%F %T') train done"
fi
CK=$(ls $RUN_ROOT/*/models/model_epoch_${EP}.pth 2>/dev/null | tail -1)
if [ -z "$CK" ]; then echo "[run_baseline] 체크포인트 없음 — 중단"; exit 4; fi
REC=logs/rollout_records/${NAME}_e${EP}_seed${ESEED}.hdf5
OUT=$OUTDIR/${NAME}_e${EP}_seed${ESEED}
if [ ! -f "$REC" ]; then
  echo "[run_baseline] $(date '+%F %T') eval $CK"
  timeout 10800 ./isaaclab.sh -p $A2/tools/play_record.py --task $TASK --checkpoint $CK \
    --num_rollouts 100 --horizon 800 --seed $ESEED --headless --record_file $REC > $FULL/b3_eval_${NAME}_e${EP}_seed${ESEED}.log 2>&1
  ERC=$?
  echo "eval rc=$ERC" >> $FULL/b3_eval_${NAME}_e${EP}_seed${ESEED}.log
  grep -E "play_record|Trial [0-9]+:|Successful trials|Success rate|Trial Results|Error|Traceback|rc=" $FULL/b3_eval_${NAME}_e${EP}_seed${ESEED}.log \
    > $A2/docs/run_logs/b3_eval_${NAME}_e${EP}_seed${ESEED}.log
  echo "[run_baseline] $(date '+%F %T') eval done: $(grep -E 'Successful trials' $FULL/b3_eval_${NAME}_e${EP}_seed${ESEED}.log)"
else
  ERC=0; echo "[run_baseline] 기록 $REC 가 이미 있다 — 평가 생략, rollout 수만 확인"
fi
# 평가(이번 실행이든 예전 실행이든)가 죽으면 부분 기록이 남는다(rollout 마다 flush) — 치우고 실패로 끝낸다
N_ROLL=$(python -I -c "import h5py; print(len(h5py.File('$REC','r')['rollouts']))" 2>/dev/null || echo 0)
if [ "$ERC" != "0" ] || [ "$N_ROLL" != "100" ]; then
  mv "$REC" "$REC.failed_$(date +%H%M%S)" 2>/dev/null
  echo "[run_baseline] 평가 실패 (rc=$ERC, rollout $N_ROLL/100) — 기록을 .failed 로 옮김"; exit 5
fi
python -I $A2/tools/classify_failures.py $REC --csv $OUT.csv --json $OUT.json --plots $OUTDIR/plots \
  --max_plots 12 > $OUT.txt 2>&1 || { echo "[run_baseline] 분류 실패"; cat $OUT.txt; exit 6; }
cat $OUT.txt
python -I $A2/tools/carry_metrics.py $REC --csv $OUT.carry.csv > $OUT.carry.txt 2>&1 || { echo "[run_baseline] carry_metrics 실패"; exit 6; }
echo "[run_baseline] $(date '+%F %T') all done"
