#!/usr/bin/env bash
# A6000 의 rollout 기록 파일을 DGX Spark(기본 gx10-2)로 보내 render_record.py 로 장면 띠·영상을 만들고 가져온다.
# A6000(드라이버 595)은 Isaac Sim 5.1 렌더링이 안 되므로 렌더만 Spark 에서 한다 (docs/spark_setup_1028.md).
#
#   bash render_on_spark.sh <기록.hdf5> <로컬 출력 폴더> [render_record.py 인자...]
#   예) bash render_on_spark.sh ~/IsaacLab_RS/logs/rollout_records/base_kgh500_s101_e2000_seed42.hdf5 \
#          ~/IsaacLab_RS/videos/base_kgh500_s101 --rollouts fail --sheet --no_video --max_rollouts 100
set -euo pipefail
REC=$1; OUT=$2; shift 2
HOST=${HOST:-gx10-2}
A2=$HOME/robotics-simulation-team12/assignment2_toy2box
NAME=$(basename "$REC" .hdf5)
scp -q "$A2/tools/render_record.py" "$A2/tools/classify_failures.py" "$HOST":~/rs_a2_tools/
ssh "$HOST" "mkdir -p ~/IsaacLab_RS/logs/rollout_records ~/IsaacLab_RS/videos/$NAME"
scp -q "$REC" "$HOST":~/IsaacLab_RS/logs/rollout_records/
ssh "$HOST" "mkdir -p /tmp/rs_a2_tmp_render_$NAME && source ~/miniforge3/etc/profile.d/conda.sh && conda activate rs_toy2box && \
  cd ~/IsaacLab_RS && TMPDIR=/tmp/rs_a2_tmp_render_$NAME PYTHONUNBUFFERED=1 ./isaaclab.sh -p ~/rs_a2_tools/render_record.py \
  --record_file ./logs/rollout_records/$NAME.hdf5 --video_folder ./videos/$NAME --headless $* 2>&1 \
  | grep -E 'render_record|Error|Traceback'" || true
mkdir -p "$OUT"
scp -q "$HOST":~/IsaacLab_RS/videos/$NAME/* "$OUT"/ 2>/dev/null || true
echo "[render_on_spark] $(ls "$OUT" | wc -l) files -> $OUT"
