#!/usr/bin/env bash
# Toy2Box 키보드 텔레옵 실행기 (Dell 등 렌더링이 되는 머신의 데스크톱 터미널에서).
#   bash teleop.sh practice              연습 — 저장 안 함
#   bash teleop.sh record <이름> <세션> [개수]   녹화 — datasets/team12_<이름>_s<세션>.hdf5 (기본 10개)
# 같은 파일이 이미 있으면 녹화를 거부한다 (record_demos.py 는 같은 이름이면 기존 시범을 지운다).
set -uo pipefail
ENV=${ENV:-rs_toy2box}
DIR=${DIR:-$HOME/IsaacLab_RS}
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$DIR"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1

case "${1:-}" in
  practice)
    ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
      --task "$TASK" --num_envs 1 --teleop_device keyboard
    ;;
  record)
    name=${2:?이름을 주세요 (예: park)}
    sess=${3:?세션 번호를 주세요 (예: 1)}
    n=${4:-10}
    out=./datasets/team12_${name}_s${sess}.hdf5
    if [ -e "$out" ]; then
      echo "이미 있는 파일입니다: $out — 세션 번호를 바꾸세요 (덮어쓰면 기존 시범이 지워집니다)."
      exit 1
    fi
    mkdir -p datasets
    echo "녹화: $out (성공 $n 개까지). 실수하면 R, 끝나면 창이 닫힙니다."
    ./isaaclab.sh -p scripts/tools/record_demos.py \
      --task "$TASK" --dataset_file "$out" --num_demos "$n" --teleop_device keyboard
    ls -la "$out" 2>/dev/null && python "$HOME/rs_a2_tools/summarize_demos.py" "$out"
    ;;
  *)
    sed -n 2,5p "$0"
    exit 1
    ;;
esac
