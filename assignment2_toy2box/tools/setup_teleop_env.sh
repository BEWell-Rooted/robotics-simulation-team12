#!/usr/bin/env bash
# 과제 2 텔레옵용 환경을 새 conda env 로 만든다 (A6000 lerobot-arena 와 같은 핵심 버전).
#   torch 2.7.0+cu128 · isaacsim 5.1.0 · numpy 1.26.0 · IsaacLab_RS e83a5d2
# 기존 conda env 와 ~/IsaacLab 은 건드리지 않는다. sudo 를 쓰지 않는다.
# Isaac Sim 5.1 은 NVIDIA 드라이버 580.65.06 이상(580 계열)을 요구한다 — 595 계열에서는 렌더링이 크래시한다.
#
#   bash setup_teleop_env.sh            (env 이름 기본 rs_toy2box, 설치 위치 ~/IsaacLab_RS)
set -euo pipefail
ENV=${ENV:-rs_toy2box}
DIR=${DIR:-$HOME/IsaacLab_RS}
COMMIT=e83a5d2
export OMNI_KIT_ACCEPT_EULA=YES PIP_NO_CACHE_DIR=1
# 사용자 pip.conf(예: 없어진 pypi.ngc.nvidia.com extra index)를 읽지 않는다 — 설정 파일은 그대로 둔다.
export PIP_CONFIG_FILE=/dev/null

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "== $(date +%T) conda env $ENV"
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.11
fi
conda activate "$ENV"
python --version

echo "== $(date +%T) torch 2.7.0+cu128"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo "== $(date +%T) isaacsim 5.1.0"
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# flatdict 4.0.1 은 빌드 때 pkg_resources 가 필요한데 setuptools 81+ 에는 없다 → 격리 없이 먼저 빌드
pip install "setuptools<81"
pip install --no-build-isolation flatdict==4.0.1

echo "== $(date +%T) IsaacLab_RS @ $COMMIT"
if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/cailab-hy/IsaacLab_RS.git "$DIR"
fi
cd "$DIR"
git checkout -q "$COMMIT"
git log -1 --oneline
./isaaclab.sh --install none

echo "== $(date +%T) numpy 1.26.0 · psutil 5.9.8 고정 (A6000 과 같게, isaacsim-kernel 요구)"
pip install numpy==1.26.0 psutil==5.9.8

echo "== $(date +%T) 버전"
python - <<'EOF'
import importlib.metadata as m, torch, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
for p in ["isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_mimic", "h5py", "numpy"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "NOT INSTALLED")
EOF
echo "== $(date +%T) done"
