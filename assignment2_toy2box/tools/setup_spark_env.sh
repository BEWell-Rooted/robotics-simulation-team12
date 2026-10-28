#!/usr/bin/env bash
# 과제 2 렌더링용 환경을 DGX Spark(aarch64 · GB10 sm_121 · 드라이버 580)에 새로 만든다. gx10-2 에서 확인 (2026-10-28).
#   Miniforge(~/miniforge3) · conda env rs_toy2box(python 3.11) · torch 2.7.0+cu128 · isaacsim 5.1.0
#   IsaacLab_RS e83a5d2 · robomimic 0.4.0 · numpy 1.26.0 · psutil 5.9.8   (A6000 과 같은 핵심 버전)
# sudo 를 쓰지 않는다. 시스템 python·드라이버·CUDA 와 기존 폴더는 건드리지 않는다.
#
# x86 판(setup_teleop_env.sh)과 다른 점:
#  1) ./isaaclab.sh --install 을 쓰지 않고 확장 5개를 pip install -e 로 직접 깐다.
#     isaaclab.sh 는 aarch64 에서 torch 를 2.9.0+cu130 으로 지우고 다시 깐다(ensure_cuda_torch, 설치 전후 두 번).
#     그런데 isaacsim-core 5.1 aarch64 휠은 torch==2.7.0 을 고정하고, A6000 도 2.7.0 이라 2.7.0 을 지킨다.
#     (cmake 가 없으면 sudo apt-get 도 부른다 — Spark 에는 cmake 가 있어 해당 없음)
#  2) 실행 때 LD_PRELOAD 두 개가 필요하다 → env 의 activate.d 훅으로 넣는다 (conda activate 하면 자동).
#     - /lib/aarch64-linux-gnu/libgomp.so.1 : isaacsim/__init__.py 의 aarch64 검사가 없으면 바로 종료한다.
#     - nvrtc 12.9 (pip nvidia-cuda-nvrtc-cu12) : torch 2.7.0+cu128 에 든 nvrtc 12.8 은 sm_121 을 몰라
#       Isaac Lab 의 @torch.jit.script 함수가 GPU 퓨전 때 "nvrtc: error: invalid value for --gpu-architecture" 로 죽는다.
#  3) egl_probe(robomimic 의존)는 aarch64 에서도 cmake 로 잘 빌드된다 → robomimic 을 그대로 깐다.
#
#   bash setup_spark_env.sh        (ENV 기본 rs_toy2box, DIR 기본 ~/IsaacLab_RS, CONDA_BASE 기본 ~/miniforge3)
# 다시 돌려도 된다(있는 것은 건너뛰거나 "already satisfied").
set -eo pipefail
ENV=${ENV:-rs_toy2box}
DIR=${DIR:-$HOME/IsaacLab_RS}
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
COMMIT=e83a5d2
export OMNI_KIT_ACCEPT_EULA=YES PIP_NO_CACHE_DIR=1
# 사용자 pip.conf 를 읽지 않는다 (Dell 에서 죽은 extra index 로 매번 재시도하던 문제) — 설정 파일은 그대로 둔다.
export PIP_CONFIG_FILE=/dev/null

[ "$(uname -m)" = "aarch64" ] || { echo "aarch64 전용 스크립트다 (x86 은 setup_teleop_env.sh)"; exit 1; }

echo "== $(date +%T) Miniforge → $CONDA_BASE"
if [ ! -x "$CONDA_BASE/bin/conda" ]; then
  mkdir -p "$HOME/rs_a2_setup"
  inst="$HOME/rs_a2_setup/Miniforge3-Linux-aarch64.sh"
  curl -fsSL -o "$inst" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  bash "$inst" -b -p "$CONDA_BASE"   # -b: 묻지 않음, ~/.bashrc 수정 안 함
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "== $(date +%T) conda env $ENV (python 3.11)"
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.11
fi
conda activate "$ENV"
python --version

echo "== $(date +%T) torch 2.7.0+cu128 (aarch64 휠, CUDA 라이브러리 내장, sm_90/100/120 — sm_120 바이너리가 GB10 sm_121 에서 돈다)"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

echo "== $(date +%T) isaacsim 5.1.0 (isaacsim-core 가 torch==2.7.0 을 고정 → torch 는 그대로 남는다)"
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
# ./isaaclab.sh --install none 과 같은 확장 설치에서 torch 교체만 뺀 것
for ext in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
  pip install --editable "source/$ext"
done

echo "== $(date +%T) robomimic 0.4.0 (isaaclab_mimic[robomimic] 과 같은 git 태그, egl_probe 는 소스 빌드)"
pip install "robomimic@git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0"

echo "== $(date +%T) nvrtc 12.9 (sm_121 지원, LD_PRELOAD 로 torch 내장 12.8 대신 쓴다)"
pip install "nvidia-cuda-nvrtc-cu12==12.9.86"

echo "== $(date +%T) numpy 1.26.0 · psutil 5.9.8 고정 (A6000 과 같게, isaacsim-kernel 요구 — ipython 이 psutil 7 로 올려 놓는다)"
pip install numpy==1.26.0 psutil==5.9.8

echo "== $(date +%T) activate.d 훅 (LD_PRELOAD · EULA)"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/rs_aarch64_env.sh" <<'EOS'
# rs_toy2box (aarch64 DGX Spark / GB10) 실행 환경 — setup_spark_env.sh 가 씀
#  1) Isaac Sim 5.1 pip 판은 시스템 libgomp 를 먼저 올려야 뜬다 (isaacsim/__init__.py aarch_preload_checking)
#  2) torch 2.7.0+cu128 에 든 nvrtc 12.8 은 sm_121(GB10)을 몰라 TorchScript 퓨전이
#     "nvrtc: error: invalid value for --gpu-architecture" 로 죽는다 → pip nvidia-cuda-nvrtc-cu12 12.9 를 먼저 올린다
export _RS_PREV_LD_PRELOAD="${LD_PRELOAD-}"
for __lib in /lib/aarch64-linux-gnu/libgomp.so.1 "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"; do
  case ":${LD_PRELOAD-}:" in *":$__lib:"*) ;; *) [ -r "$__lib" ] && export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$__lib";; esac
done
unset __lib
export OMNI_KIT_ACCEPT_EULA=YES
EOS
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/rs_aarch64_env.sh" <<'EOS'
export LD_PRELOAD="${_RS_PREV_LD_PRELOAD-}"; [ -z "$LD_PRELOAD" ] && unset LD_PRELOAD; unset _RS_PREV_LD_PRELOAD
EOS
conda deactivate
conda activate "$ENV"
echo "LD_PRELOAD=$LD_PRELOAD"

echo "== $(date +%T) 확인"
# torch.jit.script 는 소스 파일이 있어야 컴파일하므로 stdin(python -) 대신 파일로 돌린다
mkdir -p "$HOME/rs_a2_setup"
CHECK="$HOME/rs_a2_setup/check_env.py"
cat > "$CHECK" <<'EOF'
import importlib.metadata as m
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
a = torch.randn(1024, 1024, device="cuda")
print("matmul", float((a @ a).abs().mean()))


@torch.jit.script
def normalize(x: torch.Tensor) -> torch.Tensor:  # Isaac Lab math_utils 와 같은 꼴 — 퓨전이 nvrtc 로 sm_121 코드를 만든다
    return x / x.norm(p=2, dim=-1).clamp(min=1e-9).unsqueeze(-1)


x = torch.randn(4, 3, device="cuda")
for _ in range(5):
    y = normalize(x)
torch.cuda.synchronize()
print("torch.jit fusion ok", bool(torch.allclose(y.norm(dim=-1), torch.ones(4, device="cuda"))))
import robomimic.utils.file_utils  # noqa: F401
for p in ["isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_mimic", "robomimic", "h5py", "numpy", "psutil", "nvidia-cuda-nvrtc-cu12"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "NOT INSTALLED")
EOF
(cd "$HOME/rs_a2_setup" && python "$CHECK")
cd "$DIR"
./isaaclab.sh -p scripts/environments/list_envs.py --headless 2>&1 | grep -E "Toy2Box" || { echo "list_envs 에 Toy2Box 가 없다"; exit 1; }
echo "== $(date +%T) done"
