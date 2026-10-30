# 과제 2 Toy2Box — DGX Spark(gx10-2) 렌더링 환경 로그 (2026-10-28)

| 항목 | 값 |
| --- | --- |
| 세션 | RTX 6000 Claude Code 서브에이전트 (1028 세션) |
| 머신 | `ssh gx10-2` = hyounggeun@192.168.0.51, DGX Spark (`gx10-d7db`) |
| 시각 | 19:16–19:52 KST (gx10-2 시계는 UTC, 로그 시각은 UTC로 찍힘: 10:16–10:52) |
| 설치 스크립트 | `tools/setup_spark_env.sh` (gx10-2에는 `~/rs_a2_tools/setup_spark_env.sh`) |

## 요약

- **결론:** gx10-2에서 Isaac Sim 5.1 렌더링이 된다. 조교 demo 0 replay 영상(635프레임)과 RTX 6000 기록 파일의 상태 재생(`render_record.py`) 영상·장면 띠가 정상으로 나왔고, robomimic `play.py` 평가도 돈다.
- **RTX 6000 → gx10-2 흐름:** RTX 6000에서 `play_record.py`로 평가하면서 상태를 기록하고, 기록 파일을 gx10-2로 `scp`한 뒤 `render_record.py`로 mp4·jpg를 만들어 가져온다(아래 "실행 방법").
- **aarch64에서 x86과 달랐던 것 두 가지:** (1) 시스템 libgomp를 `LD_PRELOAD`로 먼저 올려야 Isaac Sim이 뜬다. (2) torch 2.7.0+cu128에 들어 있는 nvrtc 12.8은 GB10(sm_121)을 몰라서 TorchScript 퓨전이 죽는다. pip nvrtc 12.9를 `LD_PRELOAD`로 올려서 해결했다. 둘 다 conda env activate 훅에 넣어 두었다.
- **속도:** 렌더 재생은 약 16 프레임/s(960×540). 평가는 800 step rollout 하나에 32–38 s 걸려 RTX 6000(약 20 s)보다 1.6–1.9배 느리다.

## 설치한 것 (모두 새 폴더, sudo 없음)

| 위치 | 내용 |
| --- | --- |
| `~/miniforge3` | Miniforge (conda 26.9.4), `-b` 설치라 `~/.bashrc`는 그대로 |
| `~/miniforge3/envs/rs_toy2box` | 작업 env (python 3.11.18) |
| `~/IsaacLab_RS` | `cailab-hy/IsaacLab_RS` @ e83a5d2 (detached HEAD, RTX 6000과 같은 커밋) |
| `~/rs_a2_tools` | `replay_video.py`, `render_record.py`, `classify_failures.py`, `setup_spark_env.sh` (레포 사본) |
| `~/rs_a2_logs` | 실행 로그, 확인용 프레임·카메라 비교 png |
| `~/rs_a2_setup` | Miniforge 설치 파일, 시험용 스크립트(`env.sh`, `run.sh`, `run_ts.sh`, `jit_test.py`, `cam_test.py`, `lag_test.py`) |

기존 `~/comfyui`, `~/cosmos-predict2.5`, `~/Desktop`, `~/tier0_logs`, `~/wyd_spark_setup.sh`와 시스템 python 3.12는 건드리지 않았다. 패키지를 받을 때 pip 캐시(`~/.cache/pip`, 원래 있던 폴더)를 일부 썼다(4.4 GB).

## 버전

| 패키지 | gx10-2 | RTX 6000 (lerobot-arena) | 비고 |
| --- | --- | --- | --- |
| 하드웨어 | GB10 (sm_121), aarch64, 20코어, 121 GB | RTX 6000 Ada Generation 48 GB | |
| 드라이버 | 580.178.04 (CUDA 13.0) | 595.x | 5.1은 580 계열을 요구 |
| python | 3.11.18 | 3.11 | |
| torch / torchvision / torchaudio | 2.7.0+cu128 / 0.22.0 / 2.7.0 | 2.7.0+cu128 / 0.22.0+cu128 / 2.7.0+cu128 | aarch64 휠은 CUDA 라이브러리를 torch/lib에 넣어 둠 |
| isaacsim | 5.1.0.0 (`[all,extscache]`) | 5.1.0.0 | |
| isaaclab / tasks / assets / mimic / rl | 0.47.2 / 0.11.6 / 0.2.3 / 1.0.15 / 0.4.4 | 같음 | |
| robomimic / egl_probe | 0.4.0 / 1.0.2 | 0.4.0 / 1.0.2 | egl_probe는 소스 빌드(cmake), 문제없음 |
| numpy / psutil / h5py | 1.26.0 / 5.9.8 / 3.16.0 | 같음 | |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | (없음) | gx10-2에만 추가, LD_PRELOAD |
| warp-lang / imageio-ffmpeg / transformers | 1.18.0 / 0.7.0 / 5.21.0 | 1.17.0 / 0.6.0 / 4.57.1 | 버전 고정 없이 isaaclab이 깔아 생긴 차이, 실행 영향 없음 |

남은 `pip check` 경고는 Dell과 같다: click 8.5.0·idna 3.20(isaacsim-kernel은 8.1.7·3.10 고정), packaging 23.2(isaacsim-core는 23.0), ipython psutil ≥ 7. 실행에는 영향이 없었다.

### torch를 2.7.0으로 둔 이유

- 처음 지시대로 `torch==2.7.0 torchvision==0.22.0`(cu128)을 깔았다. `get_arch_list()`는 sm_90·100·120이지만 GB10(12.1)에서 matmul·conv·LSTM·torchvision nms가 모두 돌았다("no kernel image" 오류 없음). sm_120 바이너리가 sm_121에서 돈다.
- `isaacsim-core 5.1.0.0` aarch64 휠이 `torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0`을 고정한다. isaacsim 설치 때 torch는 "already satisfied"로 그대로 남았다.
- `./isaaclab.sh --install`은 aarch64에서 torch를 **2.9.0+cu130**으로 지우고 다시 깐다(`ensure_cuda_torch`, 설치 전후 두 번). isaacsim 고정값과도, RTX 6000과도 달라지므로 `--install` 대신 확장 5개를 `pip install -e`로 직접 깔았다. `isaaclab_rl[none]`/`isaaclab_mimic[none]`은 추가 의존이 없어서 결과가 같다. robomimic은 `isaaclab_mimic[robomimic]`과 같은 git 태그로 따로 깔았다.
- 2.7.0에서 남은 문제는 nvrtc 하나였다(아래 문제 2). nvrtc 12.9로 해결해서 torch를 바꾸지 않았다.

## 설치 시간 (gx10-2, KST)

| 단계 | 시각 | 걸린 시간 |
| --- | --- | --- |
| Miniforge 받기·설치, env 생성 | 19:16–19:17 | 약 1분 |
| torch 2.7.0+cu128 | 19:17:33–19:18:26 | 53 s |
| isaacsim[all,extscache] 5.1.0 | 19:18:57–19:23:45 | 4분 48초 (extscache-kit 휠만 3.96 GB) |
| setuptools<81 + flatdict, 확장 5개 | 19:24:22–19:24:52 | 30 s |
| robomimic 0.4.0 (egl_probe 빌드 포함) | 19:25:17–19:25:26 | 9 s |
| 스크립트 재현 확인 (새 env `rs_verify`, 캐시 없이, 전 단계) | 19:41:16–19:47:05 | 약 6분 |

## 확인 결과

로그는 gx10-2 `~/rs_a2_logs/`에 있다. 걸린 시간은 `/usr/bin/time` 기준 전체 wall time(앱 시작 포함)이다.

| # | 확인 | 결과 | 시간 | 로그 |
| --- | --- | --- | --- | --- |
| a | `list_envs.py --headless` | `Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0` 있음 (124번) | 5.3 s | `5a_list_envs.log` |
| b | `replay_video.py` 조교 demo 0 | `Finished replaying 1 episode.`, `wrote 635 frames` → `~/IsaacLab_RS/videos/ta_ref_replay/toy2box_agibot_ep0.mp4` (4,894,324 B, H.264 Main 1280×720, 635프레임) | 65.6 s (첫 실행, 셰이더 캐시 생성 포함), 최대 RSS 8.9 GB | `5b_replay_video.log` |
| c | `render_record.py` 기록 rollout 0 (`--sheet --stride 2`) | 401프레임 mp4(H.264 Main 960×540, 10 fps) + 장면 띠 jpg | 41.1 s (프레임 렌더 25.3 s = 약 16 프레임/s), 최대 RSS 7.5 GB | `5c_render_record.log` |
| c2 | `render_record.py --rollouts fail --no_video --sheet` | rollout 0·1·2 장면 띠 3장 (분류는 셋 다 approach/close_misaligned) | 17.8 s | `5c2_render_fail_sheets.log` |
| d | robomimic `play.py` dev 체크포인트(e100), 2 rollout, horizon 800, seed 42 | `Successful trials: 0, out of 2 trials` (약한 정책이라 예상대로) | 앱 시작 8.5 s, trial 0 37.7 s, trial 1 32.3 s, 전체 79 s | `5d_play.log` (줄마다 유닉스 시각) |

- **b 프레임 확인:** ffmpeg로 300·634번 프레임을 뽑아 보았다. 300번은 로봇·테이블·트럭·분홍 상자가 제대로 렌더되어 있고 트럭은 테이블 위에 있다. 634번에서는 트럭이 상자 안에 있다. Dell 결과(10/28)와 같다.
- **c 프레임 확인:** 기본 카메라(태스크 cfg viewer)로 먼저 돌렸더니 렌더는 정상이었지만 트럭이 15 px 정도로 작았다. 아래 "카메라"의 close 시점으로 바꾸고 다시 확인했다. rollout 0에서는 팔이 트럭 쪽으로 내려가 트럭 바로 옆 0.1 m 안에서 손을 여닫으며 맴돈다(`approach/close_misaligned`, t=298 첫 닫기). 분류 결과와 화면이 맞는다.
- **프레임 밀림 확인 (`lag_test.py`):** 상태를 바꾼 뒤 한 번만 렌더한 프레임을 같은 상태로 4번 렌더한 기준 프레임과 비교했다. 평균 화소 차는 1.3–2.2였고, 직전 상태와 비교하면 9.3–12.0이었다. 한 번 렌더로 그 step의 상태가 나오므로 영상이 한 step씩 밀리지 않는다. 남는 1–2는 TAA 잡음이다.
- **d 속도:** 800 step rollout 하나에 32–38 s 걸린다. RTX 6000의 약 20 s보다 1.6–1.9배 느리다. 100 rollout 평가는 약 1시간이 걸린다(RTX 6000 약 35분). 평가는 RTX 6000에서 하고 Spark는 렌더링에 쓰는 편이 낫다.

## 문제와 해결

1. **Isaac Sim이 바로 종료됨 (a 첫 실행, 0.6 s):** `WARNING: For the application to run, some shared libraries must be loaded before others ... LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"`. `isaacsim/__init__.py`의 `aarch_preload_checking`이 aarch64에서 시스템 libgomp(와 `torch.libs/libgomp*`, 2.7.0 휠에는 없음)가 `LD_PRELOAD`에 없으면 `sys.exit`한다. → env의 `etc/conda/activate.d/rs_aarch64_env.sh`에서 `LD_PRELOAD`에 추가했다.
2. **env 생성 중 `RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)` (b 첫 실행):** `RigidObjectData.__init__` → `math_utils.normalize`(`@torch.jit.script`)에서 났다. TorchScript GPU 퓨전이 nvrtc로 장치 아키텍처(sm_121) 커널을 컴파일하는데, torch 2.7.0+cu128에 들어 있는 nvrtc 12.8은 sm_121을 모른다. torch는 모르는 새 nvrtc 버전이면 장치 아키텍처를 그대로 넘기므로 낮춰 주지도 않는다. 같은 오류를 `jit_test.py` 5줄로 재현했다. → `pip install nvidia-cuda-nvrtc-cu12==12.9.86`을 깔고 그 `libnvrtc.so.12`를 `LD_PRELOAD`했다(같은 훅). 재현 스크립트가 통과했고, 그 뒤 b·c·d가 모두 돌았다. (대안: `PYTORCH_JIT=0`으로 TorchScript를 끄거나, isaaclab.sh 기본값인 torch 2.9.0+cu130으로 올리기. 쓰지 않았다.)
3. **psutil 7.2.2:** `isaaclab_mimic` 설치 때 ipython이 올렸다. 5.9.8로 되돌렸다(Dell과 같음). aarch64에는 5.9.8 휠이 없어 소스 빌드했고 문제없었다.
4. **flatdict 4.0.1:** Dell과 같은 문제가 있어 같은 방법으로 미리 처리했다(`setuptools<81` + `--no-build-isolation`).
5. **`render_record.py`의 mp4가 960×544로 늘어남:** imageio-ffmpeg 기본 `macro_block_size=16` 때문에 540을 544로 리사이즈했다. → `macro_block_size=2`.

## `render_record.py` 수정 (처음 실행 전에는 한 번도 돌려 보지 않은 코드)

`reset_to(state, None, is_relative=True)` → `sim.render()` → `env.render()` 흐름은 처음부터 맞게 동작했다. 고친 것은 아래와 같다.

- **카메라 프리셋 `--view close|wide`, 기본 close:** close = eye (0.55, −1.2, 0.5), lookat (0.07, −0.1, −0.42). 로봇 오른쪽 앞 위에서 손·트럭·상자를 크게 본다. wide는 태스크 cfg viewer(eye 1.5 −1.0 1.5, lookat 0.5 0 0)로 장면 전체를 본다. `--eye/--lookat`을 주면 프리셋보다 우선한다. 시작할 때 쓰는 카메라를 출력한다.
- **장면 띠:** 8장을 한 줄(3840×270, 글자가 안 보임)로 붙이던 것을 2×4 격자(1920×562)로 바꿨다. 칸마다 `t=… 이정표 이름`을 붙이고, 이정표가 모자라면 고르게 채운다. 원래 코드는 `sorted(...)[:8]`이라 앞쪽만 남고 마지막 step이 빠질 수 있었다. 칸마다 한 번 더 렌더해서 TAA 잔상을 줄였다.
- rollout의 `states/`를 프레임마다 gzip 청크로 다시 읽지 않고 한 번에 읽는다.
- 고른 rollout이 없거나 기록에 없는 번호를 주면 IndexError로 죽지 않고 알림 문구를 내고 끝낸다.
- mp4의 `macro_block_size=2`(위 5), 영상마다 렌더 시간을 출력한다.

### 카메라 고르기

`cam_test.py`로 rollout 1·2의 t = 0/300/700, 0/400/800을 후보 시점마다 렌더해 비교했다(`~/rs_a2_logs/cam_test{1..4}.png`).

- 기본(태스크 cfg): 장면 전체가 보이지만 트럭이 작아 손과 트럭의 어긋남이 안 보인다.
- 낮은 시점(eye z −0.3–0.0, lookat z −0.6): 상자가 화면을 가리고, 팔을 들면 손이 화면 위로 잘린다.
- 로봇 오른쪽 옆 위(eye (0.3, −1.0, 0.4)): 손과 트럭이 가장 크게 보이지만 팔을 높이 들면 잘린다.
- **close(채택) eye (0.55, −1.2, 0.5), lookat (0.07, −0.1, −0.42):** 트럭(로봇 쪽 끝 x −0.14 위치 포함), 손가락, 상자가 한 화면에 들어오고, 팔을 반쯤 든 자세까지 보인다. 정책이 팔을 아주 높이 뒤로 들면(rollout 1, t=457) 손이 화면 밖으로 나간다. 그럴 때는 `--view wide`를 쓴다.

## 실행 방법

gx10-2 셸에서 (env를 켜면 LD_PRELOAD와 EULA가 자동으로 설정된다):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rs_toy2box
cd ~/IsaacLab_RS
./isaaclab.sh -p ~/rs_a2_tools/render_record.py --record_file ./logs/rollout_records/<이름>.hdf5 \
    --rollouts fail --sheet --video_folder ./videos/<이름> --headless
#   --rollouts all | fail | success | 0 3 7   --stride 2 (10 fps, 절반 시간)   --no_video (장면 띠만)   --view wide
```

RTX 6000에서 한 번에 (기록 보내기 → 렌더 → 결과 가져오기):

```bash
N=<이름>   # ~/IsaacLab_RS/logs/rollout_records/$N.hdf5
scp ~/IsaacLab_RS/logs/rollout_records/$N.hdf5 gx10-2:IsaacLab_RS/logs/rollout_records/
ssh gx10-2 "source ~/miniforge3/etc/profile.d/conda.sh && conda activate rs_toy2box && cd ~/IsaacLab_RS && \
  ./isaaclab.sh -p ~/rs_a2_tools/render_record.py --record_file ./logs/rollout_records/$N.hdf5 \
  --rollouts fail --sheet --video_folder ./videos/$N --headless 2>&1 | grep -E 'render_record|Error|Traceback'"
scp -r gx10-2:IsaacLab_RS/videos/$N ~/IsaacLab_RS/videos/
```

- 시간: 앱 시작·데우기 약 15–18 s, 그다음 800 step rollout 하나에 약 50 s(stride 1)나 약 25 s(stride 2)다. 장면 띠만 만들면 rollout당 약 1 s다. 실패 rollout 12개를 stride 2로 렌더하면 약 5분이다.
- 오래 걸리는 작업(rollout 수십 개)은 `tmux new -s render`로 돌린다(gx10-2에 tmux 있음).
- `~/rs_a2_tools/`의 `render_record.py`·`classify_failures.py`는 레포 사본이다. 레포에서 고치면 `scp ~/robotics-simulation-team12/assignment2_toy2box/tools/{render_record.py,classify_failures.py} gx10-2:rs_a2_tools/`로 다시 보낸다.
- 기록 파일에 `states/`가 있어야 한다(`play_record.py`를 `--no_states` 없이 돌린 것).
- GUI 창(텔레옵)은 gx10-2에서 확인하지 않았다(headless만 확인).

## 스크립트 재현 확인

- `ENV=rs_verify DIR=~/rs_a2_setup/IsaacLab_RS_verify bash ~/rs_a2_tools/setup_spark_env.sh`로 새 env를 처음부터 만들었다(pip 캐시 끔). 19:41:16–19:47:05에 설치가 모두 끝나 약 6분 걸렸다. 로그는 `setup_spark_env_verify.log`.
- 첫 실행은 마지막 확인 단계에서만 실패했다. 확인용 python을 `python - <<EOF`(stdin)로 돌려서 `torch.jit.script`가 소스를 읽지 못했다(`OSError: could not get source code`). 확인 코드를 파일(`~/rs_a2_setup/check_env.py`)로 쓰고 돌리게 고쳤다. 같은 env에 다시 돌리니 설치 단계는 already satisfied로 1분 안에 지나갔고, 확인이 모두 통과했다(`setup_spark_env_verify2.log`, `SCRIPT_RC=0`). 통과한 확인: torch 2.7.0+cu128 GB10 (12, 1), matmul, TorchScript 퓨전, `robomimic.utils.file_utils` import, 버전 목록, `list_envs`의 Toy2Box.
- 확인이 끝난 뒤 `rs_verify` env와 `IsaacLab_RS_verify` 폴더는 지웠다. 작업 env는 `rs_toy2box` 하나다.
- 위 "실행 방법"의 RTX 6000 한 줄 명령(`conda activate`만 쓰고 시험용 `env.sh`는 쓰지 않음)도 그대로 돌려 보았다. `--rollouts fail --sheet --stride 2`로 rollout 3개의 mp4(각 401프레임, 24.5–24.9 s)와 장면 띠 3장이 나왔고, ssh까지 포함해 92.6 s 걸렸다.
