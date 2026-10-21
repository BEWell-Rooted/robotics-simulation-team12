"""replay_demos.py 를 고치지 않고 replay 영상을 남기는 래퍼.

이 체크아웃의 scripts/tools/replay_demos.py 에는 --video / --video_folder 인자가 없다
(조교 문서 명령을 그대로 쓰면 'unrecognized arguments' 오류).
이 래퍼는 두 인자를 먼저 떼어 내고 --enable_cameras 를 켠 뒤, gymnasium.make 를 감싸
base env 의 step 마다 render() 프레임을 모으고, close 때 mp4(20 fps, H.264) 로 쓴다.
replay_demos.py 는 gym.make(...).unwrapped 를 쓰므로 gym 래퍼(RecordVideo)는 통하지 않는다.

사용 (~/IsaacLab_RS 에서):
  ./isaaclab.sh -p ~/robotics-simulation-team12/assignment2_toy2box/tools/replay_video.py \
      --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
      --dataset_file ./datasets/ta_ref/toy2box_agibot.hdf5 --num_envs 1 --select_episodes 0 \
      --video_folder ./videos/ta_ref_replay --headless
여러 에피소드를 고르면 한 파일에 이어 붙는다. 확인용으로는 1개씩 고른다.
"""
import os
import runpy
import sys

argv = sys.argv[1:]
video_folder = "./videos/replay"
video_length = 2000
fps = 20
out = []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--video":
        i += 1
    elif a == "--video_folder":
        video_folder = argv[i + 1]
        i += 2
    elif a == "--video_length":
        video_length = int(argv[i + 1])
        i += 2
    else:
        out.append(a)
        i += 1
if "--enable_cameras" not in out:
    out.append("--enable_cameras")

# 파일 이름 꼬리표: 데이터셋 이름 + 고른 에피소드 번호
tag = "replay"
if "--dataset_file" in out:
    tag = os.path.splitext(os.path.basename(out[out.index("--dataset_file") + 1]))[0]
if "--select_episodes" in out:
    j = out.index("--select_episodes") + 1
    eps = []
    while j < len(out) and not out[j].startswith("--"):
        eps.append(out[j])
        j += 1
    tag += "_ep" + "-".join(eps)

import gymnasium as gym  # noqa: E402

_orig_make = gym.make


def _make_with_video(*args, **kwargs):
    kwargs.setdefault("render_mode", "rgb_array")
    env = _orig_make(*args, **kwargs)
    base = env.unwrapped
    frames = []
    orig_step, orig_close = base.step, base.close

    def step(action):
        res = orig_step(action)
        if len(frames) < video_length:
            img = base.render()
            if img is not None:
                frames.append(img.copy())
        return res

    def close():
        if frames:
            import imageio

            os.makedirs(video_folder, exist_ok=True)
            path = os.path.abspath(os.path.join(video_folder, f"{tag}.mp4"))
            imageio.mimsave(
                path, frames, fps=fps, codec="libx264", quality=7,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "main"],
            )
            print(f"[replay_video] wrote {len(frames)} frames -> {path}", flush=True)
        else:
            print("[replay_video] no frames captured", flush=True)
        return orig_close()

    base.step = step
    base.close = close
    print(f"[replay_video] capturing frames (max {video_length}) for {tag}", flush=True)
    return env


gym.make = _make_with_video

script = os.path.join(os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")), "scripts/tools/replay_demos.py")
sys.argv = [script] + out
runpy.run_path(script, run_name="__main__")
