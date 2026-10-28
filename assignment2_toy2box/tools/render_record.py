"""play_record.py 가 남긴 매 step 장면 상태를 다시 세워 rollout 영상·장면 띠(contact sheet)를 만든다.

정책을 다시 돌리지 않으므로, 평가 머신(A6000, 렌더링 불가)에서 기록한 바로 그 rollout 을
렌더링되는 머신(Dell, DGX Spark 등)에서 그대로 볼 수 있다. 물리는 진행하지 않고
step 마다 env.reset_to(기록 상태) → 렌더만 한다. (gx10-2 확인: 한 번 렌더로 그 step 상태가 나옴, 프레임 밀림 없음)

사용 (IsaacLab_RS 폴더에서):
  ./isaaclab.sh -p <A2>/tools/render_record.py --record_file <기록.hdf5> --rollouts fail \
      --video_folder ./videos/baseline_fail --max_rollouts 12 --headless
  --rollouts : all | fail | success | 0 3 7 (번호 목록)
  --stride N : N step 마다 한 프레임 (영상 fps = 20 / N)
  --sheet    : 처음·단계 이정표·끝 8장을 2×4 로 붙인 jpg 도 저장 (분류 확인용)
  --no_video : 영상 없이 장면 띠만
  --view close | wide : 카메라 프리셋. close(기본) = 로봇 오른쪽 앞 위에서 손·트럭·상자를 크게,
                        wide = 태스크 cfg 의 viewer (eye 1.5 -1.0 1.5, lookat 0.5 0 0, 장면 전체)
  --eye x y z --lookat x y z : 카메라 직접 지정 (프리셋보다 우선, 월드 좌표 m)
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

# 카메라 프리셋 (월드 좌표, env 0). 로봇 base (-0.6, 0, -1.05), 트럭 x -0.15~0.20 · y -0.30~-0.15 · z -0.65,
# 상자 x 0.25~0.35 · y 0~0.10. close 는 gx10-2 에서 후보 8개를 렌더해 비교하고 고른 값 (docs/spark_setup_1028.md)
VIEWS = {
    "close": ((0.55, -1.2, 0.5), (0.07, -0.1, -0.42)),
    "wide": (None, None),
}

parser = argparse.ArgumentParser()
parser.add_argument("--record_file", required=True)
parser.add_argument("--rollouts", nargs="+", default=["fail"])
parser.add_argument("--max_rollouts", type=int, default=20)
parser.add_argument("--video_folder", default="./videos/rollout_records")
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--sheet", action="store_true")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--view", choices=list(VIEWS), default="close")
parser.add_argument("--eye", type=float, nargs=3, default=None)
parser.add_argument("--lookat", type=float, nargs=3, default=None)
parser.add_argument("--resolution", type=int, nargs=2, default=[960, 540])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_failures as CF  # noqa: E402


def put_text(img, lines):
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        y = 6
        for ln in lines:
            d.rectangle([4, y - 2, 8 + 7 * len(ln), y + 13], fill=(0, 0, 0))
            d.text((7, y), ln, fill=(255, 255, 255))
            y += 17
        return np.asarray(im)
    except Exception:
        return img


def main():
    f = h5py.File(args.record_file, "r")
    task = f.attrs["task"]
    R = f["rollouts"]
    names = sorted(R, key=lambda s: int(s.split("_")[-1]))
    trajs = {t["id"]: t for t in CF.load_trajs(args.record_file)}
    rows = {k: CF.classify(trajs[k]) for k in names}
    sel = []
    if args.rollouts == ["all"]:
        sel = names
    elif args.rollouts == ["fail"]:
        sel = [k for k in names if not rows[k]["success"]]
    elif args.rollouts == ["success"]:
        sel = [k for k in names if rows[k]["success"]]
    else:
        sel = [f"rollout_{int(x)}" for x in args.rollouts]
    sel = sel[: args.max_rollouts]
    if not sel:
        raise SystemExit(f"--rollouts {' '.join(args.rollouts)} 에 해당하는 rollout 이 없다 ({len(names)}개 중)")
    missing = [k for k in sel if k not in R]
    if missing:
        raise SystemExit(f"기록에 없는 rollout: {missing} (있는 것: {names[0]} … {names[-1]})")
    if "states" not in R[sel[0]]:
        raise SystemExit("기록에 states/ 가 없다 (--no_states 로 기록됨)")

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    for k in list(vars(env_cfg.terminations).keys()):  # 종료 조건 모두 끔
        setattr(env_cfg.terminations, k, None)
    env_cfg.recorders = None
    env_cfg.viewer.resolution = tuple(args.resolution)
    eye, lookat = VIEWS[args.view]
    eye = args.eye if args.eye is not None else eye
    lookat = args.lookat if args.lookat is not None else lookat
    if eye is not None:
        env_cfg.viewer.eye = tuple(eye)
    if lookat is not None:
        env_cfg.viewer.lookat = tuple(lookat)
    print(f"[render_record] camera eye={tuple(env_cfg.viewer.eye)} lookat={tuple(env_cfg.viewer.lookat)}", flush=True)
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    env.reset()
    dev = env.device
    os.makedirs(args.video_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.record_file))[0]

    def load_states(g):
        # rollout 하나의 states/ 를 한 번에 읽는다 (프레임마다 gzip 청크를 다시 읽지 않게)
        return {
            g1: {g2: {g3: g["states"][g1][g2][g3][:] for g3 in g["states"][g1][g2]} for g2 in g["states"][g1]}
            for g1 in g["states"]
        }

    def state_at(S, t):
        return {
            g1: {g2: {g3: torch.as_tensor(a[t][None], device=dev) for g3, a in d2.items()} for g2, d2 in d1.items()}
            for g1, d1 in S.items()
        }

    def frame(S, t):
        env.reset_to(state_at(S, t), None, is_relative=True)
        env.sim.render()
        return env.render()

    # 첫 렌더는 검게 나오므로 몇 번 데운다
    S0 = load_states(R[sel[0]])
    for _ in range(8):
        frame(S0, 0)

    import time

    import imageio

    for k in sel:
        g = R[k]
        S = load_states(g)
        row = rows[k]
        T = S["articulation"]["robot"]["joint_position"].shape[0]
        label = f"{k}  {row['category']}/{row['subtype']}  end={row['end_reason']}"
        tag = f"{stem}__{k}__{row['category']}_{row['subtype']}".rstrip("_")
        if not args.no_video:
            frames = []
            t0 = time.time()
            for t in range(0, T, args.stride):
                img = frame(S, t)
                if img is None:
                    continue
                frames.append(put_text(img.copy(), [label, f"step {t}/{T - 1}"]))
            path = os.path.join(args.video_folder, tag + ".mp4")
            imageio.mimsave(
                path, frames, fps=max(1, 20 // args.stride), codec="libx264", quality=7,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "3.1", "-movflags", "+faststart"],
                macro_block_size=2,  # 기본 16 이면 540 → 544 로 늘려 찌그러진다
            )
            print(f"[render_record] {k}: {len(frames)} frames -> {path} ({time.time() - t0:.1f} s)", flush=True)
        if args.sheet:
            # 단계 이정표(있는 것만) + 처음·끝, 모자라면 고르게 채워 8칸
            keys = ["t_near", "t_first_close", "t_grasp", "t_lift", "t_over", "t_success"]
            marks = {0: "start", T - 1: "end"}
            for x in keys:
                if row[x] >= 0:
                    marks.setdefault(int(row[x]), x[2:])
            for t in np.linspace(0, T - 1, 8).astype(int):
                if len(marks) >= 8:
                    break
                if all(abs(int(t) - m) > T // 16 for m in marks):
                    marks[int(t)] = ""
            imgs = []
            for t in sorted(marks)[:8]:
                frame(S, int(t))  # 큰 장면 변화 뒤 TAA 잔상을 줄이려 한 번 더
                img = frame(S, int(t))
                if img is not None:
                    imgs.append(put_text(np.ascontiguousarray(img[::2, ::2, :3]), [f"t={t} {marks[t]}".strip()]))
            if imgs:
                h = min(i.shape[0] for i in imgs)
                w = min(i.shape[1] for i in imgs)
                imgs = [i[:h, :w] for i in imgs]
                imgs += [np.zeros_like(imgs[0])] * ((-len(imgs)) % 4)
                rows4 = [np.concatenate(imgs[j:j + 4], axis=1) for j in range(0, len(imgs), 4)]
                bar = put_text(np.zeros((22, rows4[0].shape[1], 3), np.uint8), [label])
                sheet = np.concatenate([bar] + rows4, axis=0)
                p = os.path.join(args.video_folder, tag + "_sheet.jpg")
                imageio.imwrite(p, sheet, quality=90)
                print(f"[render_record] {k}: sheet -> {p}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
