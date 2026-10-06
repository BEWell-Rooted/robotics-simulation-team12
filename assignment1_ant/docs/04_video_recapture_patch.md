# 04 · 발표 영상 재녹화 패치 (Rails · Stairs 지형 이탈 해결)

작성: 2026-10-06 (발표 자료 세션) · 대상: 코드 세션 · 소요 예상: 패치 10분 + 녹화 4 clip × ~2분

## 0. 왜 다시 찍나

`docs/media/final_rails.mp4`, `final_stairs.mp4`(그리고 정도는 덜하지만 `final_grid`, `final_pyramids`)에서 로봇이
초반에는 지형을 잘 걷다가 **y 방향으로 밀려 지형 밖 border 평지로 나가서** 걷는다. 발표 14장(영상)은
"다양한 지형을 완주"가 메시지라서 지형 밖 장면이 섞이면 안 된다. 현재는 Rails를 빼고 Flat(mesh)를 넣었고
Rails·Stairs는 부록 A4(이탈 사례)로 내려 둔 상태다. 깨끗한 clip이 오면 14장으로 되돌린다.

### 원인 (코드 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| 평가 지형 크기 | `num_rows=20 × num_cols=4`, tile 8 m → **길이 160 m × 폭 32 m** | `eval_envs.py::make_eval_terrain` |
| 스폰 열 | `curriculum=False`, `max_init_terrain_level=0` → TerrainImporter가 env i를 `col = i // (num_envs/num_cols)`에 배정 → `--num_envs 1`이면 **env 0 = 0행 0열** | Isaac Lab `TerrainImporter._compute_env_origins_curriculum` |
| 0열 중심 | y = −(4−1)/2 × 8 = **−12 m**, 지형 가장자리(−16 m)까지 **4 m** | 계산 |
| 정책의 y 드리프트 | locked test에서 960 step 동안 약 27%가 32 m 폭을 벗어남 (`eval_task.py` off-terrain 판정) | `results/official_test.csv` |

즉 영상용 로봇은 항상 지형의 **가장 바깥 열, 가장자리에서 4 m** 떨어진 자리에서 출발한다. 조금만 왼쪽으로
밀리면 바로 border다. 평가 수치(`eval_round2.csv`, 100 env)는 네 열에 25개씩 고르게 깔리므로 이 문제와 무관하다 — **영상만의 문제**이고 결과 CSV는 다시 돌릴 필요 없다.

### 해결 방향 (둘 다 적용)

1. **넓은 평가 지형 task** `Isaac-Ant-Eval-<Name>-Team12-Wide-v0` — 같은 sub-terrain, `num_cols=12` (폭 96 m). 지형 seed·tile 파라미터는 동일.
2. **가운데 열 스폰** `--center_spawn` — `play_one_episode.py`에서 env 0의 origin을 가운데 열(0행, `num_cols//2`열)로 덮어쓴다. 12열이면 가장자리까지 **44 m**.

둘을 합치면 960 step 안에 지형 밖으로 나갈 가능성은 사실상 없다. 지형 파라미터(블록 폭·높이, 계단 높이, 레일 간격)는 바뀌지 않으므로 "학습에 없던 지형을 완주"라는 메시지도 그대로다.

---

## 1. 패치 A — `source/ant_rough/ant_rough/tasks/__init__.py`

`EVAL_SPECS` 루프 안, `Isaac-Ant-Eval-{_name}-Team12-v0` 등록 바로 아래에 추가.

```python
    # wide (num_cols=12, 96 m) version of the same terrain, spawn-centred by play_one_episode --center_spawn:
    # video-only task so the Ant cannot drift off the 32 m evaluation terrain within 960 steps
    def _make_team_wide_cfg(_kwargs=_kwargs):
        from .submission_env_cfg import FinalDeployEnvCfg

        return make_eval_cfg(FinalDeployEnvCfg, **{**_kwargs, "num_cols": 12})

    gym.register(
        id=f"Isaac-Ant-Eval-{_name}-Team12-Wide-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _make_team_wide_cfg, "rsl_rl_cfg_entry_point": f"{_AG}:AntV12PPORunnerCfg"},
    )
```

- `_kwargs`에 이미 `num_cols`가 있는 spec은 없으므로(`EVAL_SPECS` 확인) `{**_kwargs, "num_cols": 12}`로 덮어써도 안전.
- `make_eval_terrain`의 `num_cols` 인자가 그대로 `TerrainGeneratorCfg.num_cols`로 들어가므로 `eval_envs.py`는 **수정 없음**.
- 20 × 12 = 240 tile. `apply_eval_terrain`이 `gpu_max_rigid_patch_count`를 이미 올려 두었고 `--num_envs 1`이라 메모리 문제는 없을 것. 혹시 Stairs(trimesh 많음)에서 PhysX 경고가 나면 `num_cols=8`로 내려도 가장자리까지 28 m라 충분하다.

## 2. 패치 B — `scripts/rsl_rl/play_one_episode.py`

### B-1. 인자 추가 (기존 `--real-time` 아래)

```python
parser.add_argument(
    "--center_spawn",
    action="store_true",
    default=False,
    help="Spawn every env at the centre column of row 0 of the generated terrain (video capture only).",
)
```

### B-2. origin 덮어쓰기 — `env = gym.make(...)` 바로 다음, RecordVideo / RslRlVecEnvWrapper **이전**

```python
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.center_spawn:
        # TerrainImporter puts env 0 in column 0 (4 m from the edge of a 4-column terrain); move it to the centre.
        # Must happen before RslRlVecEnvWrapper, which performs the first reset.
        terrain = env.unwrapped.scene.terrain
        n_cols = terrain.cfg.terrain_generator.num_cols
        terrain.env_origins[:] = terrain.terrain_origins[0, n_cols // 2]
        print(f"[INFO] center_spawn: env_origins <- terrain_origins[0, {n_cols // 2}] = {terrain.env_origins[0].tolist()}")
```

- `InteractiveScene.env_origins`는 terrain이 있으면 `terrain.env_origins`를 그대로 돌려주므로, 리셋 때 `reset_root_state_uniform`이 이 값을 쓴다. in-place(`[:]`) 대입이라 참조가 끊기지 않는다.
- `RslRlVecEnvWrapper.__init__`이 첫 `reset()`을 호출하므로 반드시 wrapper 생성 전에 넣어야 한다. 위 위치(gym.make 직후)면 된다.
- 카메라는 `viewer.origin_type="asset_root"`로 로봇을 따라가므로 추가 수정 없음.

### B-3. (선택) 종료 시 위치 로그 — 영상 열어보지 않고도 이탈 여부 확인용

메인 루프가 끝난 뒤(기존 `Episode steps` print 근처)에 한 줄:

```python
    root = env.unwrapped.scene["robot"].data.root_pos_w[0].tolist()
    print(f"[INFO] Final root pos (x, y, z): {root[0]:.1f}, {root[1]:.1f}, {root[2]:.2f}")
```

Wide 지형이면 |y − y_center| < 48 이면 지형 안. 가운데 열 중심 y는 12열 기준 `(6 − 5.5) × 8 = +4 m`.

## 3. 패치 C — `scripts/capture_videos.sh`

### C-1. `capture()`에 추가 인자 훅 (play 명령 한 줄만 변경)

```bash
    ~/IsaacLab_RS/isaaclab.sh -p scripts/rsl_rl/play_one_episode.py --task "$task" --seed "$seed" --num_envs 1 \
      --headless --checkpoint "$dir/model.pt" --video --video_length 960 ${PLAY_EXTRA:-} < /dev/null > "$dir/play.log" 2>&1
```

(`${PLAY_EXTRA:-}`는 의도적으로 따옴표 없이 — 비어 있으면 아무것도 안 들어가고, 기존 호출은 그대로 동작.)

### C-2. `final_wide` 모드 — 기존 `if [ "${1:-}" = final ]` 블록 **앞**에 추가

```bash
if [ "${1:-}" = final_wide ]; then
  # Wide (96 m) terrains + centre-column spawn so the Ant stays on the terrain for all 960 steps (slides 14 / A4).
  F=checkpoints/final/model_2999.pt
  export PLAY_EXTRA="--center_spawn"
  capture final_rails_wide    Isaac-Ant-Eval-Rails-Team12-Wide-v0    "$F" 24
  capture final_stairs_wide   Isaac-Ant-Eval-Stairs-Team12-Wide-v0   "$F" 24
  capture final_grid_wide     Isaac-Ant-Eval-Grid-Team12-Wide-v0     "$F" 24
  capture final_pyramids_wide Isaac-Ant-Eval-Pyramids-Team12-Wide-v0 "$F" 24
  exit 0
fi
```

- 파일 이름을 `_wide`로 분리해 기존 `final_*.mp4`를 덮어쓰지 않는다 (A4 "이탈 사례"로 쓸 수도 있으니 남겨 둠).
- seed 24 고정은 기존 clip과 같은 조건. 24에서 넘어지면(`steps < 960`) `25 26`을 뒤에 더 붙여 다시 돌리면 된다 — `_fail` 접미사가 아니므로 첫 seed 결과를 그대로 저장한다는 점만 유의.

## 4. 실행

```bash
conda activate lerobot-arena && cd ~/robotics-simulation-team12/assignment1_ant
# task 등록 확인
python -c "import ant_rough, gymnasium as gym; print([k for k in gym.registry if 'Wide' in k][:3])"
# 녹화
bash scripts/capture_videos.sh final_wide
# 확인: 네 줄 모두 steps=960, [INFO] Final root pos 의 y가 -44~+52 안이면 지형 안
grep -a "CAPTURE\|Final root" logs/video_ckpt/final_*_wide/play.log
tail -4 docs/media/videos.csv
```

## 5. 돌려줄 것

- `docs/media/final_rails_wide.mp4`, `final_stairs_wide.mp4`, `final_grid_wide.mp4`, `final_pyramids_wide.mp4`
- `docs/media/videos.csv`의 새 4줄 (reward · steps — 슬라이드 캡션 숫자용)
- (있다면) `[INFO] Final root pos` 네 줄

받으면 발표 자료 쪽에서 1280 폭 / crf 30으로 압축해 14장을 **Plane · Grid · Rails · Stairs**(또는 Pyramids) 구성으로 되돌리고, 부록 A4는 Flat 뒤집힘 + 기존 이탈 clip(원인 설명용)으로 정리한다.

## 6. 하지 않는 것

- `results/eval_round2.csv`, `official_test.csv` 재평가 — 100 env가 네 열에 고르게 깔리므로 수치는 이 문제의 영향을 받지 않는다. 15장 한계의 "지형 폭 32 m · 이탈 27%"도 그대로 유효(그건 수치 평가의 한계, 영상은 별개).
- `eval_envs.py`, `submission_env_cfg.py`, checkpoint — 손대지 않는다. 제출 task `Isaac-Ant-Team12-v0`와 무관.
