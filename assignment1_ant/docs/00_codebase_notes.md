# 00. 코드베이스 노트 — Isaac-Ant-v0 (IsaacLab_RS main @ e83a5d2, VERSION 2.3.0)

> 브리프 Phase C의 가설을 실제 코드로 검증한 결과. ✔ = 코드로 확인, ▢ = 아직 미확인(런타임 확인 필요)

## 1. 핵심 정정 — `Isaac-Ant-v0`는 **manager-based** 환경이다

| 브리프 가설 | 실제 |
|---|---|
| Direct 워크플로 (`direct/ant/`) | ✔ **Manager-based** — `manager_based/classic/ant/ant_env_cfg.py`, entry point `isaaclab.envs:ManagerBasedRLEnv` |
| 관측 36차원 | ✔ **60차원** — baseline 학습 로그 `Active Observation Terms in Group: 'policy' (shape: (60,))`, Actor MLP `in_features=60` |
| — | `direct/ant/`는 별개 태스크 `Isaac-Ant-Direct-v0` (obs 36, experiment `ant_direct`) |

- rsl_rl cfg: `manager_based/classic/ant/agents/rsl_rl_ppo_cfg.py:AntPPORunnerCfg`, `experiment_name = "ant"` → 로그 `logs/rsl_rl/ant/` (참고 자료와 일치)
- mdp 함수는 `isaaclab_tasks.manager_based.classic.humanoid.mdp` (humanoid와 공유) + `isaaclab.envs.mdp`

**설계상 의미:** manager-based이므로 지형 교체(`scene.terrain`), 마찰 랜덤화(`EventTerm` + `mdp.randomize_rigid_body_material`),
커리큘럼(`CurriculumTermCfg`), 관측 그룹(`policy` / `critic`)을 모두 cfg 상속만으로 붙일 수 있다. Direct env 우회 작업이 필요 없음.

## 2. 관측 (policy 그룹, concat)

| 항 | 함수 | 차원 | 비고 |
|---|---|---|---|
| base_height | `mdp.base_pos_z` | 1 | ✔ **월드 절대 z** (`root_pos_w[:, 2]`) — 높이 있는 지형에서 의미가 바뀜 |
| base_lin_vel | `mdp.base_lin_vel` | 3 | body frame |
| base_ang_vel | `mdp.base_ang_vel` | 3 | body frame |
| base_yaw_roll | `mdp.base_yaw_roll` | 2 | |
| base_angle_to_target | target (1000,0,0) | 1 | |
| base_up_proj | | 1 | |
| base_heading_proj | target (1000,0,0) | 1 | |
| joint_pos_norm | `joint_pos_limit_normalized` | 8 | |
| joint_vel_rel | scale 0.2 | 8 | |
| feet_body_forces | `body_incoming_wrench` ×4 발, scale 0.1 | 24 | 발 4개 × 6D wrench (body frame). 지면 접촉·마찰 정보가 간접적으로 들어옴 |
| actions | `last_action` | 8 | |
| **합계** | | **60** | `enable_corruption=False` (관측 노이즈 없음) |

## 3. 행동
- `JointEffortActionCfg(joint_names=[".*"], scale=7.5)` → 8차원 토크

## 4. 보상 (가중치)
| 항 | 가중치 | 비고 |
|---|---|---|
| progress (target 1000,0,0) | 1.0 | ✔ potential 차분, `to_target_pos[:, 2] = 0` → **수평 거리 기반**, 지형 높이에 안전 |
| alive | 0.5 | |
| upright (threshold 0.93) | 0.1 | |
| move_to_target (threshold 0.8) | 0.5 | |
| action_l2 | -0.005 | |
| energy (gear 15) | -0.05 | |
| joint_pos_limits (0.99) | -0.1 | |

Direct 버전에 있는 death_cost(-2.0)는 manager-based 버전엔 **없음**.

## 5. 종료
- `time_out` — episode 16.0 s (dt 1/120, decimation 2 → 60 Hz → **960 step**, `--video_length 960`과 일치)
- `torso_height`: `root_height_below_minimum(minimum_height=0.31)` — ✔ **월드 절대 z**. 함수 docstring에도 "currently only supported for flat terrains" 명시
  → 높이 있는 지형에서는 (a) 블록 위에서 넘어져도 종료 안 됨, (b) 지면이 z<0이면 정상 보행 중에도 종료될 수 있음

## 6. 지형·물리
- `scene.terrain = TerrainImporterCfg(terrain_type="plane", static/dynamic friction 1.0, restitution 0, combine "average")`
- `sim.physics_material`도 1.0/1.0/0.0
- `env_spacing=5.0`, `num_envs=4096`, `clone_in_fabric=True`
- events: `reset_root_state_uniform`(pose/velocity 범위 없음), `reset_joints_by_offset`(pos ±0.2, vel ±0.1)

## 7. 조교 평가 리스크 (D0 호환성에 대한 영향)
- 조교의 unseen 환경은 이 manager-based cfg에서 **terrain과 physics_material만 바꾼 형태**일 가능성이 높다 (과제 안내: "그 외 조건은 기존 환경과 동일").
- 그렇다면 조교 환경에서 정책은 **60차원 obs, 절대 z `base_height`, 절대 z 종료(0.31)** 를 그대로 받는다.
  - 우리 학습 환경에서 `base_height`를 지형 상대값으로 바꾸면 조교 환경과 분포가 어긋남 → 기본안은 **절대 z 유지**가 안전.
  - 종료 조건도 조교 쪽은 절대 z일 것 → 학습 중에도 이 조건을 유지할지 / 지형 원점 높이를 0 근처로 맞출지 설계 필요.
- 체크포인트 로드 호환: actor 입력 60 / 출력 8 / [400,200,100] elu / obs norm off 유지.

## 8. 미확인 (▢)
- [ ] `./isaaclab.sh --new` 템플릿 생성물 구조 (`tools/template/` 존재 ✔, 실행은 설치 후)

## 9. 체크포인트 로드 경로 (호환성의 근거) ✔

- `play.py`: `@hydra_task_config(args_cli.task, ...)` → **`--task`로 준 태스크의 `rsl_rl_cfg_entry_point`로 네트워크를 만든 뒤** `runner.load(resume_path)`.
  run 폴더의 `params/`는 **읽지 않는다.**
- 즉 조교가 자기 unseen 태스크(= Isaac-Ant-v0 기반, `AntPPORunnerCfg`)로 우리 `.pt`를 로드하면
  네트워크는 **[400,200,100] elu, obs norm off, actor/critic 입력 = policy obs(60)** 로 만들어진다.
- `runner.load`는 actor·critic state dict(와 optimizer)를 함께 로드 → **critic 입력 차원이 다르면 로드 실패** 가능성이 높다.
- rsl-rl 버전: `rsl-rl-lib==3.0.1` (`isaaclab_rl/setup.py`). `RslRlBaseRunnerCfg.obs_groups`로
  `{"policy": ["policy"], "critic": ["policy", "privileged"]}` 같은 **비대칭 critic 구성 지원** ✔

### → D2 V4(비대칭 critic) 주의
학습은 가능하지만, 제출 체크포인트는 critic이 60차원 입력이어야 조교 cfg로 로드된다.
대안: (a) V4는 비교 실험으로만 쓰고 제출은 대칭 critic 모델, (b) 학습 후 critic을 60차원 입력으로 재적합(value regression)해 교체한 호환 체크포인트를 만든다.
✔ `ActorCritic.load_state_dict(strict=True)` → 키·shape가 하나라도 다르면 로드 실패
(critic 입력 차원, hidden dims, obs normalizer 버퍼 유무 모두 포함). 이어서 optimizer state도 로드한다.
**결론: 제출 체크포인트는 actor·critic 모두 baseline 구조(입력 60, [400,200,100], elu, normalizer 없음)와 state dict가 완전히 같아야 한다.**

## 10. 지형 생성기 ✔

### `TerrainGeneratorCfg` (`isaaclab/terrains/terrain_generator_cfg.py`)
주요 필드: `size`(타일 크기), `num_rows`(난이도 축), `num_cols`(지형 종류 축), `curriculum`(True면 row = difficulty 순서),
`difficulty_range=(0,1)`, `border_width`, `horizontal_scale=0.1`, `vertical_scale=0.005`, `slope_threshold=0.75`,
`sub_terrains: dict[str, SubTerrainBaseCfg]`(각자 `proportion`), `seed`, `use_cache`, `color_scheme`.

### `MeshRandomGridTerrainCfg` — 안내 이미지의 "높이 제각각 정사각 블록"과 동일
- 필드: `grid_width`(블록 한 변), `grid_height_range`(난이도 0→1 사이 선형 보간), `platform_width=1.0`, `holes=False`
- 블록 높이: `grid_height = lo + difficulty*(hi-lo)` 계산 후 각 블록 윗면을 **U(-grid_height, +grid_height)** 로 이동
- 타일 중앙에 `platform_width` 정사각 플랫폼(윗면 z = +grid_height)
- **terrain origin z = +grid_height** (플랫폼 윗면) → 스폰 높이는 origin을 따라 자동 보정
- 주의: 타일 크기는 정사각이어야 함 (`size[0] == size[1]`)

### env origin 배치 (`terrain_importer.py`)
- `terrain_type="generator"`이면 `configure_env_origins(terrain_generator.terrain_origins)` →
  `_compute_env_origins_curriculum`으로 각 env를 (row=level, col=type) 타일 origin에 배치. `max_init_terrain_level`로 초기 최대 레벨 제한.
- manager-based의 `reset_root_state_uniform`은 `default_root_state + env_origins`로 리셋 → **origin z가 반영되어 스폰 높이 보정은 자동**.

### 절대 z 종료(0.31)와 블록 높이의 관계
- Ant 기본 스폰 z = 0.5 (ANT_CFG, ▢ 확인). 블록 윗면이 -h까지 내려가므로, 낮은 블록 위에서 몸통 z가 (정상 자세 높이 − h)가 된다.
  h가 크면 정상 보행 중에도 0.31 아래로 내려가 **거짓 종료**가 생길 수 있음 → 학습 지형의 h 범위와 조교 환경의 h를 모두 고려해야 함.

## 11. 조교 평가 지표로 추정되는 것 — `play_one_episode.py` ✔
- `play.py`와의 차이: 뷰어가 로봇을 따라감(`origin_type="asset_root"`), **env별 첫 에피소드**의 보상 합과 스텝 수를 누적,
  모든 env가 첫 에피소드를 마치거나 `max_episode_length`(960)에 도달하면 종료.
- 출력: `num_envs == 1` → `[RESULT] Episode reward total`, `[RESULT] Episode steps` / 여러 env → 평균·모집단 표준편차.
- 즉 조교가 공개할 "리워드"는 **조교 환경의 보상 함수(= 기본 Isaac-Ant-v0 보상 항일 가능성 높음)로 계산한 첫 에피소드 누적 보상**.
  → 학습 보상을 바꾸더라도 **평가는 원래 보상으로** 해야 한다. 우리 eval_suite도 원래 7개 보상 항으로 채점하고,
  에피소드 길이(조기 종료)를 함께 본다. 넘어지면 alive·progress 누적이 끊기므로 "안 넘어지고 오래 전진"이 곧 점수.
