# 트랙 A 기획안 — Mimic 상자 구간 재타깃팅 (2026-10-29, 담당 A · 교차 점검 · 독립 점검 반영판)

> 기준: 진단 `docs/diagnosis_1028.md`(4·7·8절), 서베이 `docs/survey_1028.md`(§2.1, §6), 교차 점검 통합(§1 #9·#10·#16·#17, §2, §4).
> **팔 정의·데이터 게이트·최소 예측·판정 수치는 `docs/prereg_1029.md` 가 유일한 기준이다** (§3 팔·데이터 게이트, §5.2 확증, §5.3 귀속, §6 해석표, §7 큐). 숫자가 다르면 prereg 가 우선한다. 이 문서에는 트랙 A 의 가설 근거·구현·스모크 결과·명령만 둔다.
> 도구: `tools/generate_yawhold.py`(모드 `--translate`·`--rotworld`·`--yawhold`), `tools/plan_A_stratify.py`, `tools/plan_A_smoke_check.py`, `tools/plan_A_demo_to_record.py`. 스모크 결과 `results/plan_A/`.
> 범위: 가설·설계·구현·소규모 검증. 전체 규모(Mimic → 2000 epoch → 100회 평가)는 아직 돌리지 않았다.

## 0. 요약

- **본 처치 A-T (확증 · 1차 원인):** 상자 구간을 "상자 위치만 따라가게"(TRANSLATE) 생성한 시범 약 650개 → baseline 시작 상태 분포에 맞춰 **층화 추출한 500개**.
- **귀속 팔 A-R:** 상자 구간에서 회전만 A-T 와 같게(원본 world 회전), 위치는 SE(3) 그대로. A-T 와 같게 시범 약 650개(seed 1–5 × 성공 130) → 같은 도구·seed 로 **층화 추출한 500개**. A-T 와 2×2 의 한 칸이라 A-T↔A-R 직접 비교가 "위치·우회 몫" 을 가른다. (대안 A-Y 는 prereg §3 에서 뺐다 — 2절)
- 스모크(각 30개)에서 손목 회전 상태는 A-T 13.0° · A-R 13.4° (baseline 47.5°)로 같고, 위치 쪽(30 step 이동·경로·상자 밂)은 A-R 이 baseline 과 같다(3.2 표).

## 1. 가설

**H-A.** 들기 이후 실패(49/68, 72%)의 주원인은 Mimic 상자 구간의 SE(3) 재타깃팅이 만든 불필요한 운반 동작이다. 이 재타깃팅은 원본 시범의 손끝–상자 상대 자세를 새 상자 자세로 옮긴다(`object_ref="box"`, `transform_source_data_segment_using_object_pose`). 그래서 운반 중 손목을 "새 상자 yaw − 원본 상자 yaw" 만큼 돌리고, 손끝 경로도 상자 yaw 차만큼 휘어 길어진다. 상자 판정(`object_a_is_into_b`)은 트럭·상자의 **위치만** 보므로 이 회전·우회는 과업에 필요 없다. BC 정책은 이 운반 습관을 배워 상자에 부딪히거나 벽 앞에서 놓는다.

**근거 (측정값)**
- 정책 rollout(baseline 200): 들기 직후 회전 명령 성공 0.10 vs 들기 이후 실패 0.49, 운반 중 손목 yaw 순변화 성공 32–35° vs 실패 58–63°, 상자 5 cm 이상 밂 실패 61% vs 성공 10%.
- 들린 rollout 174개 로지스틱(표준화 계수 z): 상자 접근 높이 +3.5 · 들기 뒤 30 step 손목 yaw −2.1 · 위치 점프 −1.2 (yaw↔위치 점프 상관 0.68).
- 학습 데이터: 들린 동안 손목 yaw 변화 사람 2.4° vs Mimic 38°. Mimic 생성 실패 630건의 76% 가 상자 구간.
- **생성기 원본 선택 정정 (스모크 `--src_log` 로 확인):** 원본 시범은 subtask 1(트럭 기준, nearest_neighbor_object)에서 한 번만 고르고 상자 구간에도 같은 원본을 쓴다. 과제 cfg 가 `generation_select_src_per_subtask=True`(`agibot_place_toy2box_mimic_env_cfg.py:32`) 여도 `generation_select_src_per_arm=False`(기본값, `isaaclab/envs/mimic_env_cfg.py:67`) 라 이미 고른 원본을 다시 쓰기 때문이다(`data_generator.py:381–384` 의 `need_source_demo_selection`). → 진단 §4.5 와 서베이 §2.1(CP-Gen·MinInter 행)의 "구간별 원본 선택" 해석은 틀렸다(두 문서에 10/29 정정을 넣었다).

**반증:** 데이터 게이트(prereg §3)를 통과했는데 A-T 가 prereg §5.2 ② 기각에 해당하면 H-A 를 기각한다(최소 예측은 prereg §3, 5절).

## 2. 설계

| | 대조 baseline (있음) | **A-T 본 처치** | **A-R 귀속** |
| --- | --- | --- | --- |
| 상자 구간 회전 | SE(3) (상자 yaw 차 포함) | 원본 world | 원본 world |
| 상자 구간 위치 | SE(3) | 평행이동만 (+상자 위치 차) | SE(3) |
| 생성 → 학습 데이터 | seed 1–5 × 100 = 500 | seed 1–5 × 130 ≈ 650 → **층화 500** | seed 1–5 × 130 ≈ 650 → **층화 500** (A-T 와 같은 도구·seed) |
| 원본 20 · 선택 · 보간 15 · noise 0.01 · 그리퍼 · 트럭 구간(subtask 1) | 조교 cfg | 같음 | 같음 |
| 학습 · 평가 | `bc_rnn_low_dim.json` 2000 epoch 마지막 ckpt, seed 101·111 / A6000 평가 seed 42 + 43 × 100회 = 400쌍 | 같음 | 같음 |
| 등급 (prereg §3) | — | 확증 · baseline 과 비교 | 귀속 · A-T 와 직접 비교 |

- **A-T 가 한 요인인 이유:** 생성기의 상자 구간 변환 함수 하나만 바꾼다(`--translate 1`). 성공 판정이 상자 위치만 본다는 이 과업의 불변성에서 나온 처치다. 선례: DexMimicGen 의 Transform/Replay 비교, Isaac Lab `SubTaskConstraintCoordinationScheme.TRANSLATE`(양팔 조율용).
- **층화 추출 (교차 점검 #9, A-T · A-R 공통):** 생성률이 baseline(44%)과 다르면(A-T 89%, A-R 58%) 성공 필터를 통과하는 장면 분포가 바뀐다. baseline 생성률은 시작 거리와 상관이 있다(< 0.30 m 28% · 0.30–0.35 32% · 0.35–0.40 45% · ≥ 0.40 53%). 그래서 두 팔 모두 약 650개에서 baseline 500 의 (시작 트럭–상자 거리 4구간 × 상대 yaw 4구간) 히스토그램에 맞춰 500개를 결정적으로 뽑는다(`plan_A_stratify.py --ref kgh_mimic_500 --n 500 --seed 0`, 3.3절). 같은 도구·seed 라 A-T↔A-R 비교에는 위치 규칙 한 요인만 남는다. 시작 거리 ≥ 0.35 m rollout 만의 Δ 는 prereg §5.2 하위 분석(보고)이고 §6 H-A2–4 의 확인 조건이다.
- **A-R 로 바꾼 이유 (교차 점검 #10):** A-Y 는 손목 yaw 상태만 없애고 회전 명령은 그대로 두며, 상자 밂·옆 이동·필터라는 새 요인을 넣는다(스모크: box_push 26%, xy30 0.288, roll·pitch 명령 0.042 → 0.072). 그래서 A-Y 가 null 이어도 "회전이 원인이 아님" 과 "A-Y 가 넣은 손해" 를 가를 수 없다. A-R 은 회전 규칙이 A-T 와 같으므로, A-T↔A-R 차이는 위치 규칙(평행이동 vs SE(3))의 몫이다.
- **A-Y 는 쓰지 않는다 (prereg §3):** 스모크에서 이미 box_push 26% 로 이전 판의 대안 게이트를 넘었다. A-R 을 못 돌리면 결론 범위를 "상자 구간 재타깃팅(회전 + 우회)" 까지로 줄인다(prereg §6 H-A4). A-Y 스모크 값(3.2절)은 이 결정의 근거로만 남긴다.

## 3. 구현과 스모크 검증

### 3.1 `tools/generate_yawhold.py`
- 이 절의 subtask 번호와 로그 문자열은 코드 인덱스(0 = 트럭 구간, 1 = 상자 구간)다. 다른 절과 prereg 는 1부터 센다(subtask 1 = 트럭 구간).
- `generate_seed.py` 와 같은 runpy 래퍼. `gym.make` 훅 안(AppLauncher 뒤)에서 `isaaclab_mimic.datagen.data_generator` 를 import 해 감싼다. IsaacLab_RS 추적 파일은 고치지 않는다.
  - `--translate 1` (A-T): subtask 1(상자 구간)에서 모듈 함수 `transform_source_data_segment_using_object_pose` 대신 위치 += (상자_새 − 상자_원본) 위치, 회전 = 원본 world.
  - `--rotworld 1` (A-R): 같은 자리에서 SE(3) 결과를 계산한 뒤 회전만 원본 world 회전으로 덮는다.
  - `--yawhold 1` (A-Y, 스모크만 — 본 실험에 쓰지 않음): `merge_eef_subtask_trajectory` 에서 subtask 1 waypoint 회전을 R(t)·R(0)ᵀ·R_last 로.
  - 모드는 하나만. `--datagen_seed N`, `--src_log FILE`(원본 선택 jsonl; subtask 0 선택만 찍힌다).
- 실행 로그에 `subtask 1: object_ref=box interp=15 noise=0.01 interpolate_from_last_target_pose=True` 와 적용된 모드를 찍는다.

### 3.2 스모크 결과 (A6000, `--num_envs 50`, `results/plan_A/gate_table_1029.md`)

수동성: `--yawhold 0`(모드 없음) seed 1 의 demo 20/20 이 baseline seed 1 샤드와 시작 상태·길이·actions 까지 bit 단위로 같다. 진행 순서(0/1 → 2/5 → … → 12/35)도 같다.

| 지표 (중앙값; `carry_metrics.py` · classify v3) | 사람(팀원) | baseline s1 | baseline 500 | **A-T s1** | **A-R s1** | A-Y s1 |
| --- | --- | --- | --- | --- | --- | --- |
| 생성 성공 | — | 100/254 (39%) | 500/1130 (44%) | **31/35 (89%)** | 30/52 (58%) | 61/114 (54%, s1+s2) |
| 손목 yaw 파지→상자 위 (°) | 2.2 | 43.1 | 47.5 | **13.0** | 13.4 | 3.5 |
| carry_yaw_net (°) | 2.4 | 38.6 | 38.4 | **7.2** | 9.9 | 2.8 |
| yaw30 (°) | 0.4 | 38.7 | 38.0 | **8.6** | 12.1 | 0.9 |
| xy30 (m) | 0.023 | 0.186 | 0.212 | **0.089** | 0.299 | 0.288 |
| approach_z 중앙 / p10 (m) | 0.243 / 0.214 | 0.245 / 0.171 | 0.243 / 0.109 | **0.250 / 0.212** | 0.249 / 0.044 | 0.252 / 0.095 |
| box_push (상자 5 cm 이상) | 0% | 8% | 16% | **6%** | 17% | 26% |
| rot60 (행동) | 0.028 | 0.177 | 0.209 | **0.063** | 0.263 | 0.195 |
| carry_pos_cmd (행동) | 0.038 | 0.130 | 0.137 | **0.060** | 0.155 | 0.140 |
| 파지→상자 위 손끝 경로 (m) | 0.76 | 1.18 | 1.22 | **0.99** | 1.29 | 1.28 |
| 시작 < 0.30 m / 상대 yaw > 90° | 5% / 50% | 5% / 48% | 9% / 47% | 6% / 55% | 3% / 47% | 3% / 55% |

- **A-R 은 설계대로다:** 회전 상태는 A-T 와 같고(13.4° vs 13.0°, carry_yaw_net 9.9 vs 7.2), 위치 쪽은 baseline 과 같거나 더 크다(xy30 0.299, 경로 1.29 m, box_push 17% ≈ baseline 500 의 16%). 행동 rot60 이 0.263 으로 큰 것은, SE(3) 로 휘어진 위치 경로를 원본 world 손목 방향으로 따라가면서 생기는 추적 오차로 본다(조작 확인 지표일 뿐이다. 팔 사이 기전 비교는 상태 매개 지표로만 한다 — prereg §4).
- **A-T 가 회전도 줄이는 이유:** 원본은 트럭 자세 최근접으로 고르므로 남는 회전 "원본 트럭 yaw − 새 트럭 yaw" 가 작다(13°).
- A-Y 의 한계(교차 점검 #10 근거, prereg §3 에서 A-Y 를 뺀 근거): 손목 yaw 상태는 사람 수준(3.5°)이지만 roll·pitch 명령이 늘고(0.042 → 0.072) 큰 회전 명령이 있는 demo 비율이 그대로(48%)이며 상자 밂이 26%.
- 생성 실패 분류(v3): baseline s1 154건 중 운반 90 · 배치 29 / A-T s1 4건(운반 2) / A-R s1 22건(운반 14, 상자 앞 놓음 9) / A-Y s1 27건(운반 17).
- 시각 확인(gx10-2 상태 재생, `results/plan_A/render/`): A-T·A-Y 가 파지 때 손목 방향 그대로 들어 상자에 넣는 것을 확인. 중복 검사: 모든 데이터셋 demo 마다 시작 상태·actions 가 다름.

### 3.3 `tools/plan_A_stratify.py` (층화 추출)
- 구간: 거리 [0, 0.30) [0.30, 0.35) [0.35, 0.40) [0.40, ∞) × 상대 yaw [0, 45) [45, 90) [90, 135) [135, 180]. 목표 = baseline 칸 비율 × 500 (최대 나머지 반올림). 칸 안은 demo 이름 정렬 뒤 `numpy.random.default_rng(seed)` 로 비복원 추출. 칸이 모자라면 같은 거리 구간의 가까운 yaw 칸 → 이웃 거리 구간 순서로 메운다(생성률과 상관된 축이 거리이므로 거리 주변분포를 먼저 맞춤).
- 출력 hdf5 의 `data.attrs['stratify']` 에 구간·baseline/묶음 히스토그램·목표·선택 demo 목록·seed 를 남긴다.
- 스모크 확인(`results/plan_A/stratify_smoke.txt`): A-T 31 + A-Y 31 묶음에서 n=20 을 뽑아 거리 비율 baseline 8.8 · 12.6 · 20.8 · 57.8% → 추출 5.0 · 15.0 · 20.0 · 60.0% (작은 묶음이라 근접 칸 부족분을 같은 거리 구간에서 메움). 같은 seed 로 두 번 돌리면 같은 20개를 뽑는다. 전체 규모(A-T · A-R 각 650 → 500)에서는 근접 칸(목표 44개)이 각 묶음(시도의 약 13.7% × 생성률)으로 충분할 것으로 본다. A-R 스모크의 근접 3% 는 30개 중 1개라 추정이 거칠다. 모자라면 칸 게이트(종료 코드 2, 4절)가 멈춘다.

## 4. 학습 전 데이터 게이트

**문턱은 prereg §3 데이터 게이트 표 하나만 쓴다.** 이전 판의 이 절 추가 게이트는 그 표로 옮겼다(A-R xy30 은 ≥ 0.15). 이 절에는 확인 명령만 둔다. 스모크 값은 3.2 표와 `results/plan_A/gate_table_1029.md`.

```bash
# 층화: 칸마다 |뽑음 − 목표| > 2 면 hdf5 를 쓰지 않고 종료 코드 2
python -I $A2/tools/plan_A_stratify.py --ref ./datasets/team12_kgh/kgh_mimic_500.hdf5 --n 500 --seed 0 --out <층화 500> <샤드 5개>
# 게이트: prereg §3 문턱과 비교, FAIL 이 하나라도 있으면 종료 코드 1
python -I $A2/tools/plan_A_smoke_check.py check AT <A-T 층화 500> --log <A-T 샤드 로그 5개>
python -I $A2/tools/plan_A_smoke_check.py check AR <A-R 층화 500> --log <A-R 샤드 로그 5개> --at_yaw <A-T 의 yaw_go>
```

- `--at_yaw` 에는 A-T `check` 출력의 `yaw_go`(손목 yaw 파지→상자 위 중앙값)를 넣는다.
- 행동 지표(rot60 · carry_pos_cmd)는 처치가 들어갔는지 보는 조작 확인용이다(prereg §4).
- **게이트를 못 넘으면 prereg §3 을 따른다:** 구현 오류를 찾아 고치면 다시 생성하고 prereg §7 변경 기록에 남긴다. 오류가 없으면 그 팔은 "게이트 미통과" 로 보고하고 학습하지 않으며 큐의 다음 팔로 간다.

## 5. 판정

**판정은 prereg 를 따른다: A-T 는 §5.2(① 효과 · ② 기각 · ③ 방향 일치 · ④ 보류, 400쌍), A-R ↔ A-T 는 §5.3(동등 · 차이 · 구분 불가, 400쌍), 해석은 §6 해석표.** 트랙 A 가 낸 값과 보는 지표는 다음뿐이다.
- A-T 최소 예측(규칙 ② 의 기준): **+4%p**. 6절 실패 이동 예측에서 유도했다(prereg §3). 이전 판의 +9.5%p 는 잡음 폭이라 효과 예측이 아니었다(prereg §7 변경 기록 1).
- 표적 묶음: 운반+배치(baseline 49건), 그 안의 귀속 (ii) 충돌 몫(19건). 상태 매개 지표(prereg §4): 건수 `push_carry`(= 귀속 충돌) · `box_push` · `low_approach`. 중앙값 `carry_yaw_net` · `rot60_state` 는 함께 보고한다.
- 하위 분석: 시작 거리 ≥ 0.35 m rollout 만의 Δ (`paired_compare.py --min_d0 0.35`, prereg §5.2).

A-T↔A-R 해석은 prereg §6 의 H-A2–4 만 쓴다: A-T + 이고 A-R 동등 → 회전 성분(H-A2) / 차이(A-T > A-R) → 위치·우회 성분(H-A3) / 구분 불가, A-R > A-T, A-R 게이트 미통과 또는 미실행 → 상자 구간 재타깃팅(회전 + 우회) 전체(H-A4). 확인 조건은 ≥ 0.35 m 하위 분석에서 A-T Δ > 0 이다. A-T 가 ② 면 H-A 기각(§6 H-A1)이고 A-R 해석은 하지 않는다.

## 6. 예상 실패 이동 (baseline → A-T, 200 rollout · 방향 예측)

| 묶음 (prereg §4 귀속 (ii) 계층: 충돌 → 스스로 놓음 → 미끄러짐 → 기타) | baseline (공용 `carry_metrics.py` 의 `attr_ii`) | A-T 예측 | 근거 |
| --- | --- | --- | --- |
| 충돌 (운반 창 상자 밂) | 19 | 줄어듦 (≤ 12) | 데이터 box_push 16% → 6%, xy30 0.212 → 0.089, approach_z p10 0.109 → 0.212 |
| 스스로 놓음 (소수 + 다수 열기) | 19 (6 + 13) | 줄어듦 (≤ 16, 약한 예측) | 벽 앞 놓음의 회전 명령 0.55 · 경로 단축 |
| **미끄러짐** | 8 | **8 ± 3 (불변)** | 미끄러짐(귀속 (ii) 8건) rot60 0.48 ≈ 들기 이후 실패 전체 0.49 — 회전·우회와의 연결 근거가 없음 (교차 점검 #17, 이전 판의 "≤ 6" 철회) |
| 기타 | 3 | 3 ± 3 | 공용 도구 기준. 트랙 C 도구는 5 로 센다(2건 차이, prereg §5.4) |
| 배치 · 입구 위까지 갔다가 밖에서 놓음 (v3 세부) | 12 | 줄어듦 | 회전 명령 0.89, 9/12 가 상자 밂 (서베이 초안의 "그대로" 를 고침) |
| 들기 전 실패 (접근+파지, 음성 대조) | 19 | prereg §5.2 음성 대조 (띠 3.03/100 — json 값 — 밖 그리고 짝 McNemar 양측 p < 0.05 이면 위반) | 트럭 구간(subtask 1)은 안 바꾼다 |

- 이 예측이 prereg §3 의 A-T 실패 이동 예측이고, 최소 예측 +4%p 의 근거다: 충돌 −7 의 대부분과 스스로 놓음 −3 의 일부가 성공으로 옮겨 간다고만 가정한다(전부 옮겨 가면 200개당 10건 = +5%p).

## 7. 실행 명령 · 계산량

모두 `~/IsaacLab_RS` 에서 conda `lerobot-arena`, A6000. 프로세스마다 TMPDIR 을 따로, tmux 창마다 하나씩(20초 간격) 띄운다. 이전 판 대비 변경: A-T · A-R 모두 샤드당 130개(합 ≈ 650) + `--src_log` + 같은 층화 단계, 게이트는 `check AT|AR`(prereg §3, 샤드 로그 5개 합산), 평가 seed 43 추가. 단계마다 종료 코드가 0 이 아니면 다음 단계로 가지 않는다(prereg §7).

```bash
set -o pipefail   # tee 뒤에도 앞 명령의 종료 코드를 남긴다
A2=~/robotics-simulation-team12/assignment2_toy2box; REF=./datasets/team12_kgh/kgh_mimic_500.hdf5
P=./datasets/team12_kgh/plan_A; L=~/IsaacLab_RS/logs/rs_a2_runs/plan_A
# (1) A-T 생성: seed 1–5 × 성공 130 (≈ 650), 샤드마다 tmux 창 하나 (s 만 바꿔 5번)
s=1; mkdir -p /tmp/rs_a2_tmp_AT_s$s $P/translate_shards
TMPDIR=/tmp/rs_a2_tmp_AT_s$s PYTHONUNBUFFERED=1 ./isaaclab.sh -p $A2/tools/generate_yawhold.py --translate 1 --datagen_seed $s \
  --src_log $L/AT_s${s}_src.jsonl --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 50 \
  --generation_num_trials 130 --input_file ./datasets/team12_kgh/toy2box_kgh_annotated.hdf5 \
  --output_file $P/translate_shards/translate_s${s}_130.hdf5 > $L/AT_s$s.log 2>&1
# (2) 층화 500 (칸 게이트 미통과면 종료 코드 2) + 데이터 게이트 (prereg §3, 미통과면 종료 코드 1)
python -I $A2/tools/plan_A_stratify.py --ref $REF --n 500 --seed 0 \
  --out $P/translate_strat500.hdf5 $P/translate_shards/translate_s{1,2,3,4,5}_130.hdf5 | tee $A2/results/plan_A/stratify_AT500.txt
python -I $A2/tools/plan_A_smoke_check.py check AT $P/translate_strat500.hdf5 \
  --log $L/AT_s{1,2,3,4,5}.log | tee $A2/results/plan_A/gate_AT500.txt
# (3) 학습·평가·분류 (2 seed, 각 tmux 창; run_baseline.sh 종료 코드 0 성공 · 5 평가 실패 · 6 분류 실패)
bash $A2/tools/run_baseline.sh translate500 $P/translate_strat500.hdf5 101
bash $A2/tools/run_baseline.sh translate500 $P/translate_strat500.hdf5 111
# (4) A-R: (1) 을 --translate 1 → --rotworld 1, 이름 rotworld · AR 로 (샤드당 130, --src_log $L/AR_s${s}_src.jsonl)
#     층화는 A-T 와 같은 도구·seed, 게이트는 A-T 의 yaw_go 와 비교
python -I $A2/tools/plan_A_stratify.py --ref $REF --n 500 --seed 0 \
  --out $P/rotworld_strat500.hdf5 $P/rotworld_shards/rotworld_s{1,2,3,4,5}_130.hdf5 | tee $A2/results/plan_A/stratify_AR500.txt
python -I $A2/tools/plan_A_smoke_check.py check AR $P/rotworld_strat500.hdf5 --log $L/AR_s{1,2,3,4,5}.log \
  --at_yaw $(grep -oP 'yaw_go = \K[0-9.]+' $A2/results/plan_A/gate_AT500.txt) | tee $A2/results/plan_A/gate_AR500.txt
python -I $A2/tools/plan_A_smoke_check.py table --set baseline $REF \
  --set A-T $P/translate_strat500.hdf5 $L/AT_s1.log,$L/AT_s2.log,$L/AT_s3.log,$L/AT_s4.log,$L/AT_s5.log \
  --set A-R $P/rotworld_strat500.hdf5 $L/AR_s1.log,$L/AR_s2.log,$L/AR_s3.log,$L/AR_s4.log,$L/AR_s5.log
bash $A2/tools/run_baseline.sh rotworld500 $P/rotworld_strat500.hdf5 101   # 111 도
# (5) 평가 seed 43: 학습된 run 을 평가만 (prereg §7 큐에서 B3' 뒤에 baseline · B2 · B3' 와 묶어 돈다)
bash $A2/tools/run_baseline.sh translate500 $P/translate_strat500.hdf5 101 2000 43   # 111 도, rotworld500 도
# (6) 판정 400쌍 (prereg §5.2 · §5.3). A-R ↔ A-T 는 --a 를 translate500, --b 를 rotworld500 으로
R=$A2/results/baseline
python -I $A2/tools/paired_compare.py --require_carry --label_a baseline --label_b A-T \
  --a $R/base_kgh500_s{101,111}_e2000_seed{42,43}.csv --b $R/base_translate500_s{101,111}_e2000_seed{42,43}.csv
#     같은 명령에 --min_d0 0.35 (하위 분석), --bundle approach,grasp (음성 대조)
```

- **계산량 (A6000):** A-T 생성 약 20–25분(성공률 89% → 시도 약 730회) + 층화·게이트 1분 + 학습 2 seed 동시 약 45분 + 평가 2개 동시 약 65분 ≈ **2.3 h**. A-R 은 생성 약 45분(생성률 58% → 성공 650 에 시도 약 1,120회) 포함 ≈ **2.6 h**. 평가 seed 43 은 팔마다 2 체크포인트 동시 약 65분. 일정은 prereg §7 직렬 큐를 따른다.

## 8. 위험 · 대안 · 규칙 확인

- **규칙:** 환경·판정·종료 조건 변경 없음(생성기 래퍼만), 사람 시범 20(≤ 100), 학습에 쓰는 Mimic 성공 500(≤ 1000; A-T · A-R 은 생성 650 중 층화 500 만 학습에 쓰고 나머지는 분석에만), 학습 조건·체크포인트 규칙·평가 규약 동일, seed 101·111. 생성 시도 수·생성률을 보고한다.
- **위험 1 — 성공 필터:** 층화로 거리·상대 yaw 주변분포는 맞추지만, 칸 안의 다른 요인(트럭 yaw 와 원본 최근접 거리 등)까지 맞추지는 못한다. A-A(SE(3), datagen seed 6–10)는 prereg §3 의 선택 팔이고 큐 밖이다. 큐가 끝나고 슬롯이 남을 때만 데이터 재추출 분산의 참고값으로 재며, 판정·확인 조건에는 쓰지 않는다.
- **위험 2 — 원본 world 방향 접근:** 상자가 많이 돌아간 장면에서 원본 접근 방향이 벽과 어긋날 수 있다. 스모크: A-T box_push 6%, 성공 시점 xy 0.053. 전체 500 에서 게이트로 다시 본다.
- **위험 3 — A-R 의 큰 행동 회전 명령(0.263):** 정책이 큰 회전 명령을 배울 수 있다. 기전 비교는 상태 매개 지표로만 하고(prereg §4), A-R 이 baseline 보다 나빠지면 "위치 SE(3) + 원본 회전의 결합 손해" 로 한계에 적는다.
- **위험 4 — A-R 을 못 돌림 (큐 시간 부족 또는 게이트 미통과):** A-Y 대안은 없다(prereg §3). 그때는 결론 범위를 "상자 구간 재타깃팅(회전 + 우회)" 까지로 줄인다(prereg §6 H-A4).
- **범위 밖 실패와 대안 실험:** 근접 시작(< 0.30 m: 성공 43%, 실패 16건)과 트럭을 놓친 뒤 회복 못 함(빈손으로 상자에 감) — A-T 로는 바뀌지 않을 것으로 예측. 대안 A-N(DART 식: Mimic `action_noise` 0.01 → 0.03, 미구현), 대안 A-H(근접 영역 사람 시범 추가 — 개수 맞춤·텔레옵 필요, 이번 일정 제외).

## 9. 남은 질문

1. (정해짐) A-R 은 prereg §7 큐에 들어갔다(B2 뒤, 약 2.6 h). 못 돌리면 A-Y 없이 결론을 "상자 구간 재타깃팅(회전 + 우회)" 까지로만 쓴다(prereg §3).
2. (완료 10/29) 생성기 "원본은 subtask 1 에서만 고름" 정정을 진단 §4.5 · 서베이 §2.1 에 넣었다.
3. 층화 뒤에도 남는 칸 안 분포 차이를 A-A 로 잴지 — A-A 는 prereg §3 의 선택 팔(큐 밖)이라 슬롯이 남을 때만 정한다. 판정에는 쓰지 않는다.
