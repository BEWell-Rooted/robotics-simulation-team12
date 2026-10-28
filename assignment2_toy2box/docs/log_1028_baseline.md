# 과제 2 Toy2Box — 팀 데이터 baseline · rollout 기록/실패 분류 도구 (2026-10-28 저녁)

| 항목 | 값 |
| --- | --- |
| 세션 | A6000 Claude Code (1028 텔레옵 이전 세션 이어서) |
| 머신 | A6000 `cai-Precision-7960-Tower` (RTX 6000 Ada, 드라이버 595 → 렌더링 불가, headless 무렌더만) · DGX Spark `gx10-2` (렌더링용 환경 설치) |
| 데이터 | 팀원 시범 `toy2box_kgh.hdf5` 20개 + annotate 본 `toy2box_kgh_annotated.hdf5` (형근이 Downloads 로 받음) |
| 시작 | 18:56 |

(작성 중 — 결과가 나오는 대로 채운다)

## 데이터 확인

- 사본: `~/IsaacLab_RS/datasets/team12_kgh/` (원본 `~/Downloads/` 그대로 둠). md5 `toy2box_kgh.hdf5` df35fb05883aede410b26c6dbde8819b · `toy2box_kgh_annotated.hdf5` 8ddcba040c4d44e0ae933df2855b28ff.
- 20개 모두 `success=True`, 총 7,793 step, 길이 15.4–24.6초(평균 19.5초). 그리퍼 전환 20개 모두 2회(재파지 0), 멈춤(idle) 8–38%(대부분 12–27%).
  - 조교 시범 20개(평균 27.2초, 재파지 7개, idle 평균 28%, 닫기 전 트럭을 최대 8 cm 밀고 잡음)보다 짧고 일관적이다. 팀원 시범은 닫기 전 트럭 변위가 최대 1.8 cm.
- annotate 본: subtask 신호 `grasp` 가 20개 모두 1회 켜진다(닫기 명령 3–12 step 뒤). annotate 본의 demo 순서는 원본의 사전순(demo_0, 1, 10, 11, …)으로 다시 매겨져 있다.
- `env_args.env_name` 이 원본에서도 `…-Rel-Mimic-v0` (조교 원본은 `-v0`). 같은 장면이라 학습·평가에는 영향 없음(play.py 는 `--task` 를 씀).

## 만든 도구 (`tools/`, IsaacLab_RS 추적 파일은 고치지 않음)

| 파일 | 하는 일 | 확인 |
| --- | --- | --- |
| `train_seed.py` | `train.py` 에 `--seed` 를 준다 (json.load 를 감싸 `train.seed` 만 바꿈) | 100 epoch 시험: run 폴더 `config.json` 의 `train.seed` = 111 |
| `generate_seed.py` | `generate_dataset.py` 에 `--datagen_seed` 를 준다 (cfg 고정값 1 → 샤드별 seed) | 샤드 로그 `[generate_seed] datagen_config.seed 1 -> N` |
| `merge_shards.py` | 샤드 병합 + `data.attrs['total']`·출처 기록 (merge_hdf5_datasets.py 와 같은 복사 방식) | 시험 병합 50개 |
| `play_record.py` | `play.py` 를 그대로 실행하면서 rollout 마다 매 step 상태를 HDF5 로 기록 (관측 7키, 행동, grasp 신호, 패드 힘, 성공 조건 성분, `scene.get_state`) | 같은 seed 로 두 번 돌린 3 rollout 이 행동·상태 모두 bit 단위로 같음 |
| `classify_failures.py` | 궤적을 접근 · 파지 · 운반 · 배치로 분류 (기록 파일·사람 시범·Mimic 생성물 모두 같은 기준) | 사람 시범 60개·Mimic 성공 52개 전부 성공 판정, 원본(신호 없음)과 annotate 본(신호 있음)의 들기 시점 일치 |
| `render_record.py` | 기록된 상태를 다시 세워 영상·장면 띠 생성 (렌더링 되는 머신용: Dell, DGX Spark) | Spark 설치 후 확인 |
| `analyze_baseline.py` | seed 별 분류 CSV → 깔때기·범주·시작 위치·트럭 방향 그림 + 표 | |
| `run_baseline.sh` | 학습(seed) → 평가(seed 42, 100회) → 기록 → 분류 한 줄 실행 | |

### 실패 분류 기준 (사람 시범 40개 실측에서 정함)

- 이정표: `t_near` 손끝–트럭 3D < 0.05 m (사람 grasp 순간 0.006–0.045 m) → `t_grasp` 환경 grasp 신호 3 step 연속 → `t_lift` 트럭이 처음 높이 + 0.03 m 위 · 닫기 명령 · 손끝–트럭 < 0.08 m(Mimic 운반 중 최대 0.055 m)가 5 step 연속 → `t_over` 들린 트럭이 상자 중심 xy < 0.10 m → 성공.
- 범주 = 가장 멀리 간 이정표의 다음 단계. 세부 유형:
  - 접근: knocked(닫기 전에 트럭을 5 cm·30° 이상 밀거나 넘어뜨림) / close_misaligned(10 cm 안에서 닫았지만 5 cm 안엔 못 감) / hover / far
  - 파지: miss(닫았지만 grasp 안 됨) / slip(잡았지만 못 듦) / no_close
  - 운반: dropped(닫은 채 미끄러짐) / released_early(상자 위 전에 열어서 놓음) / *_near_box(상자 16 cm 안) / off_table / stall
  - 배치: no_release / released_outside / in_box_not_open / other
- 성공 조건 실측: 성공 순간 트럭–상자 z 차 −0.091 ~ −0.100 m 인데 조건은 |z 차| < 0.10 m → 높이 조건은 사실상 "테이블 높이 이하"면 통과, xy < 0.10 m 와 그리퍼 열림이 실제 관문.

## 파이프라인에서 찾은 함정

1. **RMPFlow 는 env reset 마다 컨트롤러를 모든 로봇에 대해 다시 만들고, URDF·설정 파일 3개를 `force_download=True` 로 `tempfile.gettempdir()`(`/tmp`)에 다시 받는다** (`rmpflow_task_space_actions.py:189` → `rmp_flow.py initialize`).
   - 같은 머신에서 Isaac 프로세스 여러 개를 동시에 띄우면 `/tmp/agibot.urdf` 를 서로 덮어써 `Failed to parse URDF` 로 죽는다(샤드 2–5 첫 시도). → **프로세스마다 `TMPDIR` 을 따로 준다** (`run_baseline.sh` 에 넣음).
   - Mimic 생성(env 여러 개)에서는 env 하나가 끝나 reset 될 때마다 *다른 env 의 RMPFlow 도 다시 만들어진다*. 이 cfg 는 `ignore_robot_state_updates=True`(내부 상태 적분)라 진행 중인 다른 env 의 내부 상태가 현재 로봇 상태로 다시 맞춰진다. 생성 품질에 영향이 있는지는 확인하지 않았다(관찰만).
2. Mimic 생성 seed 가 cfg 에 1로 고정 → 프로세스를 나누면 같은 시범이 중복 → `generate_seed.py`.
3. Mimic 행동 = (목표 손끝 자세 − 현재 손끝 자세), ±1 로 자름 (`franka_stack_ik_rel_mimic_env.py:69–85`). 로봇이 목표를 늦게 따라가는 오차가 행동에 들어가므로, 사람 시범(−0.05/0/+0.05 세 값, 멈춤 20%)과 분포가 크게 다르다: Mimic 생성물 Δx 상위 10% 0.127, 회전 최대 1.0, 멈춤 0.8%. **BC 가 실제로 배우는 것은 이 Mimic 행동이다.**

## Mimic 생성 (팀원 시범 20개 → 성공 500개)

- 조교 명령(`--num_envs 100 --generation_num_trials 500`, 단일 프로세스)으로 시작했더니 CPU 1코어(112%)에 묶여 10분에 시도 68·성공 37 → 500개에 약 2시간 예상. 중단하고(`datasets/team12_kgh/_stopped/`, 성공 48·실패 39) seed 1–5 × 성공 100개 샤드 5개(`--num_envs 50`, 프로세스별 `TMPDIR`)로 다시 돌렸다. 19:14–19:53 (샤드 2–5 는 URDF 충돌로 한 번 죽은 뒤 19:17–19:18 재시작).
- 결과: **성공 500 / 시도 1,130 (44.2%)** — 샤드별 39.4 · 49.3 · 45.5 · 43.5 · 44.8%. 조교 데이터 스모크(4/6 = 67%)보다 낮다.
- 병합 `datasets/team12_kgh/kgh_mimic_500.hdf5` (500 demo, 207,979 step, md5 5e9bebaf4ee4111f4cc4cae9b58c994d). 성공 시범 길이 중앙값 394 step(사람 373).
- **생성 실패 630건 분류** (`results/mimic/kgh_mimic_500_and_failed.csv`): 운반 374 (59.4%: 상자 위에 가기 전에 그리퍼를 열어 놓음 245, 닫은 채 미끄러짐 97, 상자 16 cm 안에서 놓음 32) · 배치 106 (16.8%: 상자 밖에서 놓음 102) · 접근 105 (16.7%: 10 cm 안에서 닫았지만 5 cm 안에 못 감 83) · 파지 45 (7.1%).
  - → 생성 실패의 3/4 이 **두 번째 구간(상자 기준 운반·놓기)** 에서 난다. 원본 시범의 놓는 위치가 성공 경계에 가까운 경우가 있다(트럭–상자 xy 최대 0.099 m, 경계 0.10 m). 상자 기준으로 옮긴 궤적에 추적 지연·행동 노이즈(0.01)가 더해지면 경계를 넘는 것으로 본다(추정).
- 조교 데이터 대조군 Mimic(같은 방식, seed 1–5 × 100)은 A6000 학습을 느리게 해서(2.1 → 1.1 s/epoch) gx10-2 로 옮겨 생성 중.
  - gx10-2 에서 조교 데이터 샤드 5개를 20:00–20:08 돌린 결과 성공 25 / 시도 136 (18.4%). A6000 에서 멈추기 전 조교 샤드는 3 / 12 (25%), 1021 스모크(env 4개)는 4 / 6. 표본이 작아 **기계 차이인지 조교 데이터 특성인지 판단할 수 없다.** 비교를 같은 조건으로 두려고 생성·학습·평가는 모두 A6000 에서 하고 Spark 는 기록 상태 재생(렌더링)에만 쓴다. Spark 부분 산출물은 gx10-2 `datasets/ta_ref/mimic_shards/_stopped_spark/`.

## 기록 도구가 평가를 바꾸지 않는지 확인

- 결정성: 같은 체크포인트·seed 42 로 `play_record.py` 를 두 번 돌린 3 rollout 이 행동·관측·관절 상태 모두 차이 0 (A6000, Mimic 샤드 5개가 동시에 돌던 상태).
- 단독 대조: baseline s101 e2000 을 조교 `play.py` 그대로(기록 없이) 동시에 돌려 rollout 별 성공/실패를 비교 → **처음 31회가 완전히 같음** (단독 실행은 35회에서 멈춤, 본 평가 속도를 위해). `docs/run_logs/b3_eval_plain_base_kgh500_s101_e2000_stopped.log`.
- 같은 평가 seed 에서는 정책이 달라도 rollout 번호별 시작 위치(트럭·상자 자세)가 같다 → seed 101·111 정책을 rollout 단위로 짝지어 비교할 수 있다.
  - **기계 차이 확인 (21:03–21:13):** gx10-2 에서 팀원 데이터 Mimic 을 A6000 과 같은 seed 1 로 돌렸더니 시도 206회에 성공 77 (37.4%) — A6000 seed 1 의 같은 시점 78/206 (37.9%) 과 같다. → Spark 에서 본 조교 데이터의 낮은 생성률(18%)은 기계가 아니라 **조교 시범 데이터의 특성**이다. 조교 대조군 Mimic 을 gx10-2 에서 다시 생성(21:14~).

## Baseline 학습·평가 (A6000)

| run | 학습 | 평가 (seed 42, 100회, horizon 800) | 기록 |
| --- | --- | --- | --- |
| `base_kgh500_s101` | 19:53–20:36 (2000 epoch, 1.1–2.1 s/epoch) | 20:36–21:40 · **63/100** | `~/IsaacLab_RS/logs/rollout_records/base_kgh500_s101_e2000_seed42.hdf5` |
| `base_kgh500_s111` | 19:53–20:35 | 20:35–21:38 · **69/100** | `…/base_kgh500_s111_e2000_seed42.hdf5` |

- **66.0 ± 3.0% (표본표준편차 4.2%)**. 조교 시연(조교 시범) 26%.
- 체크포인트: `~/IsaacLab_RS/logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/base_kgh500_s{101,111}/<시각>/models/model_epoch_2000.pth`, 저장 config `train.seed` 101 · 111 확인.
- 평가 속도: 평가 3개 동시(기록 2 + 단독 대조 1)에 rollout 당 약 37–48 초 → 100회 약 65분. GPU 사용률 99%.
- 분류·분석: `results/baseline/base_kgh500_s{101,111}_e2000_seed42.{csv,json,txt}`, `results/baseline/analysis/` (깔때기·범주·시작 위치·yaw 그림, `summary.md`). 진단 정리는 `docs/diagnosis_1028.md`.
- 대표 영상(gx10-2 렌더, H.264 Main): `~/IsaacLab_RS/videos/base_kgh500_s101_e2000/` — rollout 0·1 성공, 2 운반/미끄러짐, 5 접근(트럭이 상자 옆), 12 배치/상자 밖, 14 운반/일찍 놓음. 실패 68건 장면 띠(jpg)도 같은 폴더와 scratch 에 있음.

## 조교 데이터 대조군 (진행 중)

- 같은 파이프라인(seed 별 샤드)으로 조교 시범 20개 → Mimic 500 생성 중: gx10-2 seed 1·2 × 100, A6000 seed 3–8 × 50. 생성 성공률 gx10-2 17.5–18.7%, A6000 14.5–20.8% → **두 기계에서 같은 수준이고, 팀원 시범(44%)의 절반 이하.**

## 실패 분류 검증 (워크플로 `toy2box-classifier-verify`, 21:52–22:15)

- 실패 68건 전부를 gx10-2 에서 장면 띠(2×4, 처음·이정표·끝)로 렌더(`tools/render_on_spark.sh ... --rollouts fail --sheet --no_video`), 시간축 그림(`classify_failures.py --plots`)과 분류 행을 묶어 작업자 9명(8건씩)이 이미지를 직접 보고 라벨을 반박하게 했다. 이견·저확신 5건은 재판정 작업자가 상자 몸체 좌표로 다시 계산했다.
- v1 결과: **범주 일치 63/68 (93%)**, 세부 유형 일치 36/68 (53%), 확신 high 38 · medium 30.
- 고친 것(v3, `classify_failures.py` docstring 에 정리):
  1. "상자 위"를 원형 xy < 0.10 m 대신 상자 몸체 좌표 입구(반폭 0.11 m) 판정 + 3 step 연속으로 (1 step 스침, 기울어진 상자 중심이 트럭 밑을 지나간 가짜 판정 제거).
  2. 들기 대안 조건(grasp 신호 + 정지 높이 + 5 cm, 3 step) — 트럭은 reset 뒤 1.9 cm 가라앉으므로 정지 높이를 따로 잡음.
  3. 배치 세부 `in_box_offcenter`(상자 안 구석인데 중심 0.10 m 밖), `dropped_outside`; 판정 순서 수정(`in_box_not_open` 이 실제로 잡히게).
  4. 운반 세부 `box_knocked_over`, `box_collision_released/dropped` — 놓치기 직전 10 step 창에서만 상자 이동을 본다. **상자를 미는 것 자체는 성공 시범 81/540 에서도 일어나므로** 범주를 바꾸지 않고 `box_disp_max` 열로만 남긴다.
  5. 그리퍼 1-step 스파이크 제거, '상자 가까이' 반경 0.20 m, 접근 세부 `wrong_object`·`freeze`·`to_box_without_truck`, 표시 열(`empty_at_box`, `shallow_grasp`, `rot60_after_lift`).
- v3 에서 사람 시범 80개(조교·팀원 원본·annotate 본)·Mimic 500개는 모두 성공. 실패 범주: 접근 9 · 파지 10 · 운반 30 · 배치 19 (v1: 9 · 11 · 35 · 13). 범주가 바뀐 9건은 대부분 운반 → 배치.
- 판정 원문: 워크플로 journal (`~/.claude/projects/-home-cai-robotics-simulation-team12/9bd6bf1a-…/subagents/workflows/wf_eeba3be5-b7a/journal.jsonl`), 재판정 패치 원본 scratch `v2/cf_v3.py`.

## 학습 곡선 · 기계 간 평가 차이 (gx10-3, 20:30–22:51)

- seed 101 의 500·1000·1500·2000 epoch 체크포인트를 gx10-3(DGX Spark, 설치 7분 — `tools/setup_spark_env.sh`)에서 `play_record.py` 로 평가(4개 동시, rollout 당 약 1.5분):

  | epoch | 500 | 1000 | 1500 | 2000 |
  | --- | --- | --- | --- | --- |
  | 성공 (gx10-3) | 70 | 71 | 61 | 74 |
  | 운반+배치 실패 / 실패 | 22/30 | 19/29 | 23/39 | 18/26 |

- 같은 2000 epoch 체크포인트가 A6000 에서는 63/100. rollout 별 결과도 처음 17회 중 3회가 달랐다. 생성(Mimic) 성공률은 두 기계에서 같았으므로(팀원 seed 1: 37.4% vs 37.9%), 평균은 같고 개별 결과가 갈리는 수치 차이로 본다. → 비교 실험은 같은 기계(A6000)에서만 한다.
- 기록·분류 CSV: `results/learning_curve_spark/spark3_s101_e*.csv` (원본 기록은 gx10-3 `~/IsaacLab_RS/logs/rollout_records/spark3_*`).

## 조교 데이터 대조군 (계속)

- Mimic 500 완료 22:49: gx10-2 seed 1·2 (100/529, 100/592) + A6000 seed 3–8 × 50 (시도 253·270·318·277·334·288) → **500/2,861 = 17.5%**. 병합 `datasets/ta_ref/ta_mimic_500.hdf5` (282,944 step, md5 a99ccdb0a24505729f2f01de5d70add9).
- 조교 Mimic 500 도 운반 중 손목 yaw 변화 24°(p90 90°) — 사람(조교) 3°. 회전은 Mimic 변환의 성질.
- 학습 `base_ta500_s101`·`s111` 22:50 시작 (A6000), 끝나면 같은 평가.

## 서베이 (워크플로 `toy2box-diagnosis-survey`, 21:52–23:05) · 잡음 기준선 (23:07–00:16)

- 트랙 3개(A Mimic 생성·데이터 품질 / B 행동·관측 표현 / C 정책 구조·폐루프) × 후보 수집 → 정독 → 인용·수치 원문 대조 + 실험 설계 → 통합 → 완결성 점검, 작업자 11명. 결과 `docs/survey_1028.md`, 작업자가 측정에 쓴 스크립트 `tools/survey_analysis/`.
- 점검이 지적한 판정 기준 문제(잡음만으로 McNemar 통과)를 확인하려고 잡음 기준선을 쟀다: gx10-3 s101 e1800 73 · e1900 65, gx10-2 s111 e1900 72 · e2000 69 (`results/noise_baseline/`). 같은 기계 체크포인트 15쌍 차 최대 13%p, 양측 p 모두 ≥ 0.079. 같은 체크포인트 다른 기계는 24–25개 rollout 이 뒤집힘. 진단 문서 7절.
- A6000 에서도 e1900 2개를 같이 돌렸으나(23:35) 평가 4개 동시로 rollout 당 약 70초가 되어 조교 대조군을 먼저 끝내려고 23:47 에 멈춤(부분 기록 `logs/rollout_records/_stopped/`).
- `tools/paired_compare.py`: 같은 평가 seed 의 두 평가를 rollout 단위로 짝지어 정확 McNemar·범주 이동표·기전 지표 비교.
- 결과 페이지(비공개): https://claude.ai/artifact/5mQnzB4tWAAT5pkioTsqFG

## 조교 데이터 대조군 결과 (00:53)

| run | 학습 | 평가 | 결과 |
| --- | --- | --- | --- |
| `base_ta500_s101` | 22:50–23:34 | 23:34–00:52 | **37/100** |
| `base_ta500_s111` | 22:50–23:34 | 23:34–00:53 | **33/100** |

- **35.0 ± 2.0%** (조교 보고 26%). 팀원 baseline 과 200쌍 비교: +31%p, 불일치 18:80, McNemar p < 0.001. 실패 130건: 접근 43 · 파지 23 · 운반 47 · 배치 17 (팀원 68건: 9 · 10 · 30 · 19). 들기 전 실패 66 → 19. 진단 문서 8절.
