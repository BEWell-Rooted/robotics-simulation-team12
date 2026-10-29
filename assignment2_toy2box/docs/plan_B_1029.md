# 트랙 B (담당 B) — 행동 라벨 표현: 기전 분해 팔 B2 · 귀속 팔 B3' (2026-10-29, 교차 점검 · 독립 점검 반영판)

> 연구 축 C(관측·행동 표현). 팀의 원인 가설은 **H-A 하나**(Mimic 상자 구간 SE(3) 재타깃팅이 만든 불필요한 운반 동작, 확증 처치 **A-T**)다. 트랙 B 의 팔은 그 원인을 **분해**하는 역할이다: B2 는 "확증·기전 분해" 팔, B3' 는 B2 와 직접 비교하는 "귀속" 팔 (등급은 prereg §3).
> **판정 수치는 `docs/prereg_1029.md` 를 따른다:** B2 는 §5.2, B3' 는 §5.3, 해석은 §6, 큐는 §7. 숫자가 다르면 prereg 가 우선이다. 이 문서에는 트랙 B 의 구현, 조작 확인 게이트, 상태 지표의 예측 방향만 둔다. 최소 예측 +4 는 prereg §3 표와 같다.
> 근거 문서: `docs/diagnosis_1028.md`, `docs/survey_1028.md` §2.2·§6, 교차 점검 통합(§1 #2·#7·#11·#12·#17·#20, §2, §4). 산출물: `tools/relabel_actions.py`, `tools/replay_check.py`, `tools/plan_B_speed_metrics.py`, 데이터 `~/IsaacLab_RS/datasets/team12_kgh/plan_B/`, 점검 결과 `results/plan_B/`.

## 0. 요약

- **B2 (reach k=4):** 행동 라벨을 "4 step 뒤 실제 도달 손끝 자세 − 현재"로 다시 정의한다. 상태는 그대로 두고(손목 yaw 38.4° 유지) 명령 크기·방향을 실제 동작에 맞춘다. A-T 와 함께 보면 "회전 상태가 필요조건인지, 공통 기전이 명령 크기·우회인지"를 가른다 (교차 점검 §4).
- **B3' (원본 방향 × B2 크기):** B2 와 크기 분포가 똑같고(노름 중앙값·p90 차이 0.00%) 방향만 원본이다. B2 와 직접 비교해 "크기 vs 방향·일관성"을 귀속한다. 구판 B3(상한)은 크기를 맞추지 못해 폐기했다 (#12).
- **B2-carry (예비, 해석용):** 파지 뒤 닫힘 구간만 reach k=4. 접근 단계를 원본 그대로 두고 운반 라벨만 바꾼다. B2 에서 "들기 전 실패 증가"가 나오면 그것을 떼어 보는 용도. prereg §7 큐 밖이다.
- **B1 (운반 회전 라벨 0) 은 실행하지 않는다** (3.3).

## 1. 가설과 역할

**H-A 의 기전 분해 질문 (B2):** H-A 가 만든 운반 실패가 "손목이 도는 상태 그 자체" 때문인가, 아니면 Mimic 라벨이 실제 동작과 어긋난 *큰 명령*이라서 생기는 흔들림·우회·충돌 때문인가.
B2 는 데이터의 회전 상태(손목 yaw 38.4°)를 그대로 두고 명령만 실제 동작에 맞춘다. B2 가 개선되면서 B2 정책의 운반 손목 yaw(`carry_yaw_net`)가 baseline 근처에 머물면 "회전 상태는 필요조건이 아니고 공통 기전은 명령 크기·우회"로 해석한다 (교차 점검 §4). 해석과 확인 조건은 prereg §6 의 'H-A 와 H_B 를 함께 읽기' 를 따른다.

근거 (운반 구간 = 그리퍼 닫기 · 트럭 z > 처음 z + 0.03 · 손끝–트럭 < 0.08, 데이터 500개):

| | 사람 시범(팀원 20) | Mimic 500 (baseline 학습 데이터) | B2 | B3' |
| --- | --- | --- | --- | --- |
| 라벨 방향 vs 실제 다음 1 step 이동 (cos 중앙값) | 1.00 | 0.68 | 0.94 | 0.68 (원본과 같음) |
| 라벨 방향 vs 실제 4 step 이동 | 1.00 | 0.59 | 1.00 | 0.59 |
| 라벨 크기 / 실제 4 step 이동 | 1.06 | 2.31 | 1.00 | 1.00 |
| 위치 라벨 ≥ 0.08 비율 | 0% | 52.4% | 5.0% | 5.0% |
| `carry_rot_cmd` · `carry_pos_cmd` (라벨) | 0.022 · 0.038 | 0.179 · 0.137 | 0.091 · 0.055 | B2 와 같은 크기 분포 |
| `carry_yaw_net` (상태) | 2.4° | 38.4° | 38.4° | 38.4° |

- RMPFlow 이득(명령 대비 실제 1 step 이동, Mimic 데이터): 위치 0.22, 회전 0.20 → k = 1/이득 ≈ 4 (학습 전에 고정).
- baseline rollout 에서 들기 이후 실패는 운반 손끝 속도가 더 빠르다: 0.0133–0.0140 m/step (성공 0.0101–0.0103) — `results/plan_B/speed_baseline_rollouts.txt`. **B2 가 운반을 늦추기만 해도 개선될 수 있으므로** 운반 속도를 상태 매개 지표 `carry_speed`(prereg §4)로 따로 잰다 (#11).

**최소 예측 (기각 기준용, prereg §3 표 · §5.2 ②):** B2 대 baseline Δ = **+4%p**.

**B3' 의 질문:** B2 와 B3' 는 크기 분포가 같고 방향만 다르다. 둘이 다르면 "방향·일관성"이, 같으면 "크기"가 B2 효과의 몫이다. 판정(동등 · 차이 · 구분 불가)은 prereg §5.3, 해석은 prereg §6 의 H_B2–4 를 따르고, B2 의 `carry_speed` 차를 함께 보고한다. 귀속 팔은 성공률 '효과'를 주장하지 않는다.

**보고서 서술 (#20):** B2·B3'·B2-carry 는 **같은 성공 에피소드(Mimic 이 성공으로 확인한 500개)의 상태에 행동만 hindsight 로 다시 정의**한 데이터다(축 C). 관측·상태·초기 상태는 bit 단위로 원본과 같고, 재라벨 행동을 open-loop 로 재생하면 성공하지 않을 수 있다(3.2: B2 4/10, B3' 0/10). 그래서 "성공 시범" 시비가 생기면 이 정의를 명시하고 필요하면 조교에게 확인한다.

## 2. 한 요인 · 대조 · 처리 · 고정 조건

| 팔 | 한 요인 | 비교 (prereg) | 데이터 (md5) |
| --- | --- | --- | --- |
| B2 | 행동 라벨 정의 `actions[:,0:6]` = (p[t+4] − p[t], rotvec(R[t+4]·R[t]ᵀ)) base 좌표, ±1 로 자름 | baseline 400쌍 (seed 42+43) | `kgh_mimic_500_reach4.hdf5` (32f3c488ce9435e073adc62be94a9853) |
| B3' | 라벨 방향만 원본으로 (크기는 B2) | B2 와 직접 400쌍 (seed 42+43) | `kgh_mimic_500_dirorig_magb2.hdf5` (a025e391eacf65ca1cf3b27140f160ce) |
| B2-carry (예비) | 파지 뒤 닫힘 구간만 B2 라벨 | B2 와 직접 (prereg §7 큐 밖) | `kgh_mimic_500_reach4carry.hdf5` (3d5fb2875ac4fa35ae84e0620283e044) |

- 원본: `kgh_mimic_500.hdf5` (5e9bebaf4ee4111f4cc4cae9b58c994d). 세 처치 모두 같은 500개 시범, 관측·상태·초기 상태·그리퍼 라벨 bit 동일.
- 고정: 사람 시범 20 · Mimic 500 · `bc_rnn_low_dim.json` 그대로(행동 정규화 없음, 2000 epoch × 100 step, batch 100) · 마지막 체크포인트 · 학습 seed 101·111 · 평가 seed 42 + 43 · 100회 · horizon 800 · A6000 한 대 · 분류 v3 · 지표 스크립트는 prereg 를 커밋한 커밋의 판(그 커밋은 `docs/log_1029.md` 에 기록). 학습량 seed 당 200k gradient step.
- 규칙: 환경·성공 판정·종료 조건 그대로, 사람 시범 20 ≤ 100, Mimic 500 ≤ 1000(같은 성공 에피소드), 행동 표현(축 C)은 자유 항목.

## 3. 구현과 소규모 검증

### 3.1 `tools/relabel_actions.py`

모드: `reach --k 4`(B2) · `dir_orig_mag_b2 --k 4`(B3') · `reach_carry --k 4`(B2-carry) · `carry_norot`(B1, 실행 안 함) · `cap`(구판 B3, 폐기). 원본 복사 후 `actions` 만 다시 쓰고 `data.attrs` 에 모드·정의·바뀐 step 수·원본 md5, demo attrs 에 `relabel_n_changed`(B2-carry 는 `relabel_range`)를 남긴다. 끝에 자동 검사: `obs/`·`states/`·`initial_state/` 와 그리퍼 열이 원본과 bit 동일 — **모든 모드 통과**. 회전 규약은 Mimic 라벨과 같은 base 좌표 R_{t+k}·R_tᵀ (Mimic 라벨과 실제 1 step 회전의 cos 0.87, 몸체 좌표는 −0.38 로 확인).

| 모드 | 바뀐 step | 비고 |
| --- | --- | --- |
| B2 reach4 | 207,979 / 207,979 | 행동 최대 절댓값 < 1 |
| B3' dir_orig_mag_b2 | 207,979 | 최대 0.558, 원본 노름 < 1e-6 인 step 0개 (Mimic 행동 노이즈 0.01) |
| B2-carry reach_carry | 72,186 (34.7%) | 닫힘 구간 밖 원본 bit 동일, 안은 B2 bit 동일 (위반 0) |

### 3.2 open-loop 재생 게이트 (`tools/replay_check.py` + `tools/plan_B_speed_metrics.py --replay_src`, gx10-3, 같은 시범 10개)

`replay_demos.py` 를 그대로 돌리며 매 step 상태를 기록하고, 원본 시범과 같은 시점의 손끝 위치 편차를 잰다. (구현 중 `replay_demos.py` 가 CPU 행동 텐서 하나를 제자리에서 바꿔 넘겨 기록 행동이 덮이는 버그를 찾아 `.copy()` 로 고쳤다. 그 전 기록은 행동을 데이터셋 값으로 바꿔 분석했다. 상태·성공 기록은 처음부터 맞았다.)

| 라벨 | 재생 성공 | dev50 중앙 (최대) | dev_lift 중앙 (최대) | devmax 중앙 | 접근 step | 들기→상자 위 | 운반 속도 m/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 원본 (대조) | 9/10 | 0.00 (0.00) cm | 0.03 (0.40) | 1.1 | 227 | 91.5 | 0.0119 |
| **B2** | 4/10 | **1.33 (6.73)** | **6.54 (37.3)** | 13.9 | 243 | 83 | 0.0099 |
| **B3'** | 0/10 | 3.05 (33.6) | 13.1 (34.2) | 31.4 | — | — | — |
| **B2-carry** | 8/10 | 0.00 (0.00) | 1.58 (2.13) | 8.2 | 227 | 92 | 0.0100 |
| B1 (참고) | 5/10 | 0.00 (0.00) | 0.03 (0.40) | 35.7 | 227 | 95 | 0.0110 |
| 구판 B3 (참고) | 0/10 | 5.46 (14.1) | 19.4 (35.2) | 36.1 | 248 | — | — |

- **해석:** 이 재생은 "라벨이 기록된 상태에 묶여 있는가"를 보는 버그 점검이다. 좌표계·부호 버그면 접근 초반(t=50)부터 수십 cm 가 벗어난다 — B2 는 1.33 cm 로 버그가 아니다. B2 의 dev_lift 6.5 cm 는 라벨 크기가 줄어 접근 궤적이 원본에서 밀린 것이다(표의 접근 step 243 대 227 은 시범 묶음이 달라 비교에 쓰지 않는다. 같은 5개끼리 짝 비 중앙값은 0.99 다). B3' 는 원본 방향(지연 보정 방향)에 작은 크기를 붙여 open-loop 로는 따라가지 못한다(폐루프 정책은 상태를 보고 움직이므로 판정 근거가 아님). B2-carry 는 접근이 원본 그대로라 들기 시점 편차 1.6 cm.
- **표의 접근 step · 들기→상자 위 · 운반 속도는 그 단계까지 간 시범만의 중앙값이다** (원본 10 · 10 · 10개, B2 5 · 4 · 6개). 그래서 행끼리 그대로 비교하지 않는다. 원본 0.0119 와 B2 0.0099 를 그대로 비교한 "−17%" 는 쓰지 않는다. 같은 시범끼리 보면 운반은 약 6–9% 느리다: B2 짝 비 중앙값 0.915(6개, 같은 6개의 원본 중앙값 0.0105 → 0.0099), B2-carry 0.911(10개). 감속은 혼입 요인이라 `carry_speed` 로 보고한다(prereg §3 · §6 H_B 주).
- 원자료: `results/plan_B/replay_gate_{orig,carrynorot,reach4,cap,dirorig_magb2,reach4carry}.csv`, 합본 `replay_gate_all.csv`, 기록 `~/IsaacLab_RS/logs/rs_a2_runs/plan_B/replay_sub10_*.hdf5`.

### 3.3 B1 을 실행하지 않는 이유

B1 데이터는 운반 중 관측(eef_quat·트럭 자세)이 계속 도는데 회전 라벨은 0 이다. B1 정책이 시험 때 만들 "손목을 들 때 자세로 고정한" 운반 상태를 만들어 다른 시범 400개의 운반 상태 49,804개에 대한 최근접 거리(23-D 정책 관측, 표준화)를 쟀다: 실제 상태 p50 1.32 · 손목 고정 상태 p50 1.84 (**×1.40**; 들 때 이후 회전 15° 미만 ×1.06, 45–90° ×1.69). B1 은 실패해도 "분포 밖"으로 설명되어 회전 원인의 반박 근거가 되지 않는 비대칭 검정이고, 같은 원인은 A-T(생성 단계에서 상태와 행동을 함께 바꿈)가 깨끗하게 겨냥한다. open-loop 에서도 5/10.

### 3.4 정지 라벨 (B2 의 감속·멈춤 위험)

| | 정지 라벨 | 단계 분포 | 연속 길이 p50 / p90 / max |
| --- | --- | --- | --- |
| 사람 | 20.5% | 접근 52% · 그리퍼 전환 ±8 step 23% | 6 / 14 / 27 |
| 원본 Mimic | 0.8% | 접근 70% | 1 / 1 / 11 |
| B2 | 7.4% | 접근 57% · 그리퍼 ±8 24% · 끝 4 step 7% | 2 / 6 / 26 |

### 3.5 학습 스모크

`train_seed.py --seed 101 --dataset .../kgh_mimic_500_carrynorot.hdf5 --name _plan_B_smoke_carrynorot --epochs 20` (A6000, 10/29 14:00) → `finished run successfully`, `model_epoch_20.pth`. 재라벨 파일은 모드와 상관없이 키·dtype·shape 가 원본과 같아 그대로 읽힌다. 성능은 보지 않았다.

## 4. 조작 확인 게이트 (학습 전, 통과 못 하면 학습하지 않음)

행동 지표(`rot60`·`carry_rot_cmd`·`carry_pos_cmd`)는 **처치가 들어갔는지 확인하는 게이트로만** 쓰고 판정에는 쓰지 않는다 (#7: 처치 정의로 저절로 줄어드는 값, prereg §4). 게이트를 못 넘을 때의 처리는 prereg §3 데이터 게이트의 처리와 같다.

| 게이트 | B2 | 측정 | B3' | 측정 | B2-carry | 측정 |
| --- | --- | --- | --- | --- | --- | --- |
| 관측·상태·초기 상태·그리퍼 bit 동일 | 필수 | 통과 | 필수 | 통과 | 필수 | 통과 |
| 운반 라벨/실제 4 step 크기 비 | 0.8–1.2 | 1.00 | B2 와 같음 | 1.00 | 0.8–1.2 | 1.00 |
| 운반 라벨 방향 vs 실제 4 step (cos) | ≥ 0.9 | 1.00 | 원본과 같음 (0.59) | 0.59 | ≥ 0.9 | 1.00 |
| 위치·회전 노름 분포 vs B2 (중앙값·p90) | — | — | ±1% | **0.00%** (위치 0.0263·0.0545, 회전 0.0431·0.1186) | — | — |
| step 별 방향 vs 원본 (cos) | — | — | = 1 | **1.000** (최소 1.000) | — | — |
| `carry_rot_cmd` · `carry_pos_cmd` (라벨) | ≤ 0.10 · ≤ 0.07 | 0.091 · 0.055 | B2 와 같은 분포 | 통과 | — | — |
| `carry_yaw_net` (상태, 바뀌면 안 됨) | 38.4° | 38.4° | 38.4° | 38.4° | 38.4° | 38.4° |
| 정지 라벨 | ≤ 10% | 7.4% | — | — | — | — |
| 재생 dev50 중앙 (버그) | ≤ 2 cm | 1.33 | 기록만 | 3.05 | ≤ 2 cm | 0.00 |
| 재생 dev_lift 중앙 (시간 밀림) | ≤ 10 cm, 기록 | 6.54 | 기록만 | 13.1 | ≤ 3 cm | 1.58 |

## 5. 판정 — prereg §5.2 (B2) · §5.3 (B3') 를 따른다

- 비교: B2 는 baseline 대비 400쌍(평가 seed 42+43, A6000), B3' 는 B2 와 직접 400쌍(평가 seed 42+43). B2 의 ①–④ 는 prereg §5.2, B3' 의 동등 · 차이 · 구분 불가는 prereg §5.3, 해석은 prereg §6 의 H_B 표와 'H-A 와 H_B 를 함께 읽기' 를 따른다.
- **최소 예측(B2): +4%p** (prereg §3 표, §5.2 ② 의 기준).
- **상태 매개 지표로만 기전 비교** (prereg §4). 아래는 예측 방향만이다. 띠와 문턱은 prereg §5.1 null 띠 · §5.2 ③ 을 따른다.

| 상태 지표 | baseline (s101 / s111) | B2 예측 방향 |
| --- | --- | --- |
| 운반 창 상자 밂 `push_carry` (= 귀속 충돌, prereg §4) | 10 / 9 | ↓ |
| `box_push` (rollout 전체 상자 5 cm 이상 밂, prereg §4) | 22 / 24 | ↓ |
| `low_approach` (approach_z < 0.15 m 건수, prereg §4) | 15 / 13 | ↓ (적어도 늘지 않음) |
| `carry_yaw_net` (전체 중앙값) | 40.0° / 39.9° | **유지** — A-T 와 가르는 1차 지표, 확인 조건은 prereg §6 'H-A 와 H_B 를 함께 읽기' |
| 운반 손끝 속도 `carry_speed` (전체 중앙값) | 0.0106 / 0.0106 m/step | ↓ — 감속 혼입, 차를 함께 보고 (prereg §6 H_B 주) |
| 들기→상자 위 `lift_to_over` (중앙값) | 86 / 83.5 step | 기록 (감속이면 ↑) |
| 접근 `approach_steps` (중앙값) | 228 / 235.5 step | 기록 (B2 는 접근 라벨도 작아져 ↑ 예상) |
| 시간 초과 (horizon, 실패 = 전부) | 37 / 31 | 기록 |

- **감속 혼입:** B2 는 운반을 늦춘다(재생에서 같은 시범끼리 약 6–9%, 3.2). 감속만으로도 개선될 수 있으므로 B2 결과에는 `carry_speed` 차를 함께 보고한다(prereg §3 · §6 H_B 주). 미리 정한 속도 층 분석 도구는 없으므로 층별 Δ 로 귀속하지 않는다. B3' 는 B2 와 `carry_speed` 가 같은지 확인한다(같은 크기 분포라 같아야 함).
- **들기 전 실패:** B2 는 접근 라벨도 바꾼다. 들기 전 실패가 prereg §5.2 의 음성 대조 조건(null 띠 3.03/100 — json 값 — 밖으로 늘고 그리고 짝 McNemar 양측 p < 0.05, `--bundle approach,grasp`)에 걸리면 '음성 대조 위반'으로 적는다(prereg §5.2). 떼어 보려면 B2-carry 를 쓴다(prereg §7 큐 밖) (교차 점검 §4: "성공률 Δ≈0, 들기 이후↓·들기 전↑").

## 6. 예상 실패 이동 (방향만, v3)

- 들기 이후 실패(운반+배치)와 운반 창 충돌 몫이 줄어든다. 충돌 몫 귀속은 prereg §4 의 귀속 (ii).
- 접근·파지는 유지 또는 소폭 증가(B2 의 접근 감속·정지 라벨). 들고 멈춤·freeze 감시.
- 구판의 "box_push ≤ 15%" 같은 하위 수치 예측은 근거가 약해 뺐다 (#17: 데이터 자체가 16.2%, 잡음 범위 14–28%).

## 7. 실행 명령 · 계산량

```bash
cd ~/IsaacLab_RS
# 데이터 (이미 만들어 둠 — 다시 만들 때)
python -I ~/robotics-simulation-team12/assignment2_toy2box/tools/relabel_actions.py --mode reach --k 4 --src ./datasets/team12_kgh/kgh_mimic_500.hdf5 --out ./datasets/team12_kgh/plan_B/kgh_mimic_500_reach4.hdf5
```

```bash
python -I ~/robotics-simulation-team12/assignment2_toy2box/tools/relabel_actions.py --mode dir_orig_mag_b2 --k 4 --src ./datasets/team12_kgh/kgh_mimic_500.hdf5 --out ./datasets/team12_kgh/plan_B/kgh_mimic_500_dirorig_magb2.hdf5
```

```bash
python -I ~/robotics-simulation-team12/assignment2_toy2box/tools/relabel_actions.py --mode reach_carry --k 4 --src ./datasets/team12_kgh/kgh_mimic_500.hdf5 --out ./datasets/team12_kgh/plan_B/kgh_mimic_500_reach4carry.hdf5
```

학습·평가는 prereg §7 직렬 큐와 공용 `run_baseline.sh` 를 따른다. 평가 seed 는 다섯째 인자(`<tag> <data> <학습 seed> [eval epoch=2000] [평가 seed=42]`), 출력 폴더는 환경 변수 `OUTDIR`(처치 팔은 `results/runs` 권장)이다. 단계마다 종료 코드(0 성공 · 5 평가 실패 · 6 분류 실패)를 확인한다(prereg §7). 태그 `reach4` · `dirorig_magb2` · `reach4carry`, 데이터는 위 경로.

```bash
# 예: B2 학습 + 평가 seed 42 (seed 111 도 같게), 그 뒤 평가 seed 43 만
A2=~/robotics-simulation-team12/assignment2_toy2box
OUTDIR=$A2/results/runs bash $A2/tools/run_baseline.sh reach4 ./datasets/team12_kgh/plan_B/kgh_mimic_500_reach4.hdf5 101
OUTDIR=$A2/results/runs bash $A2/tools/run_baseline.sh reach4 ./datasets/team12_kgh/plan_B/kgh_mimic_500_reach4.hdf5 101 2000 43
```

```bash
# 판정 (prereg §5.2 · §5.3). baseline seed 43 은 큐의 run_baseline.sh 출력(기본 OUTDIR = results/baseline)
R=$A2/results
# B2 대 baseline (짝 약 400쌍, 표적 묶음 · 상태 매개 지표 포함)
python -I $A2/tools/paired_compare.py --require_carry --label_a baseline --label_b B2 --json $R/runs/judge_B2.json \
  --a $R/baseline/base_kgh500_s{101,111}_e2000_seed{42,43}.csv --b $R/runs/base_reach4_s{101,111}_e2000_seed{42,43}.csv
# 음성 대조 (들기 전 실패 묶음)
python -I $A2/tools/paired_compare.py --require_carry --bundle approach,grasp --label_a baseline --label_b B2 \
  --a $R/baseline/base_kgh500_s{101,111}_e2000_seed{42,43}.csv --b $R/runs/base_reach4_s{101,111}_e2000_seed{42,43}.csv
# baseline e1900 민감도 (평가 seed 42 만)
python -I $A2/tools/paired_compare.py --require_carry --label_a baseline_e1900 --label_b B2 \
  --a $R/noise_baseline/a6000_base_kgh500_s{101,111}_e1900_seed42.csv --b $R/runs/base_reach4_s{101,111}_e2000_seed42.csv
# B3' 대 B2 (귀속, 짝 약 400쌍)
python -I $A2/tools/paired_compare.py --require_carry --label_a B2 --label_b "B3'" --json $R/runs/judge_B3p.json \
  --a $R/runs/base_reach4_s{101,111}_e2000_seed{42,43}.csv --b $R/runs/base_dirorig_magb2_s{101,111}_e2000_seed{42,43}.csv
```

```bash
# 평가가 끝난 뒤 트랙 B 상태 지표 (속도·접근·시간 초과)
python -I ~/robotics-simulation-team12/assignment2_toy2box/tools/plan_B_speed_metrics.py <B2 기록 s101.hdf5> <B2 기록 s111.hdf5> --csv ~/robotics-simulation-team12/assignment2_toy2box/results/plan_B/speed_b2_rollouts.csv
```

- 계산량(A6000): 생성 없음. 팔 하나 = 학습 2 seed 동시 약 45분 + 평가 2개 동시 약 65분 ≈ 2 h. B2 · B3' 는 평가 seed 43 이 각각 약 65분 더 든다(prereg §7 큐). B2-carry 약 2 h (prereg §7 큐 밖).

## 8. 위험 · 대안 · 규칙 확인

- **감속:** 라벨 평균 노름이 위치 0.064 → 0.030, 회전 0.121 → 0.053 (B2·B3' 같음). 재생에서 같은 시범끼리 운반이 약 6–9% 느리다(짝 비 중앙값 B2 0.915 · B2-carry 0.911, `results/plan_B/replay_gate_all.csv`). 접근 step 은 같은 시범끼리 차가 거의 없다(짝 비 0.99). 감속은 혼입 요인이라 `carry_speed` 차를 함께 보고한다(prereg §3 · §6 H_B 주).
- **정지 라벨 7.4%:** 접근·그리퍼 전환 부근의 짧은 정지(3.4). 맴돎·멈춤 실패 감시.
- **B2 가 회전 상태도 바꿀 가능성:** 데이터 상태는 같지만 폐루프에서 정책이 덜 돌 수 있다. 그래서 `carry_yaw_net` 를 A-T 와 가르는 1차 지표로 둔다(prereg §6 'H-A 와 H_B 를 함께 읽기' 의 확인 조건).
- **B3' 의 open-loop 0/10:** 원본 방향에 작은 크기를 붙이면 지연 보정이 맞지 않는다. 폐루프 판정과 무관하지만, B3' 가 크게 나빠지면 "방향 원본 + 작은 크기 조합 자체의 손해"일 수 있음을 한계로 적는다.
- 규칙: 환경·판정·종료 조건 불변, 시범 수·학습량·체크포인트 규칙 baseline 과 같음. 재라벨 데이터의 성격은 1절 끝(#20)대로 보고서에 명시.
- 공용 도구 메모(고치지 않음): `play_record.py` 의 행동 기록에 `.copy()` 가 없다 — play.py 는 매 step 새 CUDA 텐서를 넘겨 지금은 문제없지만, 텐서를 재사용하는 스크립트에 붙이면 `replay_check.py` 와 같은 버그가 난다.

## 9. 남은 질문

1. B2 의 k=4 는 학습 전에 고정했다(이득 0.22). 결과 뒤 바꾸지 않는다.
2. B2-carry 를 돌릴 슬롯이 있는지 — prereg §7 큐에 없다. 돌리면 prereg 밖 탐색으로 따로 보고한다.
3. A-T 와 B2 가 둘 다 개선되면 prereg §6 의 'H-A 와 H_B 를 함께 읽기' 로 쓴다. 확인 조건도 거기 있다.
