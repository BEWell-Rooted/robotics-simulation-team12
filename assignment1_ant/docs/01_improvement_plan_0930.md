# 01. 2차 개선 계획 — 1차 세션(0929) 결과 기반, 논문 아이디어 정리

- 작성: 2026-09-30 (수), Cowork에서 `log_0929.md`·`00_codebase_notes.md`·`figures/`를 읽고 작성
- 독자: 워크스테이션 Claude Code. 핸드오프 브리프(`과제1_Ant_핸드오프_브리프.md`)와 운영 규약(5장)은 그대로 유효
- 남은 시간: 10/6(화) 23:59 제출까지 약 6일 → 새 학습 변형은 **최대 3개**로 제한
- 모든 항목은 ⚠ Claude 제안 상태이며, 착수 전 형근이 우선순위를 확정한다

---

## 0. 한 줄 요약

1차 결과의 가장 큰 구멍은 **메시 평지 낙상(V2 67%, V1 85%)** 이다.
조교의 unseen 지형은 메시일 가능성이 높고 평가 지표는 첫 에피소드 누적 보상이므로, 낙상을 줄이는 것이 점수에 가장 직접적이다.
따라서 2차는 "원인 진단 → torso 높이 마진 확보 → 접촉 관측 분포 변화에 둔감한 정책" 순서로 진행하고, 여유가 있으면 비대칭 critic·대칭성 증강을 독창성 카드로 붙인다.

---

## 1. 1차 결과에서 확인된 개선 대상

### 1.1 P-A. 메시 평지에서 전 모델 낙상 급증 (최우선)

- 해석적 평면 대비 메시 평지에서 낙상률이 크게 오른다: baseline 7%→40%, V1 24%→90%, V2 11%→67%.
- 저마찰 메시 평지(FlatIce μ0.2)에서는 baseline 낙상이 5%로 낮다 → 메시 자체보다 **메시 + 높은 마찰** 조합이 의심된다.
- 낙상 직전 torso z가 0.31~0.36으로, 정책들이 종료 임계(0.31) 바로 위로 낮게 걷는다.
- V4(수직 속도·roll/pitch 속도 페널티)와 V5(평지 비율 증가)는 모두 개선하지 못했다 → hopping 단독 원인, 평지 노출 부족 가설은 기각됨.

### 1.2 P-B. 지형 간 트레이드오프

- V2는 12종 평균 1위지만 Stairs 7.0으로 전진을 못 한다.
- V5는 계단을 학습해 Stairs 47.7을 얻었지만 나머지 지형 점수가 전반적으로 떨어졌다.

### 1.3 P-C. 정책이 지형을 "보지" 못함

- actor 입력은 60차원 고유감각(proprioception)뿐이고, 호환성 때문에 높이 스캔·관측 히스토리를 추가할 수 없다.
- 지형 정보는 발 wrench 24차원에 간접적으로만 들어온다.

### 1.4 P-D. 파인튜닝(V3) 붕괴

- baseline에서 resume하자 직후 보상이 0.7로 붕괴했고, 3000 iter에서도 24.2에 머물렀다.
- 제출 후보는 아니지만 발표에서 "warm start가 왜 실패했나"를 한 줄로 설명할 근거가 있으면 좋다.

---

## 2. 원인 진단 먼저 (P-A, 예상 1.5h, 학습 없음)

메시 평지 낙상의 원인 후보는 세 가지이며, 각각 기존 체크포인트로 평가만 돌려서 가를 수 있다.

### 2.1 가설 H1 — 높이 마진 부족 (임계값 효과)

- 정책은 잘 걷고 있는데, 메시 접촉의 미세한 높이 차이(contact/rest offset 등)로 torso가 0.31 아래로 잠깐 내려가 종료되는 경우.
- 진단: Eval-Flat에서 `minimum_height`만 0.31 → 0.28 → 0.25로 낮춰 평가한다.
- 판정: 낙상률이 급감하고 전진 거리가 유지되면 H1이 주원인이다.
- 추가 기록: 종료 순간 `up_proj`(몸통 기울기)를 함께 저장해, "뒤집혀서 종료"와 "똑바로 선 채 낮아서 종료"를 구분한다.

### 2.2 가설 H2 — 발 wrench 관측의 분포 이동

- 메시 접촉은 해석적 평면보다 접촉력이 튀는(spike) 경향이 있고, 관측 24차원이 스케일 0.1만 곱한 원시 wrench라 정책 입력이 학습 분포 밖으로 나갈 수 있다.
- 진단 1: plane vs 메시 평지에서 `feet_body_forces` 24차원의 분포(평균·p99·최대)를 기록해 비교한다.
- 진단 2: 메시 평지 평가 중 wrench 관측만 clip(예: plane의 p99)하거나 0으로 대체해 낙상률 변화를 본다 (진단용 eval 전용 cfg, 제출과 무관).
- 판정: clip만으로 낙상이 크게 줄면 H2가 주원인이다.

### 2.3 가설 H3 — 고마찰 메시에서 발 걸림

- Ant는 발을 끌며 걷는 성분이 있어, 마찰이 높으면 메시 삼각형 경계(내부 edge의 ghost contact)에서 발이 걸려 넘어질 수 있다.
- 진단: 메시 평지 μ ∈ {0.2, 0.5, 0.8, 1.0, 1.5} × plane 같은 μ 격자로 낙상률 곡선을 그린다.
- 판정: 메시에서만 μ에 따라 낙상이 가파르게 오르면 H3 성분이 있다.
- 이 격자 그림은 그대로 발표 분석 슬라이드 재료가 된다.

### 2.4 진단 산출물

- `results/diag_meshflat.csv`, `docs/figures/diag_threshold.png`, `docs/figures/diag_mu_grid.png`
- `docs/log_0930.md`에 H1/H2/H3 판정과 근거 수치를 기록한다.
- 3장의 V6/V7 중 무엇을 먼저 돌릴지는 이 판정으로 정한다 (H1 → V6 우선, H2 → V7 우선, 둘 다면 V6+V7 결합).

---

## 3. 논문에서 뽑은 개선 아이디어

모든 아이디어는 D0 안전 모드(actor·critic 입력 60, [400,200,100] elu, obs norm off, state dict 완전 동일)를 지킨다.
예외는 3.4(비대칭 critic)이며, 학습 후 critic 교체 절차로 호환성을 복구한다.

### 3.1 V6 — 제약을 종료로: torso 높이 마진 학습 (P-A/H1 대응, 1순위 후보)

- 근거 논문: Chane-Sane et al., "CaT: Constraints as Terminations for Legged Locomotion Learning" (IROS 2024).
- 논문 아이디어: 제약 위반을 보상 페널티 대신 **확률적 종료**로 넣는다 — 위반 정도에 비례한 확률로 이후 보상을 끊어, 튜닝할 가중치 없이 제약을 지키게 만든다.
- 적용: 학습 환경에서만 "지형 상대 torso 높이 ≥ h_safe(예: 0.40)" 제약을 CaT식으로 건다.
- 구현 1: 각 env에서 torso 아래 지형 높이를 구한다 (지형 origin z 근사 대신 `RayCasterCfg` 1점 높이 센서 또는 terrain mesh 질의 — 관측이 아닌 보상/종료 계산에만 사용).
- 구현 2: 위반량 `v = max(0, h_safe - z_rel)`에 대해 종료 확률 `p = p_max · clip(v / v_max, 0, 1)` (논문 기본 p_max 0.05 수준에서 시작), 이 종료는 `time_out=False`로 처리해 가치가 부트스트랩되지 않게 한다.
- 구현 3 (단순 대안): CaT 대신 `-w · v²` 페널티 항 하나. 비교를 위해 둘 중 하나만 먼저 돌리고, 시간이 남으면 다른 하나를 ablation으로 돌린다.
- 기존 절대 z 종료(0.31)는 학습에서도 유지한다 — 조교 환경 조건과 동일하게 두기 위함.
- 기대 효과: 정책이 임계 위로 여유를 두고 걷게 되어, 메시·지형 높이 차이로 인한 거짓 종료에 강해진다.
- 발표 포인트: "종료 조건이 절대 z일 때 torso 높이 마진이 곧 강건성" → 진단(H1)과 해법(V6)이 한 줄로 이어진다.
- 베이스: V2 커리큘럼 cfg 위에 얹는다 (현재 1위이므로 단일 변수 비교가 된다).

### 3.2 V7 — 접촉 관측 강건화: 관측 노이즈 + 립시츠 제약 (P-A/H2 대응)

- 근거 논문 1: Chen et al., "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies" (LCP, arXiv 2024).
- LCP 아이디어: 정책 출력의 입력에 대한 gradient norm을 페널티로 넣어(gradient penalty) 정책을 매끄럽게 만든다 — 입력이 조금 튀어도 행동이 크게 튀지 않는다.
- 근거 논문 2: Mysore et al., "Regularizing Action Policies for Smooth Control with Reinforcement Learning" (CAPS, ICRA 2021).
- CAPS 아이디어: 시간적 smoothness(연속 상태의 행동 차이)와 공간적 smoothness(근처 상태의 행동 차이) 두 정칙화 항.
- 적용 A (구조 변경 없음): 학습 시 `feet_body_forces` 항에만 관측 노이즈를 켠다 (`enable_corruption=True`, 해당 항에 곱셈형+덧셈형 노이즈). 평가 cfg는 원본 그대로.
- 적용 B (알고리즘 변경): rsl_rl `PPO`를 상속한 `PPOLipschitz`에서 surrogate loss에 `λ · E[‖∇_obs μ(obs)‖²]`를 추가한다 (LCP 기본 λ 범위 0.001~0.01에서 시작). 네트워크 구조는 그대로라 체크포인트 호환성에 영향이 없다.
- 순서: A만 먼저 돌려 효과를 보고, 부족하면 A+B.
- 진단 H2가 음성이면 V7은 보류한다.

### 3.3 V8 — 대칭성 증강 (독창성 카드, 구조 변경 없음)

- 근거 논문: Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies" (ICRA 2024).
- 논문 아이디어: 로봇·태스크의 대칭 변환으로 rollout 데이터를 증강하거나(data augmentation), 대칭 행동 차이를 손실로 넣어(mirror loss) 샘플 효율과 보행 균형을 높인다.
- 적용: Ant의 목표 방향은 +x이므로 **xz 평면 좌우 반사(y → −y)** 가 태스크를 보존하는 대칭이다.
- 매핑: 좌우 다리 관절 교환 + hip 관절 부호 반전, 관측에서는 lin_vel_y·ang_vel_x·ang_vel_z·yaw·roll·angle_to_target 부호 반전, 발 wrench는 좌우 발 교환 + y성분·x/z 토크 부호 반전.
- Isaac Lab에는 `isaaclab_rl.rsl_rl`에 `RslRlSymmetryCfg`(PPO cfg의 `symmetry_cfg`)가 있어 증강 함수만 작성하면 된다 — 필드 이름과 rsl-rl-lib 3.0.1 지원 여부는 코드로 확인할 것.
- 반사 매핑이 맞는지 먼저 검증한다: 한 rollout의 관측·행동을 반사한 뒤 정책에 넣어, 원본과 반사 결과가 대칭인지 단위 테스트.
- 기대 효과: 한쪽으로 치우친 보행·회전 편향 감소, 지형 불규칙성에 대한 좌우 대칭 대응. 효과가 작더라도 "구조를 안 바꾸는 사전지식 주입"으로 독창성 설명이 된다.

### 3.4 V9 — 비대칭 actor-critic + critic 교체 (P-C 대응, 시간 남으면)

- 근거 논문 1: Pinto et al., "Asymmetric Actor Critic for Image-Based Robot Learning" (RSS 2018).
- 근거 논문 2: Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL 2021) — legged_gym의 critic 특권 관측 구성.
- 아이디어: critic에만 특권 정보(지형 높이 스캔, 실제 마찰계수, 지형 상대 torso 높이)를 줘 가치 추정을 정확하게 하고, actor는 60차원 그대로 둔다.
- 구현: `obs_groups = {"policy": ["policy"], "critic": ["policy", "privileged"]}` (rsl-rl-lib 3.0.1 지원 ✔, `00_codebase_notes.md` 9장).
- 호환성 복구 절차 (필수): 학습 후 critic을 60차원 입력 [400,200,100] critic으로 교체한다.
- 교체 방법: 학습된 비대칭 critic의 가치를 타깃으로, 60차원 critic을 rollout 데이터에서 회귀(value regression)한다. play는 actor만 쓰므로 회귀 정확도는 점수에 영향이 없고, 목적은 strict 로드 통과와 "의미 있는 critic" 유지다.
- optimizer state도 새 네트워크 기준으로 다시 만들어 저장한다 (`runner.load`가 optimizer까지 로드하므로).
- 검증: 교체한 `.pt`를 `--task Isaac-Ant-v0`의 `play.py`로 로드해 에러 없이 돌아가는지 확인한다. 이 검증 없이 제출 후보로 올리지 않는다.
- 참고 (채택 안 함): RMA(Kumar et al., RSS 2021), teacher-student(Lee et al., Science Robotics 2020)는 actor에 히스토리 인코더가 필요해 D0 호환성을 깨므로 제외.

### 3.5 저비용 보조 조정 (변형 추가 없이 V6/V7에 같이 넣을 후보)

- 엔트로피 계수: baseline은 0.0이다. legged_gym 계열은 0.01을 쓰며, 탐색이 늘어 거친 지형 적응에 도움이 될 수 있다 — 단 V6과 동시에 바꾸면 단일 변수 비교가 깨지므로, 넣을 거면 V2 재학습 대조군에도 같이 넣는다.
- 외란·초기 상태 랜덤화: 주기적 push(수평 속도 교란), 초기 자세 랜덤화(`reset_root_state_uniform` 범위 확대)는 회복 동작을 학습시켜 낙상을 줄인다 (Rudin et al. 2021, Tobin et al. IROS 2017의 도메인 랜덤화 원칙). 평가 조건은 원본 유지.
- 지형 믹스에 plane 근사 메시 평지와 저·고마찰 평지를 조금 더 섞는다. 단 V5에서 평지 비율 증가만으로는 효과가 없었으므로 단독 변형으로 돌리지 않는다.

### 3.6 체크포인트 선택을 held-out 검증으로 (학습 없음, 바로 적용)

- 근거 논문: Cobbe et al., "Quantifying Generalization in Reinforcement Learning" (ICML 2019) — 학습 성능과 unseen 성능이 갈라지므로 held-out 레벨로 일반화를 측정해야 한다.
- 적용: 제출 체크포인트를 "마지막 iter"가 아니라 **held-out 평가 점수 최고 iter**로 고른다.
- 절차: 후보 run(V2 3 seed, V6/V7 결과)의 model_{1500,2000,2500,2999}.pt를 held-out 지형(Boxes, Rails + 3개 내외)에서 평가해 선택한다.
- 주의: 선택에 쓴 held-out 지형과 발표용 최종 보고 지형을 분리해야 선택 편향이 없다 — Boxes/Rails는 선택용, 보고용으로 held-out 1~2종(예: 경사+블록 혼합, 좁은 발판)을 새로 추가한다.

### 3.7 V3 붕괴 설명 (발표 한 줄용, 실험 선택)

- 근거 논문: Nikishin et al., "The Primacy Bias in Deep RL" (ICML 2022), Dohare et al., "Loss of Plasticity in Deep Continual Learning" (Nature 2024).
- 설명 후보: 평지에서 수렴한 정책은 행동 노이즈(std)가 작고 critic이 평지 가치에 과적합되어, 새 분포에서 탐색·적응이 막힌다.
- 확인 실험(선택, 1 run): resume 시 critic만 재초기화 + action std를 1.0으로 리셋한 V3'. 붕괴가 사라지면 설명이 뒷받침된다.
- 시간이 부족하면 실험 없이 "원인 추정"으로만 발표에 쓴다.

---

## 4. 실행 계획 (예상 소요, 날짜 배치는 형근이 함)

### 4.1 액션 아이템

- 진단 H1/H2/H3 (2장) + 그림 2장 — 1.5h (학습 없음)
- 체크포인트 선택 파이프라인 (3.6) + 보고용 held-out 1~2종 추가 — 1h
- V6 구현(지형 상대 높이 계산 + CaT 종료 또는 페널티) + 스모크 — 2h
- V6 학습 3000 iter × seed 2 — GPU 약 1.5h (2개 병행)
- V7-A 구현(힘 관측 노이즈) + 학습 seed 2 — 구현 0.5h + GPU 1.5h (H2 양성일 때만)
- V8 대칭 매핑 + 단위 테스트 + 학습 seed 2 — 구현 2h + GPU 1.5h (독창성 카드)
- V9 비대칭 critic + critic 교체 + 호환 로드 검증 — 3h + GPU 1.5h (여유 있을 때만)
- 전체 평가(12종 + 보고용 held-out) 재실행·표·그림 갱신 — 1.5h
- 최종 체크포인트 선택·호환성 검증(`--task Isaac-Ant-v0` 로드) — 0.5h

### 4.2 명명

- `v6_margin_{cat|pen}_s{seed}`, `v7_forcenoise_s{seed}`, `v7_lcp_s{seed}`, `v8_sym_s{seed}`, `v9_asym_s{seed}`
- 태스크: `Isaac-Ant-Rough-Margin-v0`, `Isaac-Ant-Rough-ForceNoise-v0`, 대칭·비대칭은 agent cfg 변형으로 등록

### 4.3 중단 규칙

- V6이 메시 평지 낙상을 V2 대비 10%p 이상 줄이지 못하고 12종 평균도 떨어지면, 제출은 V2(held-out 선택 체크포인트)로 확정하고 V6은 분석 결과로만 쓴다.
- 새 변형이 스모크에서 3회 이상 막히면 log에 상태를 적고 멈춘다 (브리프 운영 규약).

---

## 5. 발표 스토리 연결 (독창성 30점)

- 문제 발견: "같은 평지인데 메시로 바꾸면 낙상이 7%→40%" — 자체 평가 세트로 찾은 비자명한 실패 모드.
- 원인 분석: 임계값·마찰·관측 분포 진단 격자 (2장 그림).
- 해법: 제약-종료(CaT)로 높이 마진을 학습시키고, 필요 시 접촉 관측 강건화·대칭성 증강을 더함 — 모두 정책 입출력 규격을 바꾸지 않는 방법이라는 점이 조교 평가 조건에 맞춘 설계 의도.
- 검증: 학습 분포 / 선택용 held-out / 보고용 held-out 3단 분리.

---

## 6. 참고문헌

- Chane-Sane, E. et al. CaT: Constraints as Terminations for Legged Locomotion Learning. IROS 2024.
- Chen, Z. et al. Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies. arXiv 2024.
- Mysore, S. et al. Regularizing Action Policies for Smooth Control with Reinforcement Learning (CAPS). ICRA 2021.
- Mittal, M. et al. Symmetry Considerations for Learning Task Symmetric Robot Policies. ICRA 2024.
- Pinto, L. et al. Asymmetric Actor Critic for Image-Based Robot Learning. RSS 2018.
- Rudin, N. et al. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. CoRL 2021.
- Tobin, J. et al. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS 2017.
- Cobbe, K. et al. Quantifying Generalization in Reinforcement Learning. ICML 2019.
- Nikishin, E. et al. The Primacy Bias in Deep Reinforcement Learning. ICML 2022.
- Dohare, S. et al. Loss of Plasticity in Deep Continual Learning. Nature 2024.
- Kumar, A. et al. RMA: Rapid Motor Adaptation for Legged Robots. RSS 2021. (호환성 문제로 제외)
- Lee, J. et al. Learning Quadrupedal Locomotion over Challenging Terrain. Science Robotics 2020. (호환성 문제로 제외)
