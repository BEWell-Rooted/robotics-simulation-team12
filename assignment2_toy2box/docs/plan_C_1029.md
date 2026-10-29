# 트랙 C (담당 C) — 정책 구조와 폐루프 강건성: 가설·실험 기획 (2026-10-29, 교차 점검 반영판)

> 기준: `docs/diagnosis_1028.md`(진단·7절 잡음 기준선·8절 조교 대조군), `docs/survey_1028.md` §2.3·§6, 교차 점검 통합(1029) §1 #4·#8·#13·#18. 판정표·해석표는 prereg §5·§6 이 대신한다.
> **판정 수치는 공용 사전등록 `docs/prereg_1029.md` 를 따른다.** 이 문서는 트랙 C 의 설계·구현·최소 예측만 적는다. 숫자가 다르면 prereg 가 우선이다.
> 산출물: `tools/plan_C/{play_gripmix.py, attribute_c3.py, gmm_offline.py, train_override.py, run_plan_C.sh}`, `results/plan_C/`.

## 0. 요약

- **공용 구도에서 트랙 C 의 자리:** 원인 가설은 H-A 하나다(Mimic 상자 구간 SE(3) 재타깃팅이 만든 불필요한 운반 동작, 처치 A-T). 트랙 C 는 두 역할을 맡는다.
  - **C3 = 경쟁 원인 팔.** 경쟁 원인은 "그리퍼 소수 성분 샘플"이다. C3 는 위약과 같은 Spark 에서 200쌍으로 짝지어 비교한다. 이 설계는 H_C 를 '지지' 할 수만 있고 기각할 수는 없다(prereg §2·§5.4).
  - **C-1mode = 제출 가능한 근사 팔.** C3 가 위약 대비 '지지'일 때만 돌리고, A6000 이 부족하면 가장 먼저 뺀다(prereg §3). 한 요인 처치가 아니라서 H_C 해석에 쓰지 않고 제출 판단에만 쓴다(prereg §6 H_C1·§7).
- **제출 경로:** 조교는 `play.py`(미공개 seed)로 평가하므로 추론 시점 규칙(C3·위약)은 조교 평가에 실리지 않는다. 그래서 C3 는 진단 실험이다. `play.py` 는 체크포인트 안 config 로 정책을 다시 만들기 때문에, C-1mode(`algo.gmm.num_modes` 5→1)는 조교 평가에도 그대로 적용된다(스모크로 확인).
- **새로 잰 것 (1029):**
  1. 들기 직후 큰 회전은 모드 샘플링에서 오지 않는다. 혼합 평균/실행 비율이 0.99–1.00, 모드 간 퍼짐이 0.008–0.014 다.
  2. baseline 들기 이후 실패 49건을 공용 귀속 (ii)(`carry_metrics.py` 의 `attr_ii`)로 나누면 충돌 19 · 스스로 놓음 19 · 미끄러짐 8 · 기타 3 이다. 트랙 C 도구(`attribute_c3.py`)로는 충돌 17 · 비충돌 소수 열기 6 · 다수 열기 13 · 미끄러짐 8 · 기타 5 다. 두 도구는 2건만 다르다(§1).
  3. 같은 체크포인트를 Spark 에서 `sample` 로 돌리면 들기 이후 실패가 39건이고, 비충돌 소수 열기는 8건이다.
- **건수가 적다.** C3 판정 묶음(비충돌 스스로 놓음, prereg §5.4)은 200 rollout 중 12–19건이고, 그중 소수 열기는 6–8건이다. 트랙 C 의 예측 순감소는 약 3건이라 **예상 판정은 '미지지'(구분 불가)** 이고, 그러면 C-1mode 는 돌리지 않는다(§5).

## 1. 가설 (경쟁 원인)

**H_C:** 들기 이후 실패 중 일부는 GMM 이 매 step 혼합 성분 하나를 뽑을 때 소수 성분의 '열기'가 뽑혀서 생긴다. 그리퍼를 혼합 다수결로 정하면 그런 실패가 사라진다. H_C 가 맞다면 이 실패들은 운반 궤적(H-A)이 아니라 그리퍼 결정 규칙에서 오는 독립 원인이다.

근거:
- `results/plan_C/gmm_offline_summary.txt` 는 baseline s101·s111 e2000 을 기록 상태에 다시 넣어 계산한 값이다.
- `results/plan_C/c3_attribution_baseline_{a6000,spark}.txt` 는 트랙 C 도구(`attribute_c3.py`)의 귀속 (ii) 결과다. 아래 표가 그 값이다.

| | A6000 baseline (200) | Spark baseline `sample` (gx10-3 s101 + gx10-2 s111, 200) |
| --- | --- | --- |
| 들기 이후 실패 | 49 | 39 |
| (ii) 충돌 | 17 | 10 |
| (ii) 비충돌 스스로 놓음 — 소수 열기 (P(open) < 0.5) | **6** (P = 0.001–0.44) | **8** (P = 0.006–0.49) |
| (ii) 비충돌 스스로 놓음 — 다수 열기 | 13 | 4 |
| (ii) 미끄러짐 | 8 | 11 |
| (ii) 기타(놓침 없음) | 5 | 6 |
| 민감도 (i) ±5 step: 충돌 · 소수 · 다수 · 미끄러짐 · 기타 | 3 · 11 · 16 · 14 · 5 | 1 · 12 · 7 · 13 · 6 |
| 민감도 (iii) 전체: 충돌 · 소수 · 다수 · 미끄러짐 · 기타 | 28 · 2 · 10 · 4 · 5 | 15 · 5 · 3 · 10 · 6 |

- 트랙 C 도구의 귀속 규칙 (ii)는 다음과 같다(`tools/plan_C/attribute_c3.py`, 위에서부터 처음 맞는 하나).
  1. 운반 창 [t_lift, t_loss+5] 에서 상자가 t_lift 위치보다 xy 5 cm 넘게 밀렸거나 70° 넘게 기울었으면 충돌.
  2. 아니면, 놓친 직후 명령(행동 t_loss)이 '열기'이면 스스로 놓음. 그 열기 구간 첫 step 의 P(open) < 0.5 면 소수, 아니면 다수.
  3. 닫힌 채 놓쳤으면 미끄러짐.
  4. 그 밖은 기타.
- 공용 `carry_metrics.py` 의 `attr_ii`(prereg §4)는 A6000 baseline 49건을 충돌 19 · 스스로 놓음 19 · 미끄러짐 8 · 기타 3 으로 나눈다. 트랙 C 도구와는 2건만 다르다. 트럭을 끝까지 쥔 채 상자를 민 s101 rollout_66 · s111 rollout_70 을 공용 도구는 충돌, C 도구는 기타(놓침 없음)로 센다. 스스로 놓음(소수 6 + 다수 13 = 19)과 미끄러짐 8 은 같다. 판정은 공용 도구로 하고(prereg §5.4), C 도구는 소수·다수 나눔을 보고하는 데만 쓴다.
- RNN 은닉 상태 초기화(10 step 마다)는 원인이 아니다. 위상별 회전 크기 비율이 0.99–1.00 이다.

## 2. 한 요인 · 대조 · 처리 · 고정 조건

| | C3 (경쟁 원인) | 위약 (C3 의 짝) | C-1mode (제출 가능한 근사 팔) |
| --- | --- | --- | --- |
| 그리퍼 결정 | 혼합 다수결: P(open) = Σ_k w_k·1[μ_k,grip ≥ 0] ≥ 0.5 이면 열기 | 별도 `torch.Generator`(seed 4242)로 Bernoulli(P(open)) 를 다시 뽑음 — 주변 분포는 baseline 과 같고 난수만 새로 | 학습 config `algo.gmm.num_modes` 5 → 1 (평가 때 결정적 평균) |
| 비교 | **위약과 직접 200쌍, 같은 Spark** (s101 은 gx10-3, s111 은 gx10-2) | — | A6000 baseline e2000 과 400쌍, 평가 seed 42 + 43 (prereg §5.2, ② 첫 조건 없음) |
| 고정 | 같은 체크포인트(A6000 체크포인트와 md5 확인), 팔 6차원(기본 샘플 그대로, 기본 난수 소비 1회 그대로), RNN 초기화 주기 10, 평가 seed 42 · 100회 · horizon 800 | 같음 | 데이터 `kgh_mimic_500`, 2000 epoch · batch 100 · lr 1e-3 · seq 10 · LSTM 400×2, 마지막 체크포인트, 학습 seed 101·111, 평가 seed 42 + 43 · 100회 · horizon 800, A6000 |
| 한 요인인가 | 예 (그리퍼 결정 규칙 하나) | 예 (그리퍼 난수만) | **아니다** — 팔 6차원도 단일 가우시안 평균이 되고 학습 목적함수도 바뀐다(#13). 그래서 H_C 해석에 쓰지 않고 제출 판단에만 쓴다(prereg §6 H_C1·§7) |

## 3. 구현과 스모크 검증

| 파일 | 내용 | 검증 |
| --- | --- | --- |
| `tools/plan_C/play_gripmix.py` | `play_record.py` 위의 runpy 래퍼. `RNNGMMActorNetwork.forward` 를 감싸 기본 샘플은 그대로 한 번 뽑고, 그리퍼 차원만 `majority` / `mean` / `resample`(위약) 규칙으로 덮어쓴다. `RolloutPolicy.start_episode` 로 rollout 경계를 잡는다. 바뀐 step 마다 rollout · step · 단계(approach / grasp / transport / place / released) · P(open) · 원래·새 부호를 `<record>.gripchanges.csv` 에 남기고, 단계별 변경 수를 출력한다 | `sample` 10 rollout(A6000) = baseline 기록과 행동까지 bit 동일 10/10. `majority` 10 rollout(gx10-2 s111): 7/10, 4,806 step 중 13 step 변경(접근 4 · 파지 4 · 운반 1 · 배치 4). 단계 기록을 넣은 뒤 다시 돌려도 결과·변경이 그대로 같음. `resample` 10 rollout(gx10-2 s111): 5/10, 5,973 step 중 24 step 변경(접근 11 · 파지 2 · 배치 5 · 놓친 뒤 6). 같은 기계 baseline 첫 10회는 7/10 |
| `tools/plan_C/attribute_c3.py` | 귀속 (ii) 계층형과 민감도 (i)·(iii). P(open) 은 `gmm_offline.py` 결과를 rollout·step 키로 붙인다 | baseline A6000·Spark 결과는 §1 표 |
| `tools/plan_C/gmm_offline.py` | 기록 상태에 체크포인트를 다시 넣어 step 별 실행 행동·혼합 평균·P(open)·모드 퍼짐을 저장(Isaac 불필요) | baseline 4개 기록에 적용 |
| `tools/plan_C/run_plan_C.sh` | `c3spark <seed> [평가 seed]` / `placebo <seed> [평가 seed]` 순서: seed→Spark 대응(101→gx10-3, 111→gx10-2) → 도구 복사 → 체크포인트 md5 대조(다르면 복사) → ssh 로 평가 → 기록·변경 로그 회수 → A6000 에서 분류·`gmm_offline`·트랙 C 귀속·`carry_metrics.py`. `1mode <seed> [평가 seed]` 는 A6000 학습·평가(체크포인트가 이미 있으면 학습을 건너뜀). 평가 seed 는 생략하면 42. 출력은 `<NAME>_seed<E>.csv` 와 그 옆 `<NAME>_seed<E>.carry.csv`. 기록 rollout 수가 100 이 아니면 종료 코드 5 | 문법 확인만 (`bash -n`) |
| `tools/plan_C/train_override.py` | `--seed` 와 `--set key.path=JSON값` 으로 config 를 바꾸는 학습 래퍼 | `num_modes=1` 100 epoch 스모크: config 반영, 기본 `play.py` 경로 평가 정상(4 rollout, RC 0). epoch 100 손실 −7.9 (5 모드 −23.3) |

## 4. 게이트

- **C3 · 위약 조작 확인 (평가 뒤, 판정 전):**
  - 바뀐 step 수 > 0 이고 단계별 분포를 보고한다.
  - C3 기록에서 '운반·배치 단계에서 P(open) < 0.5 인 열기'가 0 이어야 한다(규칙상). 0 이 아니면 래퍼 오류다. 소수 열기 몫이 0 이 되는 것은 이 조작 확인으로만 쓴다(prereg §5.4).
  - 위약 기록에서는 소수 열기 비율이 Spark `sample` baseline 과 같은 수준이어야 한다(위약이 '다시 뽑기'만 하는지 확인).
- **C-1mode:** C3 가 prereg §5.4 의 '지지'일 때만 학습한다. 학습 뒤 `config.json` 에서 `num_modes == 1`·`seed` 를 확인한다(스크립트가 자동 확인, 다르면 평가하지 않음).

## 5. 판정 (prereg §5.4) · 트랙 C 의 최소 예측

판정 규칙은 prereg §5.4 하나만 따른다. 여기에는 다시 적지 않는다.
- 비교는 C3 대 위약, 학습 seed 별 같은 Spark, 평가 seed 42, 200쌍이다.
- 판정 묶음은 들기 이후 실패 중 비충돌 스스로 놓음(소수 + 다수 열기, 공용 `attr_ii`)이다. 명령은 §7.
- '지지' · '미지지' 조건은 prereg §5.4, null 띠 값은 §5.1, HOLD 세부 유형은 §4 에 있다. 미지지는 H_C 기각이 아니고 "구분 불가" 다.
- 음성 대조는 들기 전 실패다(아래, prereg §5.2).
- 성공률 '효과'는 주장하지 않는다. 성공률 Δ ≥ 0 은 '지지' 의 한 조건일 뿐이다.

**판정 묶음을 이렇게 잡은 이유 (트랙 C 의 제안, prereg 변경 기록 6 에 반영):**
1. **소수 열기 몫은 C3 에서 정의상 0 에 가깝다.** 다수결 규칙은 P(open) < 0.5 에서 열지 않기 때문이다. 그래서 그 몫이 0 이 되는 것은 조작 확인(§4)일 뿐이고, 이것만으로는 H_C 를 시험하지 못한다.
   - **실제로 답이 갈리는 곳은 그 rollout 들이 어디로 가느냐다.** 성공으로 바뀌면 H_C 쪽이다. 같은 자리에서 조금 뒤 다수 열기·HOLD 로 옮겨 가면 그리퍼는 원인이 아니고 팔의 위치가 원인일 수 있다(H-A 쪽).
   - 그래서 판정 묶음은 비충돌 스스로 놓음 전체(소수 + 다수)이고, HOLD 를 함께 본다.
2. **건수가 적다.** 위약에서 기대되는 판정 묶음은 12–19건이고(트랙 C 도구 기준 Spark `sample` 8 + 4, A6000 6 + 13), 그중 소수 열기는 6–8건이다. 아래 예측대로 순감소가 약 3건이면 '지지' 문턱(prereg §5.4)에 못 미친다. **예상 판정은 '미지지'** 이고, 그러면 C-1mode 는 돌리지 않는다(prereg §3·§5.4). 이 결과도 그대로 보고한다.

**최소 예측 (트랙 C):**

| 묶음 (200쌍) | 위약 예측 | C3 예측 |
| --- | --- | --- |
| 비충돌 소수 열기 (조작 확인) | 6–8 | 0–1 (구성상) |
| 비충돌 스스로 놓음 전체 (판정 묶음) | 12–19 | 9–17 |
| HOLD (정의 prereg §4) | 6–9 | 7–12 |
| 미끄러짐 (보고만) | 8–11 | 8–11 (변화 없음) |
| 들기 전 실패 (보고만) | — | 위약과 차를 보고 (prereg §5.4: 판정을 바꾸지 않음) |
| 성공 | — | 위약 대비 −4 ~ +6%p |

- **들기 전 실패:** C3 대 위약의 들기 전 실패 변화는 보고만 하고 판정을 바꾸지 않는다(prereg §5.4). C-1mode 의 음성 대조는 prereg §5.2 를 따른다. 다수결은 접근·파지 단계 결정도 바꾼다. 스모크 13 step 중 8 step 이 들기 전이었다. 그래서 C3 에서 이 대조가 특히 중요하다.
- rot60_state · approach_z · carry_yaw_net · 미끄러짐의 차는 보고만 한다(prereg §5.1 null 띠와 나란히 적는다). 처음 기획의 ±15% 폭은 평가 잡음보다 좁아 버렸다(#18). Spark 에서 rot60 중앙값만 0.101–0.160 으로 흔들린다.

**C-1mode (강등):** prereg §5.2 를 따른다(400쌍 = 평가 seed 42 + 43, ② 는 첫 조건 없이 둘째 조건만). 트랙 C 의 예측은 Δ ∈ [−8, +8]%p 이고 표적 묶음은 −30% 안팎이다(예측일 뿐, 판정에 쓰지 않음). 음성 대조는 prereg §5.2 규칙만 쓰고, rot60_state · approach_z 차는 보고만 한다. 결과는 H_C 해석에 쓰지 않고 제출 판단에만 쓴다(prereg §6 H_C1·§7).

## 6. 해석 (prereg §6 중 트랙 C 해당 행)

해석은 prereg §6 의 H_C 표(H_C1 · H_C2)만 따른다. 이 표는 그 행을 트랙 C 의 다음 단계와 잇기만 한다.

| prereg §5.4 판정 | 해석 (prereg §6) | 다음 |
| --- | --- | --- |
| 지지 | H_C1: 그리퍼 소수 성분 샘플이 들기 이후 실패의 독립 원인 일부 (들기 전 실패 변화는 함께 보고) | C-1mode 를 A6000 큐에 붙인다(prereg §3·§7). C-1mode 결과는 제출 판단에만 쓴다 |
| 미지지 | H_C2: H_C 는 "판단 보류" 로만 적는다(기각 아님) | C-1mode 실행 안 함. 그대로 보고 |

- 미지지일 때 소수 열기 rollout 이 어디로 옮겨 갔는지(성공 · 다수 열기 · HOLD)는 범주 이동표로 기술만 한다. "그리퍼는 증상, 원인은 팔 위치" 같은 원인 주장은 하지 않는다.
- C-1mode 의 ①–④ 등급(prereg §5.2)은 H_C 해석에 넣지 않는다. 제출 후보 규칙은 prereg §7 이다.
- (등록 밖, 탐색) C3 가 지지인데 최종 후보(A-T 등) 위에서 C3 대 위약 차가 없으면 그리퍼 실패가 회전·충돌의 하류일 수 있다. 10/31 선택 실험(최종 후보 위 C3 + 위약)으로 보고만 한다.

## 7. 실행 명령 · 계산량 · 일정

```bash
A2=~/robotics-simulation-team12/assignment2_toy2box; R=$A2/results/plan_C
# A6000 의 tmux 창 네 개에서 하나씩 (기록 수집·분류는 A6000, 평가는 Spark). 같은 Spark 에서 C3·위약 동시 실행 가능.
# 같은 학습 seed 는 c3spark 를 먼저 띄우고 'ckpt md5' 줄이 찍힌 뒤 placebo 를 띄운다 (원격 체크포인트 복사가 겹치지 않게).
# 셋째 인자는 평가 seed (생략하면 42). 출력: $R/<NAME>_seed42.csv 와 그 옆 <NAME>_seed42.carry.csv
bash $A2/tools/plan_C/run_plan_C.sh c3spark 101   # gx10-3
bash $A2/tools/plan_C/run_plan_C.sh placebo 101   # gx10-3
bash $A2/tools/plan_C/run_plan_C.sh c3spark 111   # gx10-2
bash $A2/tools/plan_C/run_plan_C.sh placebo 111   # gx10-2

# 판정 (prereg §5.4): C3 대 위약 200쌍. 판정 묶음 = 비충돌 스스로 놓음
PL="$R/C3placebo_s101_e2000_gx10-3_seed42.csv $R/C3placebo_s111_e2000_gx10-2_seed42.csv"
C3="$R/C3maj_s101_e2000_gx10-3_seed42.csv $R/C3maj_s111_e2000_gx10-2_seed42.csv"
python -I $A2/tools/paired_compare.py --a $PL --b $C3 --label_a 위약 --label_b C3 \
  --bundle_col attr_ii --bundle self_release --require_carry
# 같은 짝으로 HOLD (세부 유형은 prereg §4)
python -I $A2/tools/paired_compare.py --a $PL --b $C3 --label_a 위약 --label_b C3 \
  --bundle_col subtype --bundle no_release,in_box_not_open,stall --require_carry
# 들기 전 실패 (prereg §5.4: 보고만)
python -I $A2/tools/paired_compare.py --a $PL --b $C3 --label_a 위약 --label_b C3 \
  --bundle approach,grasp --require_carry
# 보고만: 소수·다수 나눔 $R/<NAME>_seed42_attr.csv (트랙 C 도구), 조작 확인 $R/<NAME>_seed42_gripchanges.csv

# C-1mode (C3 '지지' 일 때만, A6000). 평가 seed 42 로 학습·평가한 뒤 seed 43 평가 (두 번째 호출은 학습을 건너뜀)
bash $A2/tools/plan_C/run_plan_C.sh 1mode 101
bash $A2/tools/plan_C/run_plan_C.sh 1mode 111
bash $A2/tools/plan_C/run_plan_C.sh 1mode 101 43
bash $A2/tools/plan_C/run_plan_C.sh 1mode 111 43
# 판정 (prereg §5.2): baseline e2000 대비 400쌍. baseline seed 43 은 A6000 큐의 run_baseline.sh 출력(기본 OUTDIR)
B=$A2/results/baseline
python -I $A2/tools/paired_compare.py \
  --a $B/base_kgh500_s101_e2000_seed42.csv $B/base_kgh500_s111_e2000_seed42.csv $B/base_kgh500_s101_e2000_seed43.csv $B/base_kgh500_s111_e2000_seed43.csv \
  --b $R/C1mode_kgh500_s101_e2000_seed42.csv $R/C1mode_kgh500_s111_e2000_seed42.csv $R/C1mode_kgh500_s101_e2000_seed43.csv $R/C1mode_kgh500_s111_e2000_seed43.csv \
  --label_a baseline --label_b C-1mode --bundle transport,place --require_carry
```

- **계산량:** Spark 2대 × 약 1–1.5 h 다(대마다 C3·위약 평가 2개 동시, 앞선 Spark 2개 동시 측정 기준). A6000 은 쓰지 않는다(분류·`gmm_offline`·`carry_metrics.py` CPU 몇 분만).
- **C-1mode:** 400쌍이라 A6000 학습 2개 + 평가 seed 42 2개 약 2 h, 평가 seed 43 2개 약 1.1 h 로 합 약 3.1 h 다(prereg §3·§7).
- **일정(prereg §7):** C3·위약은 Spark 2대에서 A6000 직렬 큐와 병행한다. 고정 슬롯은 없다. C-1mode 는 C3 가 '지지'일 때만 A6000 큐에 붙고, A6000 이 모자라면 가장 먼저 뺀다.

## 8. 위험 · 대안 · 규칙 확인

- **제출 규칙:** C3·위약·교차 seed 투표·BID·히스테리시스는 추론 시점 규칙이라 조교 `play.py` 평가에 실리지 않는다. 분석 도구(자유)로만 쓴다. C-1mode 는 체크포인트 config 에 들어가 조교 평가에서도 같은 정책이 된다.
- **C-1mode 위험:**
  - 단일 가우시안이 다봉 행동을 평균한다. epoch 100 손실 격차(−7.9 vs −23.3)가 그 신호다.
  - 정상 놓기가 늦어질 수 있다(HOLD).
  - 팀이 고정한 GMM 5 를 바꾸는 일반 하이퍼파라미터 교체라 독창성 점수가 낮다.
- **기계:** C3·위약은 Spark 에서 짝지어 비교하므로 A6000 baseline 과 섞지 않는다. 체크포인트는 md5 로 A6000 것과 같음을 확인한다(스크립트).
- **고려했다가 뺀 것:**
  - EMA·lr decay: 체크포인트 다수결 70% ≈ 평균 69% 이라 근거가 약하다.
  - C1(교차 seed 다수결): seed 규칙에 걸리고 학습량이 3배이며 제출에 실리지 않는다.
  - C2(스냅샷 다수결): 체크포인트 규칙에 걸린다.
  - RNN horizon: 리셋 위상과 무관하다.
- **다루지 못하는 실패:** 트럭을 놓친 뒤 빈손으로 상자에 가는 회복 실패(실패 18 · 성공 2)는 데이터 쪽 몫이다(DART 식 교란·회복 시범, 트랙 A 와 조율).

## 9. 남은 질문

1. (정해짐) C3 판정 묶음은 '비충돌 스스로 놓음 전체(소수 + 다수)'다. 소수 열기 몫은 조작 확인으로만 쓴다(prereg §5.4).
2. (정해짐) 판정은 공용 `carry_metrics.py` 의 `attr_ii` 로 한다(prereg §4·§5.4). 트랙 C 도구와는 s101 rollout_66 · s111 rollout_70 두 건만 다르다(공용 충돌, C 도구 기타 — §1).
3. 조교에게 "평가 명령에 래퍼 스크립트를 써도 되는가"를 물을지. 된다면 C3 가 제출 후보가 된다(지금은 안 되는 것으로 가정).
