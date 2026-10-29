# 트랙 A 스모크 결과 (2026-10-29, A6000 `--num_envs 50`)

기획안: `docs/plan_A_1029.md`. 데이터는 `~/IsaacLab_RS/datasets/team12_kgh/plan_A/`, 생성 로그·원본 선택 로그는 `~/IsaacLab_RS/logs/rs_a2_runs/plan_A/`.

| 파일 | 내용 |
| --- | --- |
| `smoke_gate_t1_vs_s1.txt` | yaw-hold seed 1 (31개) vs baseline seed 1 샤드 — 데이터 지표·생성률·시작 상태 |
| `smoke_gate_t2_vs_s2.txt` | yaw-hold seed 2 (30개) vs baseline seed 2 샤드 |
| `smoke_gate_x1_vs_s1.txt` | TRANSLATE seed 1 (31개) |
| `render/{baseline,yawhold,translate}/` | 생성 시범 장면 띠 (gx10-2, 상태 재생) |

- 수동성: `--yawhold 0` seed 1 의 demo 20개가 baseline seed 1 샤드와 시작 상태·길이·actions 모두 bit 동일.
- 생성 성공: baseline s1 36.4%(55회 시점) · yaw-hold s1 53.4% · s2 53.6% · TRANSLATE s1 88.6%.
- 손목 yaw 파지→상자 위 중앙값: 사람 2.2° · baseline 43.1° · yaw-hold 3.4–3.5° · TRANSLATE 13.0°.
- 들기 뒤 60 step 회전 명령: 사람 0.028 · baseline 0.177/0.260 · yaw-hold 0.195/0.131 · TRANSLATE 0.063.

## 교차 점검 반영 (1029 오후)

| 파일 | 내용 |
| --- | --- |
| `gate_table_1029.md` | 사람 · baseline s1 · baseline 500 · A-T s1 · A-R s1 · A-Y s1(+s2) 게이트 지표 나란히 (`plan_A_smoke_check.py table`) |
| `stratify_smoke.txt` | `plan_A_stratify.py` 스모크 (A-T 31 + A-Y 31 → 20개, 거리·yaw 칸 맞춤 표) |

- A-R (`--rotworld 1`, seed 1, 성공 30): 생성 30/52 (57.7%), 손목 yaw 파지→상자 위 13.4° (A-T 13.0°), xy30 0.299, 경로 1.29 m, box_push 17%, rot60 0.263.
