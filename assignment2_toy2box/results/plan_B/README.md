# 트랙 B 점검 결과 (2026-10-29, 교차 점검 반영)

기획안: `docs/plan_B_1029.md`. 판정 규칙은 `docs/prereg_1029.md`.

| 파일 | 내용 |
| --- | --- |
| `data_gate_carry_metrics.csv` | 원본·B1·B2·구판 B3 데이터 500개씩 `carry_metrics.py` 행 (라벨 기준 운반 지표) |
| `replay_gate_{orig,carrynorot,reach4,cap,dirorig_magb2,reach4carry}.csv`, `replay_gate_all.csv` | gx10-3 open-loop 재생 10개 × 6 변형: 재생 성공, v3 범주, dev50·dev_lift·devmax, 접근 step·들기→상자 위·운반 속도 (`plan_B_speed_metrics.py --replay_src`) |
| `speed_baseline_rollouts.{csv,txt}` | baseline 200 rollout 속도·접근·시간 초과 기준값 (`plan_B_speed_metrics.py`) |
| `speed_data_human_mimic.csv` | 사람 시범 20 · Mimic 500 같은 지표 |

재라벨 데이터 (`~/IsaacLab_RS/datasets/team12_kgh/plan_B/`, git 밖, 원본 `kgh_mimic_500.hdf5` md5 5e9bebaf4ee4111f4cc4cae9b58c994d):

| 파일 | 팔 | md5 | 바뀐 step |
| --- | --- | --- | --- |
| `kgh_mimic_500_reach4.hdf5` | B2 | 32f3c488ce9435e073adc62be94a9853 | 전체 |
| `kgh_mimic_500_dirorig_magb2.hdf5` | B3' | a025e391eacf65ca1cf3b27140f160ce | 전체 |
| `kgh_mimic_500_reach4carry.hdf5` | B2-carry (예비) | 3d5fb2875ac4fa35ae84e0620283e044 | 72,186 (34.7%) |
| `kgh_mimic_500_carrynorot.hdf5` | B1 (실행 안 함) | 3d15fe64b68ea52f8db7b27a0bd1213e | 62,564 (30.1%) |
| `kgh_mimic_500_cap.hdf5` | 구판 B3 (폐기) | b65d318c7e12b4f51b1082f012dfdef2 | 103,093 (49.6%) |

재생 기록: `~/IsaacLab_RS/logs/rs_a2_runs/plan_B/replay_sub10_*.hdf5`. 10/29 14:00 이전 기록(orig·carrynorot·reach4·cap)은 `replay_check.py` 수정 전이라 행동 열이 마지막 값으로 덮여 있다 — 분석은 데이터셋 행동으로 바꿔 계산한다(`plan_B_speed_metrics.py --replay_src` 가 자동으로 함). 학습 스모크 run `logs/robomimic/.../_plan_B_smoke_carrynorot/` (지워도 됨).

B1 분포 밖 분석(운반 상태 최근접 거리, 23-D 표준화, 시범 100개 질의 · 400개 풀): 실제 p50 1.32 · 손목 고정 상태 p50 1.84 (×1.40).
