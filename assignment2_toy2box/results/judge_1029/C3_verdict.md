# C3 ↔ 위약 판정 (사전등록 `docs/prereg_1029.md` §5.4) — 2026-10-29 19:22

- 평가: 학습 seed 별 같은 Spark (s101 gx10-3 · s111 gx10-2), baseline e2000 체크포인트(md5 A6000 과 같음), 평가 seed 42, 100회, horizon 800. 평가 18:08–19:21.
- 짝: 시작 상태로 짝지은 198쌍 (s101 쪽 dropped 위약 2 · C3 1).
- 명령: `paired_compare.py --a <위약 2개> --b <C3 2개> --bundle_col attr_ii --bundle self_release --require_carry` (출력 `C3_vs_placebo_self_release.{txt,json}`), HOLD `--bundle_col subtype --bundle no_release,in_box_not_open,stall`, 들기 전 실패 `--bundle approach,grasp`.

| 항목 | 위약 | C3 (혼합 다수결) | 판정 조건 | 충족 |
| --- | --- | --- | --- | --- |
| 판정 묶음: 들기 이후 비충돌 스스로 놓음 (attr_ii) | 14 | 14 (위약만 7 : C3 만 7, 단측 p = 0.60) | 감소 × 100/n > 4.04 · 단측 p < 0.10 | 아니오 (감소 0) |
| HOLD (no_release · in_box_not_open · stall) | 7 | 6 | 증가 ≤ 2.03/100 | 예 |
| 성공 | 141 / 198 (71.2%) | 139 / 198 (70.2%), Δ −1.0%p, 양측 p = 0.89, 95% CI [−8.0, +6.0] | Δ ≥ 0 | 아니오 |
| 들기 전 실패 (보고만) | 15 | 14 | — | — |

**판정: 미지지 → H_C 판단 보류 (§6 H_C2). C-1mode 는 돌리지 않는다 (§3).**

- 조작 확인: 소수 성분 열기(트랙 C `attribute_c3.py`, 들기 이후 실패 중) 위약 6 → C3 0. 그리퍼 부호가 바뀐 step: C3 s101 200 · s111 242 / 위약 232 · 251. 처치는 들어갔다.
- 기술(원인 주장 아님): 사라진 소수 열기 대신 다수 성분 열기가 7 → 15 로 늘어 스스로 놓음 전체(트랙 C 집계 13 → 15)는 줄지 않았다. 상태 매개 지표 차는 모두 null 띠 안(`push_carry` +2, `box_push` −5, `rot60_state` −8.2°).
- 독립 재계산(별도 에이전트, `paired_compare.py` 를 쓰지 않고 csv 모듈로 직접): 짝 198 · 성공 141 / 139 · 판정 묶음 14 / 14 (7 : 7, 단측 p = 0.6047) · HOLD 7 → 6 · 소수 열기 6 → 0 · 판정 미지지 — 모든 숫자 일치. 원격 로그의 grip_rule(majority / resample)과 그리퍼 변경 수가 결과 파일과 맞아 파일 이름표도 확인됐다.
