| run | n | 성공 | 접근 | 파지 | 운반 | 배치 |
| --- | --- | --- | --- | --- | --- | --- |
| seed 101 | 100 | 63 | 4 | 7 | 12 | 14 |
| seed 111 | 100 | 69 | 5 | 3 | 18 | 5 |

성공률 평균 ± 표준편차(모표준편차, n=2): 66.0% ± 3.0% · 표본표준편차 4.2%

| run | 범주 | 세부 유형 | 건수 |
| --- | --- | --- | --- |
| seed 101 | 접근 | far | 4 |
| seed 101 | 파지 | miss | 4 |
| seed 101 | 파지 | slip | 3 |
| seed 101 | 운반 | released_near_box | 3 |
| seed 101 | 운반 | dropped | 3 |
| seed 101 | 운반 | released_early | 3 |
| seed 101 | 운반 | box_collision_released | 2 |
| seed 101 | 운반 | dropped_near_box | 1 |
| seed 101 | 배치 | released_outside | 8 |
| seed 101 | 배치 | dropped_outside | 2 |
| seed 101 | 배치 | in_box_offcenter | 2 |
| seed 101 | 배치 | in_box_not_open | 1 |
| seed 101 | 배치 | no_release | 1 |
| seed 111 | 접근 | hover | 2 |
| seed 111 | 접근 | wrong_object | 1 |
| seed 111 | 접근 | knocked | 1 |
| seed 111 | 접근 | close_misaligned | 1 |
| seed 111 | 파지 | miss | 1 |
| seed 111 | 파지 | no_close | 1 |
| seed 111 | 파지 | slip | 1 |
| seed 111 | 운반 | released_near_box | 7 |
| seed 111 | 운반 | dropped | 4 |
| seed 111 | 운반 | stall | 3 |
| seed 111 | 운반 | dropped_near_box | 2 |
| seed 111 | 운반 | box_knocked_over | 1 |
| seed 111 | 운반 | released_early | 1 |
| seed 111 | 배치 | released_outside | 4 |
| seed 111 | 배치 | in_box_not_open | 1 |

### 같은 시작 상태 짝짓기 (seed 101 × seed 111, n=100)

- 둘 다 실패 18 · 한쪽만 실패 32 · 둘 다 성공 50
- 실패가 서로 독립이라면 둘 다 실패 기대값 11.5 (실패율 37% × 31%)
- 둘 다 실패한 rollout: rollout_2(place/transport), rollout_5(approach/approach), rollout_11(approach/grasp), rollout_26(grasp/grasp), rollout_27(transport/transport), rollout_38(grasp/transport), rollout_46(place/transport), rollout_55(transport/place), rollout_56(grasp/transport), rollout_62(transport/transport), rollout_64(approach/transport), rollout_69(place/place), rollout_70(place/transport), rollout_76(transport/transport), rollout_79(grasp/approach), rollout_92(approach/place), rollout_94(transport/transport), rollout_97(place/transport)

### 시작 상태 구간별 성공률 (모든 run 합침)

- 트럭–상자 xy 거리 (m): 0.15–0.25: 0% (n=4) · 0.25–0.3: 50% (n=24) · 0.3–0.35: 69% (n=26) · 0.35–0.4: 74% (n=54) · 0.4–0.45: 75% (n=36) · 0.45–0.5: 62% (n=34) · 0.5–0.65: 64% (n=22)
- 트럭 x (base, m): 0.4–0.5: 62% (n=26) · 0.5–0.6: 64% (n=58) · 0.6–0.7: 78% (n=58) · 0.7–0.8: 59% (n=58)
- 트럭 y (base, m): -0.32–-0.28: 71% (n=28) · -0.28–-0.24: 73% (n=48) · -0.24–-0.2: 76% (n=58) · -0.2–-0.16: 50% (n=46) · -0.16–-0.12: 50% (n=20)
- 트럭 yaw mod 180 (deg): -90–-60: 47% (n=36) · -60–-30: 65% (n=26) · -30–0: 54% (n=28) · 0–30: 79% (n=38) · 30–60: 59% (n=32) · 60–90.1: 85% (n=40)
- 가장 가까운 원본 시범까지 거리 (트럭·상자 xy + yaw/180대칭, 3분위): 0.025–0.081: 66% (n=68) · 0.081–0.099: 78% (n=68) · 0.099–0.161: 56% (n=68)
