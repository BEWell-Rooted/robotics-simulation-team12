# robotics-simulation-team12

로보틱스 시뮬레이션 12조 학기 공용 레포.

| 폴더 | 내용 |
|---|---|
| [`assignment1_ant/`](assignment1_ant/README.md) | 실습 과제 1 — 처음 보는 환경에서도 잘 걷는 Ant (Isaac-Ant-v0, RSL-RL PPO). 마감 10.06, 진행 중 (중간 보고 10.01) |

## 공통 환경

- Ubuntu 22.04 · Isaac Sim 5.1.0 · Isaac Lab 2.3.0 (`cailab-hy/IsaacLab_RS` main) · Python 3.11
- conda env: `lerobot-arena`

각 과제 폴더는 IsaacLab 외부 확장 프로젝트로, `pip install -e` 로 독립 설치됩니다.
