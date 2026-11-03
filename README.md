# robotics-simulation-team12

로보틱스 시뮬레이션 12조 학기 공용 레포.

| 폴더 | 내용 |
|---|---|
| [`assignment1_ant/`](assignment1_ant/README.md) | 실습 과제 1 — 처음 보는 환경에서도 잘 걷는 Ant (Isaac-Ant-v0, RSL-RL PPO). 10.06 제출 완료 · 10.08 발표. 최종 모델 `assignment1_ant/checkpoints/final/model_2999.pt`, 평가 명령 `assignment1_ant/eval_command.txt` |
| [`assignment2_toy2box/`](assignment2_toy2box/README.md) | 실습 과제 2 — Agibot Toy2Box 모방학습 성공률 향상 (Isaac Lab Mimic + robomimic BC-RNN). 11.03 LMS 마감 · 11.05 발표. 최종 모델 2개(학습 seed 101 · 111) `assignment2_toy2box/checkpoints/final/seed101/model_epoch_2000.pth` · `assignment2_toy2box/checkpoints/final/seed111/model_epoch_2000.pth`, 평가 명령 `assignment2_toy2box/eval_command.txt` |

## 공통 환경

- Ubuntu 22.04 · Isaac Sim 5.1.0 · Isaac Lab 2.3.0 (`cailab-hy/IsaacLab_RS` main) · Python 3.11
- conda env: `lerobot-arena`

과제 2 가중치는 `assignment2_toy2box/checkpoints/final/` 의 두 개(학습 seed 101 · 111)뿐이고, `assignment1_ant/` 의 `.pt` 는 과제 1(10.06 제출) 것입니다.

과제 1 폴더는 IsaacLab 외부 확장 프로젝트로 `pip install -e` 로 설치합니다. 과제 2는 설치할 패키지 없이 수업 태스크와 조교 `play.py` 를 그대로 씁니다(평가 명령 `assignment2_toy2box/eval_command.txt`).
