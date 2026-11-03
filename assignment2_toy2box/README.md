# 실습과제 2 — Agibot Toy2Box 모방학습 (LMS 마감 2026-11-03 23:59, 발표 11-05)

연구 질문(고정): 사람 시범으로 학습한 BC 정책은 Toy2Box에서 왜 성공률이 저조한가? 원인 하나를 진단하고, 그 원인을 겨냥해 성공률을 높인다.

- **최종 제출 (가중치 2개, 수업 규칙: 학습 seed 101 · 111 최종 모델)**: A-T 팔, epoch 2000. 평가 명령 [`eval_command.txt`](eval_command.txt)
  - 학습 seed 101: `checkpoints/final/seed101/model_epoch_2000.pth` (md5 `aca48edb71abbe9e8e2181715fc197d0`) — 평가 seed 42 에서 84 / 100
  - 학습 seed 111: `checkpoints/final/seed111/model_epoch_2000.pth` (md5 `fc88573fa422ac39310e2ffca5d90620`) — 평가 seed 42 에서 92 / 100
- 바꾼 것은 **학습 데이터 하나**다. 환경 · 성공 판정 · 조교 BC-RNN config · `play.py` 는 그대로 쓴다.
- 결과 요약 문서: [`docs/log_1030.md`](docs/log_1030.md) · 판정 원문: [`results/judge_1029/verdict_1029.md`](results/judge_1029/verdict_1029.md) · 사전등록: [`docs/prereg_1029.md`](docs/prereg_1029.md) · 진단: [`docs/diagnosis_1028.md`](docs/diagnosis_1028.md)

## 결과 요약

평가 seed 42, 학습 seed 101 · 111 각 100회, horizon 800, 체크포인트 epoch 2000. ± 는 2 seed 모표준편차.

| 팔 | 바꾼 것 | 성공률 | 판정 (사전등록 §5) |
| --- | --- | --- | --- |
| baseline | 팀원 사람 시범 20 → Mimic 500 → 조교 config | 66.0 ± 3.0% | — |
| **A-T** | Mimic 상자 구간을 위치만 따라가게 (원본 world 회전 유지) | **88.0 ± 4.0%** | **① 효과** (평가 seed 42 + 43 짝 395쌍 Δ +17.0%p, Holm p 2.6e-8) |
| A-R | 상자 구간 손목만 원본 방향, 경로는 SE(3) 그대로 | 64.0 ± 1.0% | 귀속: A-T 보다 낮음 → 재타깃팅의 위치 · 우회 성분이 몫 |
| B2 | 행동 라벨 = 4 step 뒤 실제 자세 − 현재 | 66.0 ± 0.0% | ② 기각 |
| B3' | 라벨 방향 원본 × B2 크기 | 38.5 ± 1.5% | 보고만 |

- 진단: baseline 실패 68건 중 49건(72%)이 트럭을 들어 올린 뒤에 났다. Mimic 은 원본 시범을 트럭 구간에서만 고르고 상자 구간에도 같은 원본을 쓰므로, 상자 구간을 SE(3)로 옮기면 새 상자와 원본 상자의 방향 차이만큼 운반 중 손목이 돌고 상자 둘레로 돌아가는 경로가 생긴다.
- 그리퍼 다수결(C3) ↔ 위약: 미지지(판단 보류).
- 한계: A-R 은 접근 높이도 낮아 우회와 낮은 접근을 나눌 수 없다. 효과 크기가 평가 seed 마다 다르다(seed 42 +22.6 · seed 43 +11.2%p). 제출 확인과 seed 43 은 재현이 아니다.

## 최종 체크포인트 실행 (조교 평가 형식)

[`eval_command.txt`](eval_command.txt) 참고. 수업 환경(conda `lerobot-arena`, `~/IsaacLab_RS`)에서 조교 `play.py` 를 고치지 않고 그대로 쓴다. 두 가중치는 `--checkpoint` 경로만 다르다.

```bash
cd ~/IsaacLab_RS
REPO=~/robotics-simulation-team12   # 이 레포를 clone 한 폴더
# 학습 seed 101
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
    --checkpoint $REPO/assignment2_toy2box/checkpoints/final/seed101/model_epoch_2000.pth \
    --num_rollouts 100 --horizon 800 --seed 42 --headless
# 학습 seed 111
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
    --checkpoint $REPO/assignment2_toy2box/checkpoints/final/seed111/model_epoch_2000.pth \
    --num_rollouts 100 --horizon 800 --seed 42 --headless
```

우리 기계(NVIDIA RTX 6000 Ada)에서 이 명령들의 결과는 seed 101 84 / 100, seed 111 92 / 100 이다(평균 88.0 ± 4.0%).

## 재현 절차 (A-T 데이터 → 학습)

`~/IsaacLab_RS` 에서, `A2=$REPO/assignment2_toy2box` (`REPO` = 이 레포를 clone 한 폴더), `L=~/IsaacLab_RS/logs/rs_a2_runs/plan_A`. 사람 시범은 팀원 키보드 시범 20개(`datasets/team12_kgh/toy2box_kgh_annotated.hdf5`, md5 `8ddcba040c4d44e0ae933df2855b28ff`), 데이터 예산은 사람 20 / Mimic 500 (규칙 상한 100 / 1,000). 데이터 파일(`*.hdf5`)은 용량 때문에 레포에 넣지 않았다(Mimic 500 파일 177 MB). md5 로 같은 파일인지 확인할 수 있다.

1. **Mimic 생성 (상자 구간 위치만):** datagen seed 마다 성공 130개 샤드를 만든다 (seed 1–6, 칸 부족으로 6번째 추가 — 사전등록 §7 처치 시작 뒤 1).
   ```bash
   # s = 1 … 6. 여러 개를 동시에 돌리면 프로세스마다 TMPDIR 를 따로 준다 (RMPFlow 파일이 부딪힘)
   TMPDIR=/tmp/rs_a2_tmp_AT_s<s> ./isaaclab.sh -p $A2/tools/generate_yawhold.py --translate 1 --datagen_seed <s> \
       --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 50 \
       --generation_num_trials 130 --input_file ./datasets/team12_kgh/toy2box_kgh_annotated.hdf5 \
       --output_file ./datasets/team12_kgh/plan_A/translate_shards/translate_s<s>_130.hdf5 \
       > $L/AT_s<s>.log 2>&1      # 데이터 게이트가 이 로그의 생성률을 읽는다
   ```
2. **층화 500개:** baseline 500 과 같은 장면 분포(시작 거리 4 × 상대 yaw 4 칸)로 뽑는다.
   ```bash
   python -I $A2/tools/plan_A_stratify.py --ref ./datasets/team12_kgh/kgh_mimic_500.hdf5 --n 500 --seed 0 \
       --out ./datasets/team12_kgh/plan_A/translate_strat500.hdf5 ./datasets/team12_kgh/plan_A/translate_shards/translate_s{1..6}_130.hdf5
   ```
   결과 `translate_strat500.hdf5` md5 `c2456916d62f3c4c2e7d09fec2fd1be9`. 학습 전 데이터 게이트 (결과 [`results/plan_A/gate_AT500.txt`](results/plan_A/gate_AT500.txt)):
   ```bash
   python -I $A2/tools/plan_A_smoke_check.py check AT ./datasets/team12_kgh/plan_A/translate_strat500.hdf5 \
       --n_logs 6 --log $L/AT_s{1..6}.log
   ```
3. **학습 (조교 config, 2000 epoch):** `train_seed.py` 는 `train.py` 를 고치지 않고 학습 seed 만 CLI 로 주는 래퍼다 (seed 101 은 조교 json 기본값).
   ```bash
   # S = 101, 111 (제출한 두 가중치)
   ./isaaclab.sh -p $A2/tools/train_seed.py --seed $S --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
       --algo bc --dataset ./datasets/team12_kgh/plan_A/translate_strat500.hdf5 --name base_translate500_s$S
   ```
4. **평가 · 분류:** `tools/run_baseline.sh` (학습 → `play_record.py` 기록 평가 → 실패 분류). 판정은 `tools/judge_1029.py`, 보고 보충은 `tools/report_1029.py`.

전체 실행 순서는 `tools/run_queue_1029.sh` · `tools/run_queue_1029_followup.sh` 에 있다.

## 폴더

- `docs/`: 진단 · 서베이 · 사전등록 · 작업 로그(`log_MMDD.md`), `run_logs/` 실행 로그 요약, `env/` 패키지 스냅샷. 인계 브리프 `handoff_MMDD.md`.
- `tools/`: 생성(`generate_yawhold.py`), 층화(`plan_A_stratify.py`), 학습 래퍼(`train_seed.py`), 기록 평가(`play_record.py`), 실패 분류(`classify_failures.py`), 운반 지표(`carry_metrics.py`), 짝 비교(`paired_compare.py`), 판정(`judge_1029.py`), 렌더(`render_record.py` 등).
- `results/`: 평가 CSV · 분류 · 판정(`judge_1029/`) · 잡음 기준선(`noise_baseline/`) · 트랙별 결과(`plan_A/B/C`).
- `checkpoints/final/`: 제출 가중치 2개(`seed101/` · `seed111/`, 폴더마다 학습 설정 사본 `config.json`) · 출처 메모 `RUN_FOLDER.txt`.
- 데이터(git 밖, `~/IsaacLab_RS/datasets/`): 조교 데이터와 파생물은 `datasets/ta_ref/`, 팀 데이터는 `datasets/team12_kgh/`. `record_demos.py` 에 기존 파일 이름을 주면 덮어쓰므로 세션마다 새 이름을 쓴다.
- 기계: 학습 · 평가는 워크스테이션 RTX 6000 (NVIDIA RTX 6000 Ada, Isaac Sim 5.1 렌더링 불가 — 렌더는 DGX Spark gx10-2). 일부 문서와 결과 파일 이름의 "A6000" 은 같은 기계다.
