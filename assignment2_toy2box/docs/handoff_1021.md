# 실습과제 2 Toy2Box A6000 세팅·검증 — Claude Code 브리프 본문 (v1.1, 2026-10-21)

원본: Cowork 인계 브리프 v1.1 §4 코드 블록. 아래는 그대로 옮긴 것이다.

~~~
# 역할
너는 연구실 워크스테이션(Precision 7960, RTX 6000 Ada 48GB)에서 로보틱스 시뮬레이션 실습과제 2(Agibot Toy2Box 모방학습)의 환경 세팅과 파이프라인 검증을 맡는다.
오늘 범위는 설치·검증까지다. 장시간 학습·생성·평가는 하지 않는다.
형근은 지금 워크스테이션 앞에 있다. 승인 게이트에서는 멈추고 묻는다.

# 먼저 할 것
- 아래에서 A2 는 ~/robotics-simulation-team12/assignment2_toy2box 를 뜻한다. A2 안의 파일은 항상 이 절대경로로 쓴다 (작업 위치가 ~/IsaacLab_RS 이므로 상대경로 금지).
- 이 블록 전체를 ~/robotics-simulation-team12/assignment2_toy2box/docs/handoff_1021.md 로 저장한다 (폴더가 없으면 만든다: docs/, docs/env/, docs/run_logs/, tools/).
- A2/docs/log_1021.md 를 만들고 아래 "로그 형식"의 상단 블록을 먼저 채운 뒤, 단계마다 즉시 기록한다.

# 절대 규칙
- sudo, NVIDIA 드라이버·CUDA·cuDNN 설치/변경, ~/.bashrc 수정 금지.
- ~/IsaacLab(기존 연구용)과 lerobot-arena 외 conda env 수정 금지.
- ~/IsaacLab_RS 추적 파일 수정 금지. git checkout/pull로 체크아웃을 바꾸는 것도 승인 게이트(형근 승인) 뒤에만.
- 모든 명령은 `conda activate lerobot-arena` 상태, 작업 위치 ~/IsaacLab_RS 에서.
- Isaac Sim 을 띄우는 모든 명령(list_envs 포함, 첫 실행은 셰이더 캐시로 몇 분 걸릴 수 있음)은 tmux 세션 rs_a2 에서 `timeout 900 <명령> 2>&1 | tee A2/docs/run_logs/<단계>_<이름>.log` 형태로 실행하고, 로그 파일을 주기적으로 읽어 끝났는지 본다. 세션 셸 백그라운드(&, nohup)는 쓰지 않는다.
- 실행을 멈출 때는 tmux send-keys -t rs_a2 C-c 를 쓴다.
- 스모크는 각 10분 안에 끝낸다. Mimic 500, epoch 2000 학습, rollout 100회는 실행하지 않는다.
- --dataset_file 에 이미 있는 파일 이름을 주지 않는다 (record_demos 는 같은 이름이면 파일을 새로 만든다).
- 같은 실패를 3번 넘게 반복하지 않는다. 기록하고 다음 단계로 가거나 멈춘다.
- git push 금지. 로컬 커밋까지만.

# 단계 0 — 사전 점검·스냅샷 (15분)
- nvidia-smi 로 GPU 사용 중인 다른 프로세스와 여유 VRAM을 기록한다. 여유가 20 GB 미만이면 형근에게 알린다.
- 버전 출력: python --version / torch 버전·CUDA / importlib.metadata 로 isaacsim, isaacsim-core, isaacsim-app / numpy / lerobot.__version__.
- pip freeze > ~/robotics-simulation-team12/assignment2_toy2box/docs/env/pip_freeze_before_1021.txt
- ~/IsaacLab_RS 에서: git status --short, git describe --tags --always, git log -1 --oneline, git remote -v, git branch -a.
- ./isaaclab.sh -p scripts/environments/list_envs.py 출력에서 "Toy2Box" 줄을 찾는다. 두 ID가 모두 있어야 한다:
  Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 / Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0
- find source -ipath "*agibot*" -name "*.json" 로 robomimic config(조교 문서상 .../place/config/agibot/agents/robomimic/bc_rnn_low_dim.json)를 찾고, train.seed 값과 train.num_epochs 를 기록한다.
- 스크립트 7개 존재 확인: scripts/environments/teleoperation/teleop_se3_agent.py, scripts/tools/record_demos.py, scripts/tools/replay_demos.py, scripts/imitation_learning/isaaclab_mimic/annotate_demos.py, scripts/imitation_learning/isaaclab_mimic/generate_dataset.py, scripts/imitation_learning/robomimic/train.py, scripts/imitation_learning/robomimic/play.py (+ scripts/tools/merge_hdf5_datasets.py).
- train.py 와 play.py 의 argparse 를 읽고 기록한다: seed 를 CLI로 줄 수 있는가(없으면 json train.seed 로만 바뀌는가), 로그·체크포인트 저장 경로 규칙, play.py 가 --video 때 enable_cameras 를 자동으로 켜는가.
- [승인 게이트 · 형근] 태스크 두 ID 또는 agibot robomimic json 이 없으면 멈춘다.
  git fetch origin --tags 후 `git log --oneline HEAD..origin/<기본 브랜치> | head -30` 과 `git diff --stat HEAD origin/<기본 브랜치> | grep -i agibot` 를 보여 주고, 체크아웃을 최신으로 옮길지 묻는다.
  승인되면 현재 커밋 해시를 log에 적고 git switch -c rs-latest origin/<기본 브랜치> 후 ./isaaclab.sh -i 는 실행하지 말고 list_envs 만 다시 확인한다 (설치 재실행이 필요해 보이면 그것도 묻는다).
- log 기록: 버전 표, GPU 상태, 커밋, 태스크·json·스크립트 존재 여부, seed 지정 방법.

# 단계 1 — robomimic 설치 (20분)
- ./isaaclab.sh -i robomimic
- egl_probe 또는 cmake 관련 빌드 오류가 나면(가정 — 자료에 없는 대처): pip install cmake 후 한 번만 재시도한다. 그래도 실패하면 오류 마지막 40줄을 log에 붙이고 멈춘다.
- python -c "import robomimic, sys; print(robomimic.__version__, robomimic.__file__)"
- pip freeze > ~/robotics-simulation-team12/assignment2_toy2box/docs/env/pip_freeze_after_robomimic_1021.txt
  diff 로 바뀐 패키지를 log에 적는다.
- torch 가 2.7.0+cu128 이 아니게 됐거나 isaacsim 버전이 바뀌었으면 멈추고 형근에게 보고한다 (되돌리지 말 것).
- numpy 가 단계 0 값과 달라졌으면(과제 1 브리프 기록상 1.26.0 고정) 단계 0 값으로 되돌리고 기록한다.
- 단계 0의 버전 출력을 다시 실행해 전후 표를 log에 적는다.

# 단계 2 — 조교 데이터 검증 (20분)
- 파일 확인: datasets/ta_ref/toy2box_agibot.hdf5 크기 8148416 바이트, md5 a1d3391db6db3f55c2d19d6118570eb9.
  없으면 형근에게 H1(파일 복사)을 요청하고 기다린다.
- 아래 스크립트를 A2/tools/summarize_demos.py 로 저장한다.
  (스크립트 본문은 A2/tools/summarize_demos.py 참조 — 브리프 원문과 동일)
- 실행: ./isaaclab.sh -p ~/robotics-simulation-team12/assignment2_toy2box/tools/summarize_demos.py datasets/ta_ref/toy2box_agibot.hdf5
- 기대 출력(Week 08 강의자료·조교 문서와 동일, 실제 출력은 칸 맞춤 공백이 더 있다 — 공백을 무시하고 비교):
  episodes: 20 | total: 10891
  demo_0 len=634 (31.7s) sw=2 idle=40% ok=1
  demo_4 len=624 (31.2s) sw=12 idle=38% ok=1
  demo_12 len=350 (17.5s) sw=2 idle=25% ok=1
  demo_19 len=454 (22.7s) sw=3 idle=22% ok=1
  총계 줄과 네 줄이 같으면 통과. 다르면 출력 전체를 log에 붙인다.
- replay 영상 1개 (첫 에피소드만):
  ./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --dataset_file ./datasets/ta_ref/toy2box_agibot.hdf5 --num_envs 1 --select_episodes 0 --video --video_folder ./videos/ta_ref_replay --headless
  --select_episodes 는 demo 이름이 아니라 순서 번호일 가능성이 높다(가정). h5py 키는 사전순이라 번호 19는 demo_9 다. 0번은 어느 쪽이든 demo_0(634 step)이므로 로그의 길이로 어느 시범이 재생됐는지 확인해 적는다.
  --select_episodes 가 인자 오류면 빼고 실행하되, 첫 영상이 생기면 tmux send-keys 로 멈춘다.
  headless 영상이 카메라 관련 오류로 실패하면 --enable_cameras 를 붙여 한 번 재시도한다.
- log 기록: 요약 일치 여부, mp4 경로·크기, 영상 속 트럭이 상자에 들어갔는지(로그의 성공 표시가 있으면 그것을 인용).
- 이 단계가 끝나면 형근에게 "H2 텔레옵 확인을 시작해도 된다"고 알리고 아래 H2 명령 두 개를 그대로 보여 준다. 단계 3은 기다리지 말고 바로 진행한다.

# H2 — 형근 키보드 텔레옵·녹화 (승인 게이트 · 형근, 20분, 단계 3과 병행)
형근이 자기 터미널에서 직접 실행한다. 너는 명령만 보여 주고, 끝났다는 말을 들으면 결과 파일을 검사한다.
- 명령을 보여 주기 전에 ls datasets/team12_park_s0.hdf5 로 같은 이름 파일이 없는지 확인한다. 있으면 s0b 처럼 새 이름으로 바꿔 보여 준다.
- H2 와 단계 3 이 같은 GPU를 쓰므로, 단계 3의 속도 수치 옆에 'H2 동시 실행 여부'를 적는다.
- 연습(저장 없음):
  ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --num_envs 1 --teleop_device keyboard
- 녹화 1개(테스트 세션 s0):
  ./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --dataset_file ./datasets/team12_park_s0.hdf5 --num_demos 1 --teleop_device keyboard
- 결과 검사: datasets/team12_park_s0.hdf5 가 있으면 summarize_demos.py 로 요약해 log에 붙인다.
- 형근이 "건너뛴다"고 하면 H2 = 미실시로 적고 넘어간다.

# 단계 3 — 파이프라인 스모크 (40분)
3-1 annotate (조교 20개 전체, 짧음)
- ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --auto --input_file ./datasets/ta_ref/toy2box_agibot.hdf5 --output_file ./datasets/ta_ref/toy2box_agibot_annotated.hdf5
- 출력 hdf5의 demo 수(20 기대)와 소요 시간을 기록한다. annotate 가 실패한 demo가 있으면 번호를 적는다.
3-2 Mimic 생성 스모크 (성공 4개)
- ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 4 --generation_num_trials 4 --input_file ./datasets/ta_ref/toy2box_agibot_annotated.hdf5 --output_file ./datasets/ta_ref/_smoke_mimic_4.hdf5
- 10분 안에 안 끝나면 멈추고 그때까지의 진행 로그를 기록한다.
- --generation_num_trials 가 성공 시범 수인지 시도 횟수인지는 확인되지 않았다(가정: 성공 수). 출력 파일과 _failed 파일의 demo 수로 어느 쪽인지 판정해 적는다.
- 기록: 걸린 시간, 출력 파일 demo 수, _failed 파일 demo 수, 성공 시범/분, H2 동시 실행 여부. 이 값으로 "Mimic 500 예상 시간(추정)"을 한 줄 쓴다 (num_envs 100이면 병렬 효과가 있어 단순 비례가 아님을 함께 적는다).
3-3 BC 학습 시작 스모크
- tmux 세션 rs_a2_smoke 에서:
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc --dataset ./datasets/ta_ref/_smoke_mimic_4.hdf5
- 3-2가 실패했거나, 4개짜리 데이터에서 train/valid 필터 키 등으로 학습이 바로 오류를 내면 --dataset ./datasets/ta_ref/toy2box_agibot.hdf5 (원본 20개)로 대신 실행하고 그렇게 했다고 적는다.
- epoch 로그가 3번 찍히거나 10분이 지나면 Ctrl-C로 멈춘다.
- 기록: 초/epoch, 2000 epoch 예상 시간(추정), 로그·체크포인트 폴더 경로, 그 폴더에 저장된 config(json)의 train.seed 값, H2 동시 실행 여부.
- 스모크 run 폴더는 이름 앞에 _smoke 가 드러나도록 옮기거나 log에 경로를 적어 나중에 지울 수 있게 한다.
3-4 play 준비 확인
- 체크포인트가 없으므로 실행하지 않는다. 단계 0에서 읽은 play.py 인자와 강의자료 44쪽 명령을 대조해 다른 점만 log에 적는다:
  --task ...-RmpFlow-v0 --checkpoint <pth> --num_rollouts 100 --horizon 800 --video --video_length 1000 --video_folder ./videos/agibot_bc_mimic --seed 42 --headless

# 단계 4 — 리포트·Notion 기록·커밋 (15분)
- A2/docs/log_1021.md 를 아래 "로그 형식"대로 마무리한다. 브리프와 다르게 한 것이 있으면 그 절에 적는다.
- A2/README.md 에 3줄: 과제 이름·마감, 폴더 구조(docs/, tools/), 데이터 위치 규칙(datasets/ta_ref/, datasets/team12_<이름>_s<세션>.hdf5).
- A2/.gitignore 를 만들어 *.hdf5, *.mp4, *.pth, logs/, videos/ 를 넣는다. run_logs/*.log 는 커밋한다.
- 아래 "단계 N — Notion 기록"을 수행한다.
- 로컬 커밋: git -C ~/robotics-simulation-team12 add assignment2_toy2box && git -C ~/robotics-simulation-team12 commit -m "assignment2: A6000 setup check and demo summary tool (1021)". push 하지 않는다.
- 정리: tmux 세션 rs_a2, rs_a2_smoke 가 남아 있으면 종료한다. 스모크 hdf5(_smoke_*)는 지우지 않는다.

### 단계 N — Notion 기록 (리포트·logs/ 요약을 다 쓴 뒤, 커밋 전에)
- 이 세션에 Notion 도구(notion-update-page)가 있는지 확인한다.
- 있으면: Notion Task https://app.notion.com/p/40098990508d813c8235fc6ee82fff98 페이지 본문 끝에 아래 블록을 insert_content(position end)로 추가한다.
  - 앞에 구분선 --- 하나.
  - 속성(완료·날짜·상황)은 건드리지 않는다. 기존 블록도 건드리지 않는다.
  - 추가 뒤 log 메타 표의 "Notion 기록"을 "완료"로 쓴다.
- 없거나 실패하면: log 메타 표의 "Notion 기록"을 "미기록"으로 쓰고 넘어간다.
  - 다음 Cowork 세션의 task-session-note가 log 요약을 옮긴다. 재시도하지 않는다.

블록 형식 (log 요약 다섯 줄을 옮긴다. 해석을 보태지 않는다):
### 📝 YY-MM-DD (요일) 세션 노트 · Claude Code · 실행
**결과**
- ⚠ <결론 한 문장>
- ⚠ <실측 — 숫자·단위·조건 그대로. 실측 없으면 이 줄 생략>
**막힌 것 / 열린 질문**
- <막힌 것. 없으면 섹션 생략>
**판단 질문** (답을 써야 완료)
- [ ] <형근이 결론을 받아들이기 전에 직접 열어 봐야 할 숫자·파일 하나>
**다음 행동** `<연구실 / 컴퓨터 등>`
- <다음 세션 첫 행동>
메모: 원본 assignment2_toy2box/docs/log_1021.md · 문서 수정 제안: <없으면 이 부분 생략>

- 결과 줄은 전부 ⚠다. 형근이 직접 확인하기 전이기 때문이다. ✅를 쓰지 않는다.
- 메모 줄의 "원본 ..." 경로는 중복 방지 키이므로 빼지 않는다.
- 판단 질문에 답을 쓰지 않는다.

# 로그 형식 (docs/log_1021.md)
상단 고정 블록:
| 항목 | 값 |
| 브리프 | docs/handoff_1021.md v1.1 |
| 머신 | Precision 7960 · RTX 6000 Ada |
| 시작–종료 | HH:MM–HH:MM |
| Notion Task | https://app.notion.com/p/40098990508d813c8235fc6ee82fff98 |
| Notion 기록 | 완료 / 미기록 |
다섯 줄 요약: 결론 / 실측 / 막힌 것 / 판단 질문 / 다음 행동 (한 줄씩).
작업 설명 (요약 바로 뒤, 본문 표 앞):
- 무엇을 왜 했나 — 3줄 이내, 결과가 아니라 방법.
- 핵심 그림 1장 — 입력 → 처리 → 산출물 흐름 (조교 hdf5 → summarize/replay → annotate → Mimic → train). 이미지가 없으면 텍스트 화살표 한 줄로 대신한다.
- 리뷰 때 이해해야 할 개념 2–3개 — 각 한 줄 정의 (예: Mimic 구간 annotation, BC-RNN, Rel-Mimic 태스크).
본문: 단계별 표(단계 · 시각 · rc · 판정 · 산출물 경로) + 단계별 상세 + "브리프와 다르게 한 것" + "다음 세션에 필요한 수치"(Mimic 속도, 초/epoch, seed 지정 방법, 2000 epoch·Mimic 500 예상 시간 — 모두 추정 표기).
~~~
