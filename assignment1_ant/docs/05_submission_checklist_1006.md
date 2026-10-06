# 05 · 제출 최종 체크리스트 (10/6, 마감 23:59 LMS)

대상: 코드 세션(Claude Code). `[사람]` 표시는 형근이 직접 하는 단계.
레포: `BEWell-Rooted/robotics-simulation-team12` (main) · 제출 폴더 `assignment1_ant/`

## 0. 검토 결과 (발표 자료 세션, 10/6 20:00)

- `eval_command.txt` — 최종 checkpoint `checkpoints/final/model_2999.pt`(v8_s44@2999), task `Isaac-Ant-Team12-v0`, TERRAIN 블록 교체 안내, 공식 평면 174.73 ± 12.67. **그대로 제출 가능.**
- `submission_env_cfg.py` — `AntTeam12EnvCfg`, docstring과 TERRAIN 주석 명확. 변경 없음.
- `checkpoints/final/` — model_2999.pt (4.4 MB) · params/ · exported/ · RUN_FOLDER.txt. 정상.
- `.gitignore` — `classmaterial/`(수업 자료 재배포 금지), `logs/`, `outputs/`, `*.mp4`, `results/diag_force_stats/` 제외. 정상. 단, **public 전환 전에 classmaterial이 한 번도 커밋되지 않았는지** 확인 필요(2절).
- README — 상태 문구가 "진행 중", 로그 링크가 log_1004까지, "남은 일" 절 잔존, 삭제 예정 문서 링크 포함 → 3절대로 갱신.
- 마지막 커밋: "발표 영상 재녹화 …" (log_1006 포함). 작업 트리에 미커밋 변경이 있는지 `git status`로 확인.

## 1. 정리 (형근 확인 완료 — 지워도 됨)

```bash
cd ~/robotics-simulation-team12
git rm -q assignment1_ant/docs/assignment1_slides_v0.pptx assignment1_ant/docs/assignment1_slides_v1.pptx \
          assignment1_ant/docs/assignment1_slides_v2.pptx assignment1_ant/docs/assignment1_slides_v3.pptx
git rm -q assignment1_ant/scripts/make_slides.py assignment1_ant/scripts/make_slides_v0.py assignment1_ant/docs/slides_draft_v0.md
git rm -rq assignment1_ant/checkpoints/candidate_v2_s44
git rm -q assignment1_ant/docs/02_weekend_plan_1002.md assignment1_ant/docs/interim_report_1001.md
```

- 삭제 후 `grep -rn "02_weekend_plan\|interim_report\|slides_draft_v0\|make_slides\|candidate_v2" assignment1_ant --include=*.md --include=*.py --include=*.sh --include=*.txt` 로 남은 참조를 찾아 README·로그에서 링크만 제거(로그 본문 서술은 그대로 둬도 됨).
- 지우지 않는 것: `docs/log_*.md`, `docs/00·01·03·04·05`, `docs/report_material.md`, `results/*.csv`, `checkpoints/baseline_flat`, `docs/figures/`, `docs/media/videos.csv`.

## 2. 최종 PPT

- 최종 PPT는 형근 노트북(Windows OneDrive) `…\로보틱스 시뮬레이션\처음 보는 지형에서도 걷는 Ant-v3.pptx` 에 있다 — 워크스테이션에는 없다.
- `[사람]` 레포에 넣으려면 그 파일을 워크스테이션 `assignment1_ant/docs/assignment1_slides_final.pptx` 로 복사(OneDrive 웹에서 받기 또는 scp). 복사되면 코드 세션이 크기 확인 후 커밋: **100 MB 초과면 커밋하지 말고** README에 "발표 자료는 LMS 제출본" 으로 적는다. 50 MB 초과는 경고만.
- 복사가 번거로우면 레포에는 넣지 않아도 된다(제출물 1은 코드·가중치, PPT는 LMS로 별도 제출). 그 경우 README 발표 자료 줄은 "발표 자료: LMS 제출 (10/6)" 로.

## 3. README 갱신

### 루트 `README.md`
- 표의 과제 1 행: "마감 10.06, 진행 중 (중간 보고 10.01)" → "10.06 제출 완료 · 10.08 발표. 최종 모델 `assignment1_ant/checkpoints/final/model_2999.pt`, 평가 명령 `assignment1_ant/eval_command.txt`".

### `assignment1_ant/README.md`
- 상단 링크 줄: 로그 범위를 `log_0929.md` ~ `log_1006.md`로, "계획: 02_weekend_plan…" 과 "중간 보고: interim_report…" 제거. 대신 "설계 메모: [`docs/01_improvement_plan_0930.md`] · 평가 기준 리뷰: [`docs/03_review_new_criteria_1005.md`] · 영상 재녹화 패치: [`docs/04_video_recapture_patch.md`]".
- "발표 재료" 줄에 "발표 자료: [`docs/assignment1_slides_final.pptx`]" 추가.
- 태스크 표에 한 행 추가: `Isaac-Ant-Eval-<Name>-Team12-Wide-v0` — 영상용 넓은 지형(12열 96 m), `scripts/rsl_rl/play_one_episode_video.py --center_spawn` 과 함께 사용.
- 명령 절의 영상 줄: `bash scripts/capture_videos.sh final_wide   # 최종 모델, 넓은 지형` 추가.
- 폴더 트리에서 `candidate_v2_s44/` 줄 삭제.
- 마지막 "## 남은 일 (10.06 제출까지)" 절 삭제.
- 결과 요약 표 아래에 한 줄 추가: "수치의 한계: 평가 지형 폭 32 m에서 빠른 정책 약 27%가 지형 밖으로 이탈(바깥 두 열이 가장자리 4 m에서 스폰) — 지형 안 로봇만 비교해도 순위 동일, `docs/log_1005.md`".

## 4. public 전환 전 안전 점검 (필수)

```bash
cd ~/robotics-simulation-team12
git ls-files classmaterial | head            # 비어 있어야 함 (수업 자료 재배포 금지)
git log --all --diff-filter=A --name-only --format= -- classmaterial | head   # 과거 커밋에도 없어야 함
git ls-files | grep -iE "\.pt$|\.mp4$|\.zip$|\.pdf$"   # .pt는 checkpoints 3개(final 1 + baseline 1)만, mp4·zip·pdf 없어야 함
grep -rniE "api[_-]?key|token|password|secret|ghp_" --include=*.py --include=*.sh --include=*.md --include=*.txt --include=*.yaml . | grep -v ".git/"   # 비어 있어야 함
git status --porcelain                      # 커밋할 것 확인
```

- `classmaterial`이 과거 커밋에 들어 있으면 **public 전환 중단**하고 보고(히스토리 정리 필요).

## 5. 커밋 · push

```bash
git add -A
git commit -m "제출 정리: 옛 슬라이드·생성 스크립트·비교용 checkpoint·진행 문서 삭제, README 갱신, 최종 PPT 추가"
git push origin main
git rev-parse --short HEAD   # 제출 링크 텍스트에 적을 commit
```

## 6. `[사람]` GitHub 웹에서 public 전환

- https://github.com/BEWell-Rooted/robotics-simulation-team12/settings → Danger Zone → Change repository visibility → Public.
- 전환 뒤 코드 세션에서 비로그인 확인: `curl -s -o /dev/null -w "%{http_code}\n" https://github.com/BEWell-Rooted/robotics-simulation-team12/tree/main/assignment1_ant` → 200.
- 추가 확인(권장): `cd /tmp && git clone --depth 1 https://github.com/BEWell-Rooted/robotics-simulation-team12.git t12 && ls t12/assignment1_ant/checkpoints/final && cat t12/assignment1_ant/eval_command.txt | head -5 && rm -rf t12` — 익명 clone으로 checkpoint·명령어가 보이는지.
- 평가 명령 재실행(정리 뒤에도 동작 확인, 1~2분): `cd ~/IsaacLab_RS && ./isaaclab.sh -p ~/robotics-simulation-team12/assignment1_ant/scripts/rsl_rl/play_one_episode.py --task Isaac-Ant-Team12-v0 --seed 24 --num_envs 100 --headless --checkpoint ~/robotics-simulation-team12/assignment1_ant/checkpoints/final/model_2999.pt` → mean ≈ 174.7.

## 7. LMS 제출 패키지 (형근 노트북에서 업로드)

PPT가 노트북에 있으므로 패키지는 노트북에서 모은다. 코드 세션은 텍스트 2개만 만들어 `assignment1_ant/submission/` 에 두고 커밋한다(노트북에서 GitHub로 내려받아 쓰기 위해).

```
assignment1_ant/submission/
├── 1_github_link.txt          ← 아래 내용 (commit 해시는 5절 push 뒤 채움)
└── 2_eval_command.txt         ← eval_command.txt 복사 + 맨 위에 `# <repo> = git clone 한 폴더 (예: ~/robotics-simulation-team12)`
```

`1_github_link.txt` 내용:
```
12조 (김경한 · 박형근 · 오준성) — 실습 과제 1: 처음 보는 지형에서도 걷는 Ant
GitHub: https://github.com/BEWell-Rooted/robotics-simulation-team12/tree/main/assignment1_ant
commit: <git rev-parse --short HEAD>
최종 checkpoint: assignment1_ant/checkpoints/final/model_2999.pt  (task Isaac-Ant-Team12-v0)
평가 명령: assignment1_ant/eval_command.txt  (조교 지형은 source/ant_rough/ant_rough/tasks/submission_env_cfg.py 의 TERRAIN 블록만 교체)
```

`[사람]` 노트북에서: 위 txt 2개 + `처음 보는 지형에서도 걷는 Ant-v3.pptx`(이름을 `team12_assignment1_slides.pptx` 로 바꿔도 됨) 를 LMS에 첨부, GitHub 링크는 본문에도 붙여넣기. 제출 후 스크린샷 1장 보관.

## 8. 마지막 로그

`docs/log_1006.md` 끝에 "제출" 절 추가: 정리한 파일 목록, 최종 commit, public 전환 시각, LMS 제출 시각.
