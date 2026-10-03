#!/usr/bin/env bash
# Capture one-episode videos (play_one_episode.py, 1280x720, 960 steps) for the report into docs/media/.
# Each clip gets its own checkpoint copy dir because play_one_episode writes <ckpt_dir>/videos/play/rl-video-step-0.mp4.
# A clip whose name ends in "_fail" retries eval seeds until the episode ends early (a fall), so the failure mode
# is actually shown; every clip's seed, reward and steps go to docs/media/videos.csv.
#
#   conda activate lerobot-arena && cd assignment1_ant && bash scripts/capture_videos.sh [final]
#   ("final": only the submission model checkpoints/final on Isaac-Ant-Eval-<Name>-Team12-v0 terrains)
set -u
cd "$(dirname "$0")/.."
R=logs/rsl_rl/ant_rough
V2=$R/2026-09-29_19-13-05_v2_curr_mix_s44/model_2999.pt
BASE=checkpoints/baseline_flat/model_999.pt
OUT=docs/media
mkdir -p "$OUT" logs/video_ckpt
[ -f "$OUT/videos.csv" ] || echo "clip,task,checkpoint,seed,reward,steps" > "$OUT/videos.csv"

capture() {  # name task ckpt seeds...
  local name=$1 task=$2 ckpt=$3; shift 3
  local dir=logs/video_ckpt/$name
  for seed in "$@"; do
    rm -rf "$dir"; mkdir -p "$dir"; cp "$ckpt" "$dir/model.pt"
    ~/IsaacLab_RS/isaaclab.sh -p scripts/rsl_rl/play_one_episode.py --task "$task" --seed "$seed" --num_envs 1 \
      --headless --checkpoint "$dir/model.pt" --video --video_length 960 < /dev/null > "$dir/play.log" 2>&1
    local reward steps
    reward=$(grep -a "Episode reward total" "$dir/play.log" | grep -o '[-0-9.]*$')
    steps=$(grep -a "Episode steps" "$dir/play.log" | grep -o '[0-9]*$')
    echo "[CAPTURE] $name seed=$seed reward=$reward steps=$steps"
    if [[ "$name" == *_fail && "${steps:-960}" -ge 960 ]]; then continue; fi  # no fall in this seed: try the next
    cp "$dir/videos/play/rl-video-step-0.mp4" "$OUT/$name.mp4"
    echo "$name,$task,$(basename "$(dirname "$ckpt")")/$(basename "$ckpt"),$seed,$reward,$steps" >> "$OUT/videos.csv"
    return
  done
  echo "[CAPTURE] $name: no qualifying seed"
}

if [ "${1:-}" = final ]; then
  F=checkpoints/final/model_2999.pt
  capture final_plane Isaac-Ant-Team12-v0 "$F" 24
  capture final_grid Isaac-Ant-Eval-Grid-Team12-v0 "$F" 24
  capture final_rails Isaac-Ant-Eval-Rails-Team12-v0 "$F" 24
  capture final_stairs Isaac-Ant-Eval-Stairs-Team12-v0 "$F" 24
  capture final_pyramids Isaac-Ant-Eval-Pyramids-Team12-v0 "$F" 24
  capture final_flat Isaac-Ant-Eval-Flat-Team12-v0 "$F" 24
  capture final_flat_fail Isaac-Ant-Eval-Flat-Team12-v0 "$F" 24 25 26 27 28 29
  exit 0
fi

capture baseline_grid Isaac-Ant-Eval-Grid-v0 "$BASE" 24
capture v2_grid Isaac-Ant-Eval-Grid-v0 "$V2" 24
capture v2_rails Isaac-Ant-Eval-Rails-v0 "$V2" 24
capture v2_plane Isaac-Ant-v0 "$V2" 24
capture v2_flat_fail Isaac-Ant-Eval-Flat-v0 "$V2" 24 25 26 27 28 29
