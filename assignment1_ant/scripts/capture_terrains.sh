#!/usr/bin/env bash
# Still image per terrain -> docs/figures/terrains/<name>.png (see capture_terrains.py)
set -u
cd "$(dirname "$0")/.."
OUT=docs/figures/terrains
shot() {  # name task view
  timeout 600 ~/IsaacLab_RS/isaaclab.sh -p scripts/capture_terrains.py --task "$2" --out "$OUT/$1.png" --view "$3" --headless \
    < /dev/null > "logs/terrain_$1.log" 2>&1
  grep -a -q "\[TERRAIN\]" "logs/terrain_$1.log" && echo "[OK] $1" || echo "[FAIL] $1 (logs/terrain_$1.log)"
  pkill -9 -f "^/home/cai/anaconda3/envs/lerobot-arena/bin/python scripts/capture_terrains.py" 2>/dev/null
}
shot Plane Isaac-Ant-v0 close
for n in Flat Grid GridTall GridWide GridNarrow Stairs Wave Boxes Rails SlopedGrid Pyramids; do shot "$n" "Isaac-Ant-Eval-$n-v0" close; done
shot BoxPrim Isaac-Ant-Diag-BoxPrim-v0 close


# training terrains: see scripts/plot_terrain_layouts.py (renders from far away wash out)
