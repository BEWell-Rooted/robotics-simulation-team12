"""Top-down layout maps of the training terrains (V1, V2, V5) built by running their real terrain generators.

Each tile is colored by sub-terrain type, with its difficulty written inside; the spawn region and the walking
direction (+x) are marked. Runs the generator classes exactly as training does (no simulation is stepped), recording
(row, col) -> (type, difficulty) through a thin subclass.

    ~/IsaacLab_RS/isaaclab.sh -p scripts/plot_terrain_layouts.py --headless --out docs/figures/terrains/train_layouts.png
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="docs/figures/terrains/train_layouts.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

from ant_rough.tasks.ant_rough_env_cfg import AntRoughCurriculumEnvCfg, AntRoughEnvCfg, AntRoughV5EnvCfg  # noqa: E402

KO = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
KO_B = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
# one fixed color per sub-terrain family (grid widths share a hue, lighter = finer)
TYPE_STYLE = {
    "grid_fine": ("#9cc3ea", "블록 w0.30"),
    "grid_mid": ("#2a78d6", "블록 w0.45"),
    "grid_coarse": ("#1c4f8f", "블록 w0.95"),
    "uniform_noise": ("#eb6834", "노이즈"),
    "obstacles": ("#eda100", "장애물"),
    "slope": ("#1baf7a", "경사"),
    "stairs": ("#7a2e2e", "계단 w0.4"),
    "stairs_narrow": ("#b36a67", "계단 w0.3"),
    "flat": ("#d8cdbf", "평지"),
}


def record_layout(env_cfg):
    gen_cfg = env_cfg.scene.terrain.terrain_generator
    names = {id(c): n for n, c in gen_cfg.sub_terrains.items()}
    base = gen_cfg.class_type
    tiles = {}

    class Recording(base):
        def _get_terrain_mesh(self, difficulty, cfg):
            self._last = (names[id(cfg)], difficulty)
            return super()._get_terrain_mesh(difficulty, cfg)

        def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
            tiles[(row, col)] = self._last
            return super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    gen_cfg.class_type = Recording
    Recording(gen_cfg, device="cpu")
    lane = base.__name__ == "LaneTerrainGenerator"
    return gen_cfg, tiles, lane, env_cfg.scene.terrain.max_init_terrain_level


def draw(ax, title, subtitle, gen_cfg, tiles, lane, max_init):
    sx, sy = gen_cfg.size
    for (row, col), (name, diff) in tiles.items():
        # base generator: row -> x, col -> y; lane generator: col -> x, row (= level) -> y
        ix, iy = (col, row) if lane else (row, col)
        color, _ = TYPE_STYLE[name]
        ax.add_patch(Rectangle((ix * sx, iy * sy), sx, sy, facecolor=color, edgecolor="#fcfcfb", linewidth=0.8))
        dark = name in ("grid_mid", "grid_coarse", "stairs", "uniform_noise", "slope")
        ax.text(ix * sx + sx / 2, iy * sy + sy / 2, f"{diff:.1f}", ha="center", va="center", fontsize=6,
                color="#ffffff" if dark else "#241e19")  # fmt: skip
    nx = gen_cfg.num_cols if lane else gen_cfg.num_rows
    ny = gen_cfg.num_rows if lane else gen_cfg.num_cols
    # spawn region
    if lane:
        spawn = Rectangle((0, 0), gen_cfg.num_spawn_cols * sx, (max_init + 1) * sy)
    else:
        spawn = Rectangle((0, 0), (max_init + 1) * sx, ny * sy)
    spawn.set(facecolor="none", edgecolor="#241e19", linewidth=1.6, linestyle=(0, (4, 2)))
    ax.add_patch(spawn)
    ax.add_patch(FancyArrow(nx * sx * 0.02, -sy * 0.9, nx * sx * 0.25, 0, width=sy * 0.18, head_width=sy * 0.5,
                            head_length=sx * 1.2, color="#241e19", length_includes_head=True))  # fmt: skip
    ax.text(nx * sx * 0.30, -sy * 0.9, "+x 전진 방향 (16 s에 ~120 m)", va="center", fontproperties=KO, fontsize=8)
    ax.set_xlim(-1, nx * sx + 1)
    ax.set_ylim(-sy * 1.6, ny * sy + sy * 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontproperties=KO_B, fontsize=12, pad=4)
    ax.text(0, ny * sy + sy * 0.35, subtitle, fontproperties=KO, fontsize=8, color="#55493f")
    if lane:
        ax.text(-sy * 0.3, ny * sy / 2, "차선(level) → 난이도 ↑", rotation=90, va="center", ha="right",
                fontproperties=KO, fontsize=8, color="#55493f")  # fmt: skip


def main():
    specs = [
        ("V1 DR", "타일마다 지형 종류·난이도 무작위 · 20×16 타일(8 m) · 점선 = 스폰 구역(앞쪽 8행)", AntRoughEnvCfg()),
        ("V2 차선 커리큘럼", "차선(y)마다 난이도 고정, 지형 종류는 열(x) 순서 고정 → 평지가 트랙 끝에만 · 점선 = 시작 차선 0–2 × 앞 4열",
         AntRoughCurriculumEnvCfg()),
        ("V5", "V2 차선 + 타일마다 지형 종류 무작위 + 계단 2종 + 평지 20%", AntRoughV5EnvCfg()),
    ]  # fmt: skip
    fig, axes = plt.subplots(len(specs), 1, figsize=(13, 15), dpi=150, gridspec_kw={"height_ratios": [16, 10, 10]})
    fig.patch.set_facecolor("#faf6f0")
    for ax, (title, sub, cfg) in zip(axes, specs):
        ax.set_facecolor("#faf6f0")
        draw(ax, title, sub, *record_layout(cfg))
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c, _ in TYPE_STYLE.values()]
    fig.legend(handles, [lbl for _, lbl in TYPE_STYLE.values()], loc="lower center", ncol=9, frameon=False,
               prop=KO, bbox_to_anchor=(0.5, 0.005))  # fmt: skip
    fig.text(0.01, 0.985, "학습 지형 배치도 — 칸 = 8 m 타일, 칸 안 숫자 = 난이도(0–1)", fontproperties=KO_B, fontsize=14,
             va="top")  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(args_cli.out, facecolor="#faf6f0")
    print(f"[LAYOUT] {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
