"""Names and ID/OOD split of the self-made evaluation tasks (Isaac-Ant-Eval-<Name>-v0).

Pure Python on purpose: ``scripts/eval_suite.py`` loads this file directly without starting Isaac Sim.
The terrain/friction definitions live in ``eval_envs.py`` and must use exactly these keys.
"""

EVAL_SPLITS = {
    "Flat": "ID",
    "Grid": "ID",
    "GridTall": "OOD",
    "GridWide": "OOD",
    "GridNarrow": "OOD",
    "GridLowMu": "OOD",
    "GridHighMu": "OOD",
    "FlatIce": "OOD",
    "Stairs": "OOD",  # unseen for V1-V4; V5 trains on stairs (step 0.02-0.08) -> in-distribution for V5
    "Wave": "OOD",
    "Boxes": "OOD",
    "Rails": "OOD",
}
