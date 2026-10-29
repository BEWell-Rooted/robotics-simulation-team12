"""처치 없는 비교(null)를 가능한 짝 조합 전부로 만들어 null 띠를 계산한다 (사전등록 docs/prereg_1029.md 5절).

null 비교 = 같은 run 의 두 평가(학습 seed 101·111 각 100회, 평가 seed 42)를 seed 별로 하나씩 골라 200쌍으로 합친 것.
  - same_machine : 같은 기계에서 이웃 체크포인트 (A6000 e1900·e2000 / gx10-3 s101 e1800·e1900·e2000 + gx10-2 s111 e1900·e2000)
  - cross_machine: 같은 체크포인트를 A6000 과 Spark 에서 (e1900·e2000)
seed 별로 (A, B) 방향을 따로 고르므로 방향을 섞은 조합도 들어간다. 모든 비교를 paired_compare.py --json 으로 돌린 뒤
null_band.py 로 지표별 최대 |차| (건수는 100개당)를 띠로 쓴다.

  python -I null_enumerate.py            # results/noise_baseline/null_all/ 에 json·txt, null_band_1029.{md,json}
"""
import itertools
import os
import subprocess
import sys

A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = os.path.join(A2, "results", "noise_baseline")
B = os.path.join(A2, "results", "baseline")
OUT = os.path.join(N, "null_all")

# (기계, 학습 seed, 체크포인트) → 분류 CSV (옆에 .carry.csv 가 있어야 함)
CSV = {
    ("a6000", 101, 2000): f"{B}/base_kgh500_s101_e2000_seed42.csv",
    ("a6000", 111, 2000): f"{B}/base_kgh500_s111_e2000_seed42.csv",
    ("a6000", 101, 1900): f"{N}/a6000_base_kgh500_s101_e1900_seed42.csv",
    ("a6000", 111, 1900): f"{N}/a6000_base_kgh500_s111_e1900_seed42.csv",
    ("spark", 101, 1800): f"{N}/spark_gx10-3_base_kgh500_s101_e1800_seed42.csv",
    ("spark", 101, 1900): f"{N}/spark_gx10-3_base_kgh500_s101_e1900_seed42.csv",
    ("spark", 101, 2000): f"{N}/spark_gx10-3_base_kgh500_s101_e2000_seed42.csv",
    ("spark", 111, 1900): f"{N}/spark_gx10-2_base_kgh500_s111_e1900_seed42.csv",
    ("spark", 111, 2000): f"{N}/spark_gx10-2_base_kgh500_s111_e2000_seed42.csv",
}


def seed_pairs(kind):
    """학습 seed 별 (A, B) 후보: 방향이 있는 짝 목록."""
    out = {}
    for s in (101, 111):
        if kind == "same_machine":
            pairs = []
            for m in ("a6000", "spark"):
                cks = sorted(c for (mm, ss, c) in CSV if mm == m and ss == s)
                pairs += [((m, s, a), (m, s, b)) for a, b in itertools.permutations(cks, 2)]
        else:
            cks = sorted({c for (mm, ss, c) in CSV if ss == s and mm == "a6000"} & {c for (mm, ss, c) in CSV if ss == s and mm == "spark"})
            pairs = []
            for c in cks:
                pairs += [(("a6000", s, c), ("spark", s, c)), (("spark", s, c), ("a6000", s, c))]
        out[s] = pairs
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    jsons = {"same_machine": [], "cross_machine": []}
    for kind in jsons:
        sp = seed_pairs(kind)
        for p101, p111 in itertools.product(sp[101], sp[111]):
            # 같은 기계 비교는 두 seed 를 같은 기계에서 고른다 (seed 별로 다른 기계를 섞지 않음)
            if kind == "same_machine" and p101[0][0] != p111[0][0]:
                continue
            tag = "__".join(f"{a[0]}{a[1]}e{a[2]}-{b[0]}e{b[2]}" for a, b in (p101, p111))
            j = os.path.join(OUT, f"{kind}__{tag}.json")
            cmd = [sys.executable, "-I", os.path.join(A2, "tools", "paired_compare.py"),
                   "--a", CSV[p101[0]], CSV[p111[0]], "--b", CSV[p101[1]], CSV[p111[1]],
                   "--label_a", "A", "--label_b", "B", "--json", j]
            with open(j[:-5] + ".txt", "w") as f:
                subprocess.run(cmd, check=True, stdout=f)
            jsons[kind].append(j)
    print(f"same_machine {len(jsons['same_machine'])} · cross_machine {len(jsons['cross_machine'])} 비교")
    subprocess.run([sys.executable, "-I", os.path.join(A2, "tools", "null_band.py"), *jsons["same_machine"], *jsons["cross_machine"],
                    "--md", os.path.join(N, "null_band_1029.md"), "--json", os.path.join(N, "null_band_1029.json"),
                    "--summary"], check=True)


if __name__ == "__main__":
    main()
