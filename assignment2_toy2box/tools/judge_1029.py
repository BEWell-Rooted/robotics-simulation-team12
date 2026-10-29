"""사전등록 docs/prereg_1029.md 의 판정(§5.2 확증 · §5.3 귀속 · §5.4 C3 · §6 해석표 · §7 제출 후보)을 한 번에 돌린다.

판정 규칙은 여기서 새로 정하지 않는다. 사전등록 문구를 그대로 옮겼고, 비교는 모두 tools/paired_compare.py 로 한다
(--require_carry, 시작 상태 짝). 띠 값은 results/noise_baseline/null_band_1029.json 의 band 를 그대로 쓴다.

  python -I tools/judge_1029.py                      # results/judge_1029/ 에 비교별 txt·json + verdict_1029.{md,json}
  python -I tools/judge_1029.py --out <dir> --eseeds 42 --pattern 'A-T=results/noise_baseline/a6000_base_kgh500_s{s}_e1900_seed{e}.csv'
                                                     # 도구 시험용 (null 데이터를 팔 자리에 넣는다)

파일이 없는 비교는 건너뛰고 '없음' 으로 적는다. 팔 판정에 필요한 비교(주 비교 · 음성 대조 · e1900 민감도 · 시작 거리 하위 분석)가
하나라도 없으면 그 팔은 '판정 미완' 이다.

10/30 판정 뒤 출력 표기만 보완 (판정 로직·json 의 기존 값은 그대로, 사전등록 §7 처치 시작 뒤 변경 3):
귀속 방향을 팔 이름으로, §6 행에 해석 문구, 체크포인트 민감도 '진행 중' 표시(ALL_DONE), 함께 읽기 '해당 없음', 제출 체크포인트 실제 경로·md5.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(A2, "tools", "paired_compare.py")
TRAIN_SEEDS = (101, 111)

# 팔 → 분류 CSV 경로 틀 ({s} 학습 seed, {ep} 체크포인트 epoch, {e} 평가 seed). A2 기준 상대 경로
PATTERNS = {
    "baseline": "results/baseline/base_kgh500_s{s}_e{ep}_seed{e}.csv",
    "A-T": "results/runs/base_translate500_s{s}_e{ep}_seed{e}.csv",
    "B2": "results/runs/base_reach4_s{s}_e{ep}_seed{e}.csv",
    "A-R": "results/runs/base_rotworld500_s{s}_e{ep}_seed{e}.csv",
    "B3'": "results/runs/base_dirorig_magb2_s{s}_e{ep}_seed{e}.csv",
}
BASE_E1900 = "results/noise_baseline/a6000_base_kgh500_s{s}_e1900_seed{e}.csv"  # P0 (평가 seed 42 만)
MIN_PRED = {"A-T": 4.0, "B2": 4.0}  # ② 첫 조건의 최소 예측 (C-1mode 는 없음)
CONFIRM = ("A-T", "B2", "C-1mode")  # Holm 3팔 고정. 돌리지 않은 팔은 p = 1
ATTR_MARGIN = 7.0  # §5.3 동등 여유 ±7%p
OLD_CRIT = 13.0  # 이전 기준 (참고만, §5.5)
STATUS_LOG = os.path.expanduser("~/IsaacLab_RS/logs/rs_a2_runs/queue_1029/status.log")
CKPT_ROOT = os.path.expanduser("~/IsaacLab_RS/logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0")
# §6 해석 칸 (사전등록 문구). 출력 표기용 — 행 선택에는 쓰지 않는다
ROW_TEXT = {
    "H-A0": "H-A 검정 안 됨 — 기각 아님, '게이트 미통과' 로 보고",
    "H-A1": "H-A 기각 (데이터 게이트 통과 확인). A-R 은 해석하지 않는다",
    "H-A2": "재타깃팅의 회전 성분 (위치 규칙의 몫은 작다)",
    "H-A3": "재타깃팅의 위치·우회 성분이 몫을 가진다",
    "H-A4": "상자 구간 재타깃팅(회전 + 우회) 전체로만 결론",
    "H-A5": "판단 보류 (표본 한계)",
    "H_B0": "H_B 검정 안 됨 — 보고만",
    "H_B1": "H_B 기각 (B3' 귀속은 H_B2–4 에서만 쓰므로 해석하지 않는다 — 보고만)",
    "H_B2": "라벨의 명령 크기가 몫",
    "H_B3": "라벨의 방향·일관성이 몫",
    "H_B4": "라벨 정의 전체로만 결론",
    "H_B5": "판단 보류",
    "H_C0": "H_C 검정 안 됨 — 보고만",
    "H_C1": "그리퍼 소수 성분 샘플이 들기 이후 실패의 독립 원인 일부",
    "H_C2": "판단 보류 (기각 아님)",
}


def attr_named(label, a, b):
    """귀속 결과의 a/b 를 팔 이름으로 (출력 표기용). a = 짝 팔(대조), b = 귀속 팔"""
    if not label:
        return label
    return label.replace("a > b", f"{a} > {b}").replace("b > a", f"{b} > {a}")


def submission_ckpt(pattern):
    """제출 팔의 학습 seed 101 e2000 체크포인트 (run_baseline.sh 처럼 run 폴더가 여럿이면 마지막). 출력 표기용"""
    import glob, hashlib, re
    m = re.search(r"base_(.+?)_s\{s\}", pattern)
    if not m:
        return None, None, 0
    hits = sorted(glob.glob(os.path.join(CKPT_ROOT, f"base_{m.group(1)}_s101", "*", "models", "model_epoch_2000.pth")))
    if not hits:
        return None, None, 0
    with open(hits[-1], "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return hits[-1], md5, len(hits)


def paths(pattern, eseeds, ep=2000):
    return [os.path.join(A2, pattern.format(s=s, ep=ep, e=e)) for s in TRAIN_SEEDS for e in eseeds]


def ready(ps):
    return all(os.path.exists(p) and os.path.exists(p[:-4] + ".carry.csv") for p in ps)


def run_pc(out, name, a, b, label_a, label_b, extra=()):
    """paired_compare.py 를 돌려 <out>/<name>.{txt,json} 을 쓰고 json 을 돌려준다. 파일이 없으면 None."""
    if not (ready(a) and ready(b)):
        missing = [os.path.relpath(p, A2) for p in a + b if not os.path.exists(p)]
        print(f"[없음] {name}: {len(missing)}개 파일 없음 (예: {missing[:1]})")
        return None
    js = os.path.join(out, name + ".json")
    cmd = [sys.executable, "-I", PC, "--require_carry", "--label_a", label_a, "--label_b", label_b, "--json", js,
           "--a", *a, "--b", *b, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=A2)
    with open(os.path.join(out, name + ".txt"), "w") as f:
        f.write("$ " + " ".join(os.path.relpath(c, A2) if c.startswith(A2) else c for c in cmd[1:]) + "\n\n")
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"paired_compare 실패 ({name}): {r.stderr.strip()[-400:]}")
    with open(js) as f:
        return json.load(f)


def holm(pvals):
    """Holm 보정. pvals {이름: p} → {이름: 보정 p}"""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, run, adj = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def per100(x, n):
    return 100.0 * x / n


def confirm_grade(main, neg, p_holm, arm, band):
    """§5.2: 위에서부터 처음 맞는 하나. 반환 (등급, 이유 목록, 덧붙임 목록)"""
    n, d = main["n"], main["delta_pp"]
    floor = max(6.0, band["success_pp"])
    seeds_pos = all(v > 0 for v in main["delta_by_train_seed"].values())
    med = main["mediators"]
    why, notes = [], []
    c1 = p_holm < 0.05 and d > floor and seeds_pos
    why.append(f"① Holm p {p_holm:.4g} < 0.05 · Δ {d:+.2f} > {floor:.2f} · 학습 seed 별 Δ "
               f"{', '.join(f's{k} {v:+.1f}' for k, v in main['delta_by_train_seed'].items())} 모두 > 0 → {c1}")
    up95 = main["ci95"][1]
    c2a = arm in MIN_PRED and up95 < MIN_PRED[arm]
    c2b = d <= -5.0 and main["p_two"] < 0.05
    why.append(f"② 95% 상한 {up95:+.2f} < 최소 예측 {MIN_PRED.get(arm, '없음')} → {c2a} · (Δ ≤ −5 이고 보정 전 p {main['p_two']:.4g} < 0.05) → {c2b}")
    bp = per100(med["box_push"]["diff"], n)
    la = per100(med["low_approach"]["diff"], n)
    c3 = d >= 5.0 and main["bundle_p_one"] < 0.10 and bp < -band["box_push"] and la <= band["low_approach"]
    why.append(f"③ Δ {d:+.2f} ≥ 5 · 표적 묶음 단측 p {main['bundle_p_one']:.4g} < 0.10 · box_push {bp:+.2f}/100 < −{band['box_push']:.2f} · "
               f"low_approach {la:+.2f}/100 ≤ {band['low_approach']:.2f} → {c3}")
    grade = "①" if c1 else "②" if (c2a or c2b) else "③" if c3 else "④"
    # 막는 쪽 조건의 주 (판정과 별도로 적는다)
    if la > band["low_approach"] / 2:
        notes.append(f"low_approach 증가 {la:+.2f}/100 이 띠의 절반({band['low_approach'] / 2:.2f})을 넘음")
    pl = per100(main["pre_lift_fail_b"] - main["pre_lift_fail_a"], n)
    if pl > band["pre_lift_fail"] / 2:
        notes.append(f"들기 전 실패 증가 {pl:+.2f}/100 이 띠의 절반({band['pre_lift_fail'] / 2:.2f})을 넘음")
    viol = None
    if neg is not None:
        inc = per100(neg["bundle_b"] - neg["bundle_a"], neg["n"])
        viol = inc > band["pre_lift_fail"] and neg["bundle_p_two"] < 0.05
        why.append(f"음성 대조: 들기 전 실패 {neg['bundle_a']} → {neg['bundle_b']} ({inc:+.2f}/100 > {band['pre_lift_fail']:.2f}) · "
                   f"양측 p {neg['bundle_p_two']:.4g} < 0.05 → 위반 {viol}")
    return grade, why, notes, viol


def attribution(js):
    """§5.3: 동등(90% CI 가 ±7 안) → 차이(보정 전 양측 p < 0.05) → 구분 불가. power_sim.py 와 같은 순서·경계."""
    lo, hi = js["ci90"]
    if -ATTR_MARGIN < lo and hi < ATTR_MARGIN:
        return "동등"
    if js["p_two"] < 0.05:
        return "차이 (" + ("b > a" if js["delta_pp"] > 0 else "a > b") + ")"
    return "구분 불가"


def status_lines():
    try:
        with open(STATUS_LOG) as f:
            return f.read().splitlines()
    except OSError:
        return []


# 그 팔을 끝내 돌리지 않았다고 큐가 적은 줄 (게이트 최종 실패 · md5 불일치). 메인 큐의 "A-T 학습 건너뜀" 은
# 보정 경로가 이어 돌렸으므로 미실행이 아니다 (prereg §7 처치 시작 뒤 변경 1).
NOT_RUN_MARK = {"A-T": ("A-T 보정 실패",), "A-R": ("A-R 보정 실패",),
                "B2": ("md5 불일치", "reach4"), "B3'": ("md5 불일치", "dirorig_magb2")}


def not_run(arm, lines):
    marks = NOT_RUN_MARK[arm]
    return any(all(m in line for m in marks) for line in lines)


def gate_passed(tag, lines):
    return any(f"{tag} 데이터 게이트 통과" in line for line in lines)


def official(pattern, ep=2000, e=42):
    """§5.5 공식 수치: 평가 seed 42, 학습 seed 별 성공 수/rollout 수, 평균 ± 모표준편차 (표본표준편차)"""
    rates = []
    for s in TRAIN_SEEDS:
        p = os.path.join(A2, pattern.format(s=s, ep=ep, e=e))
        if not os.path.exists(p):
            return None
        with open(p) as f:
            rows = list(csv.DictReader(f))
        rates.append(100.0 * sum(int(r["success"]) for r in rows) / len(rows))
    return dict(rates=rates, mean=statistics.mean(rates), pstd=statistics.pstdev(rates), sstd=statistics.stdev(rates))


def pooled_note(pooled, grade):
    """§5.2 끝 항목: 합친 Δ ≤ 0 이면 (나) 와 같이 덧붙인다. 효과를 주장하지 않는 행(①이 아닌 행)에는 보고만이라고 밝힌다."""
    if pooled is None or pooled > 0:
        return []
    tail = "" if grade == "①" else " (효과를 주장하지 않는 행 — 보고만)"
    return [f"체크포인트 민감도: 세 체크포인트 합친 Δ {pooled:+.2f} ≤ 0 — 체크포인트 선택에 민감 — 약함{tail}"]


def interpret_HA(g, attr_ar, at_not_run, ar_not_run, at_ran, pooled):
    """§6 H-A: 위에서부터 처음 맞는 행 하나 + 덧붙임 (가)·(나)·(다). g = A-T 의 confirm 결과 dict, attr_ar = A-R↔A-T 귀속.
    *_not_run = 큐가 그 팔을 끝내 돌리지 않았다고 적음(게이트 최종 실패). 파일만 없으면 진행 중 → 판정 미완."""
    add, grade = [], g["grade"]
    if at_not_run:
        return "H-A0", add
    if not at_ran or grade.startswith("판정 미완"):  # 파일이 아직 없음 (큐 진행 중)
        return "판정 미완", add
    if grade == "②":
        row = "H-A1"
    elif grade == "①":
        if ar_not_run:
            row = "H-A4"
        elif not attr_ar:
            return "판정 미완 (A-R 귀속 비교 대기)", add
        elif attr_ar == "동등":
            row = "H-A2"
        elif attr_ar == "차이 (a > b)":  # a = A-T, b = A-R → A-T > A-R
            row = "H-A3"
        else:  # 구분 불가 또는 A-R 이 유의하게 좋음
            row = "H-A4"
        if g["d035_delta"] <= 0:  # (가)
            if row in ("H-A2", "H-A3"):
                add.append(f"(가) 시작 거리 ≥ 0.35 m 하위분석 Δ {g['d035_delta']:+.2f} ≤ 0 → {row} 에서 H-A4 로 낮춤")
                row = "H-A4"
            add.append("(가) 효과가 근접 시작 장면에 몰려 있다 — 장면 분포 혼입 가능")
        if g["e1900_delta"] <= 0:  # (나)
            add.append(f"(나) baseline e1900 대비 Δ {g['e1900_delta']:+.2f} ≤ 0 — baseline 체크포인트 선택에 민감 — 약함")
    elif grade in ("③", "④"):
        row = "H-A5" + (" (방향 일치, 효과 미확인)" if grade == "③" else " (표본 한계로 판단 보류)")
    else:
        return "판정 미완", add
    if g.get("neg_violation"):  # (다)
        add.append("(다) 음성 대조 위반 — 기전은 맞을 수 있으나 들기 전 단계에서 손해")
    return row, add + pooled_note(pooled, grade)


def interpret_HB(g, attr_b3, b2_not_run, b3_not_run, b2_ran, pooled, carry_speed, band):
    """§6 H_B. attr_b3 = B3'↔B2 귀속 (a = B2, b = B3'). *_not_run = 학습 전 조작 확인 게이트(md5) 불일치로 큐가 건너뜀"""
    add, grade = [], g["grade"]
    if b2_not_run:
        return "H_B0", add
    if not b2_ran or grade.startswith("판정 미완"):  # 파일이 아직 없음 (큐 진행 중)
        return "판정 미완", add
    if grade == "②":
        row = "H_B1"
    elif grade == "①":
        if b3_not_run:
            row = "H_B4"
        elif not attr_b3:
            return "판정 미완 (B3' 귀속 비교 대기)", add
        elif attr_b3 == "동등":
            row = "H_B2"
        elif attr_b3 == "차이 (a > b)":  # B2 > B3'
            row = "H_B3"
        else:
            row = "H_B4"
        if carry_speed:
            add.append(f"B2 carry_speed 중앙값 {carry_speed['a']:.4g} → {carry_speed['b']:.4g} (차 {carry_speed['diff']:+.4g}, "
                       f"띠 {band['carry_speed']:.4g}) — 감속 혼입")
        if g["e1900_delta"] <= 0:
            add.append(f"baseline e1900 대비 Δ {g['e1900_delta']:+.2f} ≤ 0 — baseline 체크포인트 선택에 민감 — 약함")
    elif grade in ("③", "④"):
        row = "H_B5" + (" (방향 일치, 효과 미확인)" if grade == "③" else " (표본 한계로 판단 보류)")
    else:
        return "판정 미완", add
    if g.get("neg_violation"):
        add.append("음성 대조 위반")
    return row, add + pooled_note(pooled, grade)


def interpret_joint(g_at, g_b2, b2_carry_yaw_diff, band):
    out = []
    if g_at == "①" and g_b2 == "①":
        ok = b2_carry_yaw_diff is not None and abs(b2_carry_yaw_diff) <= band["carry_yaw_net"]
        out.append(f"회전 상태는 필요조건이 아니고 명령 크기·우회가 공통 기전 (확인: B2 carry_yaw_net 차 {b2_carry_yaw_diff:+.2f}° 가 "
                   f"띠 {band['carry_yaw_net']:.2f}° 안 → {ok})")
    if g_at == "②" and g_b2 == "②":
        out.append("운반 동작 쪽 원인 둘 다 기각")
    return out


def choose_submission(grade_by_arm, main_by_arm, order=("A-T", "B2"), pending=()):
    """§7 제출 후보. (1) ① 팔 중 Δ 최대 (2) Δ ≥ +5 이고 학습 seed 101 짝 Δ > 0 인 팔 중 Δ 최대 (3) baseline. 동률은 order 순.
    pending = 결과가 아직 없는 확증 팔 (큐 진행 중) — 있으면 후보를 정하지 않는다."""
    if pending:
        return "판정 미완", f"확증 팔 {', '.join(pending)} 결과 대기"

    def d(a):
        return main_by_arm[a]["delta_pp"]

    def d101(a):
        s = main_by_arm[a]["delta_by_train_seed"]
        return s.get("101", s.get(101))

    cand = [a for a in order if grade_by_arm.get(a) == "①"]
    rule = "(1) ① 확증 팔 중 Δ 최대"
    if not cand:
        cand = [a for a in order if main_by_arm.get(a) and grade_by_arm.get(a) not in (None, "판정 미완")
                and d(a) >= 5.0 and d101(a) is not None and d101(a) > 0]
        rule = "(2) ① 팔 없음 → Δ ≥ +5%p 이고 학습 seed 101 짝 Δ > 0 인 확증 팔 중 Δ 최대"
    if not cand:
        return "baseline", "(3) 해당 팔 없음 → baseline"
    best = max(cand, key=lambda a: (d(a), -order.index(a)))
    return best, rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(A2, "results", "judge_1029"))
    ap.add_argument("--eseeds", default="42,43")
    ap.add_argument("--pattern", action="append", default=[], help="팔=경로틀 (도구 시험용으로 팔 파일을 바꾼다)")
    args = ap.parse_args()
    eseeds = tuple(int(x) for x in args.eseeds.split(","))
    pat = dict(PATTERNS)
    for kv in args.pattern:
        k, v = kv.split("=", 1)
        pat[k] = v
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(A2, "results", "noise_baseline", "null_band_1029.json")) as f:
        band = json.load(f)["band"]
    P = lambda arm, ep=2000, es=eseeds: paths(pat[arm], es, ep)
    R, V = {}, {"band": band, "eseeds": eseeds, "patterns": pat}

    # ── 확증 팔 (§5.2): 주 비교 · 음성 대조 · 하위 분석 · baseline e1900 민감도 · 평가 seed 별 ──
    for arm in ("A-T", "B2"):
        k = arm.replace("-", "").replace("'", "p")
        R[arm] = dict(
            main=run_pc(args.out, f"{k}_vs_base", P("baseline"), P(arm), "baseline", arm),
            neg=run_pc(args.out, f"{k}_vs_base_prelift", P("baseline"), P(arm), "baseline", arm, ("--bundle", "approach,grasp")),
            d035=run_pc(args.out, f"{k}_vs_base_d035", P("baseline"), P(arm), "baseline", arm, ("--min_d0", "0.35")),
            e1900=run_pc(args.out, f"{k}_vs_base_e1900", paths(BASE_E1900, (42,)), P(arm, es=(42,)), "baseline_e1900", arm),
            per_eseed={e: run_pc(args.out, f"{k}_vs_base_seed{e}", P("baseline", es=(e,)), P(arm, es=(e,)), "baseline", arm)
                       for e in eseeds} if len(eseeds) > 1 else {},
        )
    # ── 귀속 팔 (§5.3) + baseline 대비 (보고만) ──
    R["A-R"] = dict(attr=run_pc(args.out, "AR_vs_AT", P("A-T"), P("A-R"), "A-T", "A-R"),
                    base=run_pc(args.out, "AR_vs_base", P("baseline"), P("A-R"), "baseline", "A-R"))
    R["B3'"] = dict(attr=run_pc(args.out, "B3p_vs_B2", P("B2"), P("B3'"), "B2", "B3'"),
                    base=run_pc(args.out, "B3p_vs_base", P("baseline"), P("B3'"), "baseline", "B3'"))
    # ── 체크포인트 민감도 (§5.2, 보고만): 같은 체크포인트끼리, 평가 seed 42 ──
    sens = {}
    for arm in ("A-T", "B2"):
        k = arm.replace("-", "")
        sens[arm] = {}
        for ep in (1800, 1900, 2000):
            a = paths(BASE_E1900, (42,)) if ep == 1900 else P("baseline", ep, (42,))
            sens[arm][ep] = run_pc(args.out, f"{k}_vs_base_ckpt{ep}", a, P(arm, ep, (42,)), f"baseline_e{ep}", f"{arm}_e{ep}")
        got = [j for j in sens[arm].values() if j]
        pooled = (100.0 * sum(j["succ_b"] - j["succ_a"] for j in got) / sum(j["n"] for j in got)) if len(got) == 3 else None
        sens[arm]["pooled_delta"] = pooled
        sens[arm]["pooled_n"] = sum(j["n"] for j in got) if len(got) == 3 else None

    # ── 미실행 / 진행 중 구분: 큐가 '끝내 돌리지 않음' 을 적은 팔만 미실행. 파일만 없으면 진행 중 ──
    lines = status_lines()
    nr = {a: not_run(a, lines) for a in NOT_RUN_MARK}
    pending = [a for a in ("A-T", "B2") if R[a]["main"] is None and not nr[a]]
    V["not_run"], V["pending"] = nr, pending
    V["queue_done"] = os.path.exists(os.path.join(os.path.dirname(STATUS_LOG), "MAIN_DONE"))
    V["all_done"] = os.path.exists(os.path.join(os.path.dirname(STATUS_LOG), "ALL_DONE"))  # 체크포인트 민감도까지 (표기용)

    # ── Holm (3팔 고정. 돌리지 않은 팔 = C-1mode(C3 미지지, §5.4) 와 큐가 미실행으로 적은 팔 → p = 1) ──
    pv = {a: (R[a]["main"]["p_two"] if a in R and R[a]["main"] else 1.0) for a in CONFIRM}
    ph = holm(pv)
    V["holm"] = dict(raw=pv, adj=ph)

    grades = {}
    for arm in ("A-T", "B2"):
        r = R[arm]
        need = [r["main"], r["neg"], r["d035"], r["e1900"]]
        if any(x is None for x in need):
            grades[arm] = dict(grade="판정 미완", why=["필요한 비교가 아직 없다"], notes=[], neg_violation=None)
            continue
        g, why, notes, viol = confirm_grade(r["main"], r["neg"], ph[arm], arm, band)
        others = [a for a in pending if a != arm]
        if others and g != "①":  # 진행 중인 팔의 p 가 1 보다 작아지면 이 팔의 Holm p 가 내려갈 수 있다 (① 은 그대로 유지됨)
            why.append(f"(Holm 족의 {', '.join(others)} 결과 대기 — 지금 등급 {g} 은 확정 아님)")
            g = f"판정 미완 (Holm 족 대기: {', '.join(others)})"
        grades[arm] = dict(grade=g, why=why, notes=notes, neg_violation=viol,
                           d035_delta=r["d035"]["delta_pp"], d035_n=r["d035"]["n"], e1900_delta=r["e1900"]["delta_pp"],
                           old_crit=r["main"]["delta_pp"] >= OLD_CRIT)
    V["grades"] = grades
    attr = {"A-R": R["A-R"]["attr"] and attribution(R["A-R"]["attr"]),
            "B3'": R["B3'"]["attr"] and attribution(R["B3'"]["attr"])}
    V["attribution"] = attr
    V["attribution_named"] = {"A-R": attr_named(attr["A-R"], "A-T", "A-R"), "B3'": attr_named(attr["B3'"], "B2", "B3'")}

    # ── C3 (§5.4): 이미 돈 비교의 json 으로 다시 판정 ──
    c3 = None
    try:
        with open(os.path.join(A2, "results", "judge_1029", "C3_vs_placebo_self_release.json")) as f:
            sr = json.load(f)
        with open(os.path.join(A2, "results", "judge_1029", "C3_vs_placebo_hold.json")) as f:
            hd = json.load(f)
        dec = per100(sr["bundle_a"] - sr["bundle_b"], sr["n"])
        hinc = per100(hd["bundle_b"] - hd["bundle_a"], hd["n"])
        sup = dec > band["self_release"] and sr["bundle_p_one"] < 0.10 and hinc <= band["hold"] and sr["delta_pp"] >= 0
        c3 = dict(support=sup, self_release_dec=dec, p_one=sr["bundle_p_one"], hold_inc=hinc, delta=sr["delta_pp"], n=sr["n"])
    except OSError:
        pass
    V["C3"] = c3

    # ── §6 해석표 ──
    rows = {}
    row, add = interpret_HA(grades["A-T"], attr["A-R"], nr["A-T"], nr["A-R"], R["A-T"]["main"] is not None, sens["A-T"]["pooled_delta"])
    rows["H-A"] = dict(row=row, add=add, at_gate=gate_passed("AT", lines), ar_gate=gate_passed("AR", lines))
    cs = R["B2"]["main"]["mediators"]["carry_speed"] if R["B2"]["main"] else None
    row, add = interpret_HB(grades["B2"], attr["B3'"], nr["B2"], nr["B3'"], R["B2"]["main"] is not None,
                            sens["B2"]["pooled_delta"], cs, band)
    rows["H_B"] = dict(row=row, add=add)
    cy = R["B2"]["main"]["mediators"]["carry_yaw_net"]["diff"] if R["B2"]["main"] else None
    rows["joint"] = interpret_joint(grades["A-T"]["grade"], grades["B2"]["grade"], cy, band)
    rows["H_C"] = dict(row=("H_C1" if c3["support"] else "H_C2") if c3 else "H_C0")
    V["rows"] = rows

    # ── 제출 후보 (§7) ──
    best, rule = choose_submission({a: grades[a]["grade"] for a in ("A-T", "B2")},
                                   {a: R[a]["main"] for a in ("A-T", "B2")}, pending=pending)
    V["submission"] = dict(arm=best, rule=rule,
                           checkpoint_pattern=f"{best} 학습 seed 101 e2000 (logs/robomimic/.../base_<tag>_s101/<시각>/models/model_epoch_2000.pth)")
    if best in pat:
        ck, md5, nrun = submission_ckpt(pat[best])
        V["submission"].update(checkpoint=ck, checkpoint_md5=md5, checkpoint_runs=nrun)

    # ── 공식 수치 (§5.5, 평가 seed 42) ──
    off = {arm: official(pat[arm]) for arm in ("baseline", "A-T", "B2", "A-R", "B3'")}
    V["official_seed42"] = off

    # ── 요약 json + md ──
    def slim(j):
        if not j:
            return None
        keep = ("n", "succ_a", "succ_b", "delta_pp", "only_a", "only_b", "p_two", "p_one_b_gt_a", "ci95", "ci90",
                "delta_by_train_seed", "unmatched", "bundle", "bundle_a", "bundle_b", "bundle_only_a", "bundle_only_b",
                "bundle_p_one", "bundle_p_two", "pre_lift_fail_a", "pre_lift_fail_b", "mediators")
        return {k: j[k] for k in keep if k in j}

    V["comparisons"] = {arm: {k: (slim(v) if not isinstance(v, dict) or "n" in v else {e: slim(x) for e, x in v.items()})
                              for k, v in R[arm].items()} for arm in R}
    V["ckpt_sensitivity"] = {arm: {str(ep): slim(sens[arm][ep]) for ep in (1800, 1900, 2000)} |
                             {"pooled_delta": sens[arm]["pooled_delta"], "pooled_n": sens[arm]["pooled_n"]} for arm in sens}
    with open(os.path.join(args.out, "verdict_1029.json"), "w") as f:
        json.dump(V, f, ensure_ascii=False, indent=2, default=str)
    write_md(os.path.join(args.out, "verdict_1029.md"), V, R, sens, band)
    print(open(os.path.join(args.out, "verdict_1029.md")).read())


def fmt_cmp(j, la, lb):
    if not j:
        return "없음"
    s = (f"짝 {j['n']} · {la} {j['succ_a']} → {lb} {j['succ_b']} · Δ {j['delta_pp']:+.2f}%p · 불일치 {j['only_a']} : {j['only_b']} · "
         f"양측 p {j['p_two']:.4g} · 95% CI [{j['ci95'][0]:+.1f}, {j['ci95'][1]:+.1f}] · 90% CI [{j['ci90'][0]:+.1f}, {j['ci90'][1]:+.1f}] · "
         f"학습 seed 별 " + ", ".join(f"s{k} {v:+.1f}" for k, v in j["delta_by_train_seed"].items()))
    um = sum(u["a"] + u["b"] for u in j["unmatched"].values())
    return s + (f" · 짝 없음 {um}" if um else "")


def write_md(path, V, R, sens, band):
    L = ["# 사전등록 판정 (docs/prereg_1029.md §5–§7) — tools/judge_1029.py 출력", ""]
    if not V["queue_done"] or V["pending"]:
        L += [f"> **큐 미완** (MAIN_DONE {'있음' if V['queue_done'] else '없음'}, 결과 대기 확증 팔: {', '.join(V['pending']) or '없음'}) — "
              "'판정 미완' 이 붙은 항목은 판정이 아니다.", ""]
    nr = [a for a, v in V["not_run"].items() if v]
    if nr:
        L += [f"> 큐가 미실행으로 적은 팔: {', '.join(nr)}", ""]
    L.append(f"평가 seed {', '.join(map(str, V['eseeds']))} · 띠 = null_band_1029.json · Holm 3팔(A-T · B2 · C-1mode, 돌리지 않은 팔 p = 1)")
    L.append("")
    L.append("## 공식 수치 (평가 seed 42, 100회, 학습 seed 101 · 111 평균 ± 모표준편차 (표본표준편차))")
    L.append("")
    L.append("| 팔 | s101 | s111 | 평균 ± std |")
    L.append("| --- | --- | --- | --- |")
    for arm, o in V["official_seed42"].items():
        L.append(f"| {arm} | " + (f"{o['rates'][0]:.0f}% | {o['rates'][1]:.0f}% | {o['mean']:.1f} ± {o['pstd']:.1f}% ({o['sstd']:.1f})" if o else "없음 | | ") + " |")
    L.append("")
    L.append("## 확증 팔 (§5.2, baseline e2000 대비)")
    for arm in ("A-T", "B2"):
        g, r = V["grades"][arm], R[arm]
        L += ["", f"### {arm}: **{g['grade']}**", "",
              f"- 주 비교: {fmt_cmp(r['main'], 'baseline', arm)}",
              f"- Holm 보정 p: {V['holm']['adj'][arm]:.4g} (보정 전 {V['holm']['raw'][arm]:.4g})"]
        L += [f"- {w}" for w in g["why"]]
        L += [f"- 주: {n}" for n in g["notes"]]
        if r["main"]:
            m = r["main"]
            L.append(f"- 표적 묶음(운반+배치) {m['bundle_a']} → {m['bundle_b']} (A에서만 {m['bundle_only_a']} : B에서만 {m['bundle_only_b']}, 단측 p {m['bundle_p_one']:.4g})")
            L.append("- 상태 매개 지표 (A → B, 차 · 띠): " + " · ".join(
                f"{k} {v['a']:.4g} → {v['b']:.4g} ({v['diff']:+.4g}{' · 띠 ' + format(band[k], '.4g') if k in band else ''})"
                for k, v in m["mediators"].items()))
        L.append(f"- 하위 분석 시작 거리 ≥ 0.35 m: {fmt_cmp(r['d035'], 'baseline', arm)}")
        L.append(f"- baseline e1900 대비 (평가 seed 42): {fmt_cmp(r['e1900'], 'baseline_e1900', arm)}")
        for e, j in r["per_eseed"].items():
            L.append(f"- 평가 seed {e} 만: {fmt_cmp(j, 'baseline', arm)}")
        if g.get("old_crit") is not None:
            L.append(f"- 이전 기준 13%p (참고만): {'충족' if g['old_crit'] else '미충족'}")
    L += ["", "## 귀속 팔 (§5.3, 짝 팔끼리 직접)", ""]
    for arm, base in (("A-R", "A-T"), ("B3'", "B2")):
        L.append(f"- **{base} → {arm}: {V['attribution_named'][arm] or '없음'}** — {fmt_cmp(R[arm]['attr'], base, arm)}")
        L.append(f"  - baseline 대비 (보고만): {fmt_cmp(R[arm]['base'], 'baseline', arm)}")
    miss = "없음" if V.get("all_done") else "진행 중"
    L += ["", "## 체크포인트 민감도 (보고만, 평가 seed 42, 같은 체크포인트끼리)", ""]
    if not V.get("all_done"):
        L += ["> 큐의 체크포인트 민감도 단계 진행 중 (ALL_DONE 없음) — '진행 중' 칸은 결과가 아직 없다는 뜻. 끝나면 이 도구를 다시 돌린다.", ""]
    L += ["| 팔 | e1800 | e1900 | e2000 | 합친 Δ |", "| --- | --- | --- | --- | --- |"]
    for arm in sens:
        cells = []
        for ep in (1800, 1900, 2000):
            j = sens[arm][ep]
            cells.append(f"{j['succ_a']} → {j['succ_b']} ({j['delta_pp']:+.1f}, 짝 {j['n']})" if j else miss)
        pdlt = sens[arm]["pooled_delta"]
        L.append(f"| {arm} | " + " | ".join(cells) + f" | {('%+.2f (짝 %d)' % (pdlt, sens[arm]['pooled_n'])) if pdlt is not None else miss} |")
    c3 = V["C3"]
    L += ["", "## C3 ↔ 위약 (§5.4)", ""]
    if c3:
        L.append(f"- 판정 묶음 감소 {c3['self_release_dec']:+.2f}/100 (> {band['self_release']:.2f}) · 단측 p {c3['p_one']:.4g} (< 0.10) · "
                 f"HOLD 증가 {c3['hold_inc']:+.2f}/100 (≤ {band['hold']:.2f}) · Δ {c3['delta']:+.2f} (≥ 0) → **{'지지' if c3['support'] else '미지지'}**")
    L += ["", "## 결과 해석 (§6)", ""]
    for h in ("H-A", "H_B", "H_C"):
        r = V["rows"][h]
        code = r["row"].split(" ")[0]
        L.append(f"- **{h}: {r['row']}**" + (f" — {ROW_TEXT[code]}" if code in ROW_TEXT else ""))
        for a in r.get("add", []):
            L.append(f"  - {a}")
        if h in ("H-A", "H_B") and not V.get("all_done") and sens[{"H-A": "A-T", "H_B": "B2"}[h]]["pooled_delta"] is None:
            L.append("  - (체크포인트 민감도 합친 Δ 의 덧붙임은 ALL_DONE 뒤 다시 실행해 정한다 — 행은 바뀌지 않는다)")
    for j in V["rows"]["joint"]:
        L.append(f"- H-A · H_B 함께: {j}")
    if not V["rows"]["joint"]:
        L.append(f"- H-A · H_B 함께: 해당 없음 (A-T {V['grades']['A-T']['grade']} · B2 {V['grades']['B2']['grade']} — "
                 "함께 읽기는 둘 다 ① 이거나 둘 다 ② 일 때만)")
    s = V["submission"]
    L += ["", "## 제출 후보 (§7)", "", f"- **{s['arm']}** — {s['rule']}"]
    if s.get("checkpoint"):
        L.append(f"- 체크포인트: `{s['checkpoint']}` (md5 {s['checkpoint_md5']}" +
                 (f"; run 폴더 {s['checkpoint_runs']}개 — 마지막 것" if s.get("checkpoint_runs", 1) > 1 else "") + ")")
    else:
        L.append(f"- 체크포인트: {s['checkpoint_pattern']}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
