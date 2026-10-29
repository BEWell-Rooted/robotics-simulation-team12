"""tools/judge_1029.py 판정 분기 단위 시험 (합성 값). 실제 처치 팔 데이터는 읽지 않는다.

  python -I tools/test_judge_1029.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_1029 as J  # noqa: E402

band = json.load(open(os.path.join(HERE, "..", "results", "noise_baseline", "null_band_1029.json")))["band"]

def M(d=10.0, p=0.001, up95=15.0, seeds=(8.0, 12.0), bp1=0.01, box=-30, low=0, n=400, pre=(19, 19)):
    return dict(n=n, delta_pp=d, p_two=p, ci95=[d - 6, up95], ci90=[d - 5, d + 5], delta_by_train_seed={"101": seeds[0], "111": seeds[1]},
                bundle_p_one=bp1, pre_lift_fail_a=pre[0], pre_lift_fail_b=pre[1],
                mediators=dict(box_push=dict(diff=box), low_approach=dict(diff=low)))
NEG0 = dict(n=400, bundle_a=38, bundle_b=38, bundle_p_two=1.0)
ok = 0
def chk(name, got, exp):
    global ok
    assert got == exp, f"{name}: {got!r} != {exp!r}"; ok += 1

# Holm
h = J.holm({"A-T": 0.01, "B2": 0.04, "C-1mode": 1.0}); chk("holm1", (round(h["A-T"],4), round(h["B2"],4), h["C-1mode"]), (0.03, 0.08, 1.0))
h = J.holm({"A-T": 0.04, "B2": 0.01, "C-1mode": 1.0}); chk("holm2", (round(h["A-T"],4), round(h["B2"],4)), (0.08, 0.03))
h = J.holm({"A-T": 0.03, "B2": 0.02, "C-1mode": 1.0}); chk("holm-mono", (round(h["A-T"],4), round(h["B2"],4)), (0.06, 0.06))
# ①
g = J.confirm_grade(M(), NEG0, 0.003, "A-T", band); chk("one", g[0], "①")
g = J.confirm_grade(M(d=6.09), NEG0, 0.003, "A-T", band); chk("one-floor-strict", g[0] == "①", False)
g = J.confirm_grade(M(d=6.2), NEG0, 0.003, "A-T", band); chk("one-floor-pass", g[0], "①")
g = J.confirm_grade(M(seeds=(15.0, 0.0)), NEG0, 0.003, "A-T", band); chk("one-seed0", g[0] == "①", False)
g = J.confirm_grade(M(), NEG0, 0.05, "A-T", band); chk("one-holm-strict", g[0] == "①", False)
# ②
g = J.confirm_grade(M(d=0.0, p=0.9, up95=3.9, seeds=(0, 0)), NEG0, 1.0, "A-T", band); chk("two-up95", g[0], "②")
g = J.confirm_grade(M(d=0.0, p=0.9, up95=4.0, seeds=(0, 0)), NEG0, 1.0, "A-T", band); chk("two-up95-strict", g[0], "④")
g = J.confirm_grade(M(d=0.0, p=0.9, up95=3.9, seeds=(0, 0)), NEG0, 1.0, "C-1mode", band); chk("two-c1mode-noclause", g[0], "④")
g = J.confirm_grade(M(d=-5.0, p=0.04, up95=1.0, seeds=(-5, -5)), NEG0, 1.0, "C-1mode", band); chk("two-neg", g[0], "②")
# ③ (Δ ≥ 5, bundle p < .10, box_push < -6/100 → n=400: diff < -24, low ≤ 7.11/100 → ≤ 28.4)
base3 = dict(d=5.5, p=0.06, up95=11.0, seeds=(5.0, 6.0))
g = J.confirm_grade(M(**base3, bp1=0.05, box=-25, low=28), NEG0, 0.12, "A-T", band); chk("three", g[0], "③")
g = J.confirm_grade(M(**base3, bp1=0.05, box=-24, low=28), NEG0, 0.12, "A-T", band); chk("three-box-strict", g[0], "④")
g = J.confirm_grade(M(**base3, bp1=0.05, box=-25, low=29), NEG0, 0.12, "A-T", band); chk("three-low", g[0], "④")
g = J.confirm_grade(M(**base3, bp1=0.10, box=-25, low=0), NEG0, 0.12, "A-T", band); chk("three-bp-strict", g[0], "④")
# 주 · 음성 대조 (n=400: 3.03/100 → 12.12건 → 13건 이상 증가가 '띠보다 큼')
g = J.confirm_grade(M(low=15), dict(n=400, bundle_a=38, bundle_b=51, bundle_p_two=0.04), 0.003, "A-T", band)
chk("neg-viol", g[3], True); chk("note-low", any("low_approach" in s for s in g[2]), True)
g = J.confirm_grade(M(), dict(n=400, bundle_a=38, bundle_b=50, bundle_p_two=0.04), 0.003, "A-T", band); chk("neg-12-no", g[3], False)
g = J.confirm_grade(M(), dict(n=400, bundle_a=38, bundle_b=60, bundle_p_two=0.06), 0.003, "A-T", band); chk("neg-p", g[3], False)
g = J.confirm_grade(M(pre=(38, 45)), NEG0, 0.003, "A-T", band); chk("note-prelift", any("들기 전" in s for s in g[2]), True)
# 귀속
chk("eq", J.attribution(dict(ci90=[-6.9, 6.9], p_two=0.5, delta_pp=0)), "동등")
chk("eq-strict", J.attribution(dict(ci90=[-7.0, 3.0], p_two=0.5, delta_pp=-2)), "구분 불가")
chk("diff-a", J.attribution(dict(ci90=[-15.0, -8.0], p_two=0.01, delta_pp=-11)), "차이 (a > b)")
chk("diff-b", J.attribution(dict(ci90=[8.0, 15.0], p_two=0.01, delta_pp=11)), "차이 (b > a)")
chk("eq-before-diff", J.attribution(dict(ci90=[1.0, 6.0], p_two=0.03, delta_pp=3.5)), "동등")
# §6 H-A  interpret_HA(g, attr_ar, at_not_run, ar_not_run, at_ran, pooled)
G = lambda grade, d035=3.0, e1900=2.0, neg=False: dict(grade=grade, d035_delta=d035, e1900_delta=e1900, neg_violation=neg)
chk("HA0", J.interpret_HA(G("①"), "동등", True, False, True, 5)[0], "H-A0")
chk("HA-pending", J.interpret_HA(G("판정 미완"), None, False, False, False, None)[0], "판정 미완")
chk("HA-pending-holm", J.interpret_HA(G("판정 미완 (Holm 족 대기: B2)"), "동등", False, False, True, 5)[0], "판정 미완")
chk("HA1", J.interpret_HA(G("②"), "동등", False, False, True, 5)[0], "H-A1")
chk("HA2", J.interpret_HA(G("①"), "동등", False, False, True, 5)[0], "H-A2")
chk("HA3", J.interpret_HA(G("①"), "차이 (a > b)", False, False, True, 5)[0], "H-A3")
chk("HA4-ARbetter", J.interpret_HA(G("①"), "차이 (b > a)", False, False, True, 5)[0], "H-A4")
chk("HA4-indist", J.interpret_HA(G("①"), "구분 불가", False, False, True, 5)[0], "H-A4")
chk("HA4-ARnotrun", J.interpret_HA(G("①"), None, False, True, True, 5)[0], "H-A4")
chk("HA-ARpending", J.interpret_HA(G("①"), None, False, False, True, 5)[0], "판정 미완 (A-R 귀속 비교 대기)")
r = J.interpret_HA(G("①", d035=-1.0), "동등", False, False, True, 5); chk("HA-ga", (r[0], len(r[1])), ("H-A4", 2))
r = J.interpret_HA(G("①", d035=0.0), "구분 불가", False, False, True, 5); chk("HA-ga-4", (r[0], len(r[1])), ("H-A4", 1))
r = J.interpret_HA(G("①", e1900=0.0), "동등", False, False, True, 5); chk("HA-na", (r[0], any("(나)" in s for s in r[1])), ("H-A2", True))
r = J.interpret_HA(G("②", neg=True), None, False, False, True, -1); chk("HA-da-pool", (r[0], len(r[1]), "보고만" in r[1][1]), ("H-A1", 2, True))
r = J.interpret_HA(G("①"), "동등", False, False, True, 0.0); chk("HA-pool-plus", (r[0], r[1][-1].endswith("약함")), ("H-A2", True))
chk("HA5-3", J.interpret_HA(G("③"), "동등", False, False, True, 5)[0], "H-A5 (방향 일치, 효과 미확인)")
chk("HA5-4", J.interpret_HA(G("④"), "동등", False, False, True, 5)[0], "H-A5 (표본 한계로 판단 보류)")
# §6 H_B  interpret_HB(g, attr_b3, b2_not_run, b3_not_run, b2_ran, pooled, cs, band)
cs = dict(a=0.0106, b=0.0098, diff=-0.0008)
chk("HB0", J.interpret_HB(G("판정 미완"), None, True, False, False, None, None, band)[0], "H_B0")
chk("HB1", J.interpret_HB(G("②"), "동등", False, False, True, 5, cs, band)[0], "H_B1")
chk("HB2", J.interpret_HB(G("①"), "동등", False, False, True, 5, cs, band)[0], "H_B2")
chk("HB3", J.interpret_HB(G("①"), "차이 (a > b)", False, False, True, 5, cs, band)[0], "H_B3")
chk("HB4", J.interpret_HB(G("①"), "차이 (b > a)", False, False, True, 5, cs, band)[0], "H_B4")
chk("HB4-notrun", J.interpret_HB(G("①"), None, False, True, True, 5, cs, band)[0], "H_B4")
chk("HB-B3pending", J.interpret_HB(G("①"), None, False, False, True, 5, cs, band)[0], "판정 미완 (B3' 귀속 비교 대기)")
chk("HB5", J.interpret_HB(G("④"), "동등", False, False, True, 5, cs, band)[0], "H_B5 (표본 한계로 판단 보류)")
chk("HB-speed", any("carry_speed" in s for s in J.interpret_HB(G("①"), "동등", False, False, True, 5, cs, band)[1]), True)
# 미실행 표시
L = ["[x] A-T 학습 건너뜀 (데이터 게이트 미통과)", "[x] [보정] AT 데이터 게이트 통과", "[x] md5 불일치 ./datasets/team12_kgh/plan_B/kgh_mimic_500_dirorig_magb2.hdf5: aaaa (기대 a025e391…)"]
chk("nr-AT", J.not_run("A-T", L), False); chk("nr-B3p", J.not_run("B3'", L), True); chk("nr-B2", J.not_run("B2", L), False)
chk("nr-AR", J.not_run("A-R", L + ["[x] [보정] A-R 보정 실패 — A-R 은 게이트 미통과로 보고"]), True)
chk("gate-AT", J.gate_passed("AT", L), True)
# joint
chk("joint++", len(J.interpret_joint("①", "①", 3.0, band)), 1)
chk("joint--", J.interpret_joint("②", "②", None, band), ["운반 동작 쪽 원인 둘 다 기각"])
chk("joint+-", J.interpret_joint("①", "④", 3.0, band), [])
# 제출
mm = lambda d, d101: dict(delta_pp=d, delta_by_train_seed={"101": d101, "111": 1.0})
chk("sub1", J.choose_submission({"A-T": "①", "B2": "①"}, {"A-T": mm(10, 5), "B2": mm(12, 5)})[0], "B2")
chk("sub1-tie", J.choose_submission({"A-T": "①", "B2": "①"}, {"A-T": mm(10, 5), "B2": mm(10, 5)})[0], "A-T")
chk("sub1-only", J.choose_submission({"A-T": "④", "B2": "①"}, {"A-T": mm(20, 5), "B2": mm(8, 5)})[0], "B2")
chk("sub2", J.choose_submission({"A-T": "③", "B2": "④"}, {"A-T": mm(5.5, 2), "B2": mm(7, -1)})[0], "A-T")
chk("sub2-none", J.choose_submission({"A-T": "④", "B2": "④"}, {"A-T": mm(4.9, 2), "B2": mm(7, 0.0)})[0], "baseline")
chk("sub3", J.choose_submission({"A-T": "②", "B2": "②"}, {"A-T": mm(-3, -2), "B2": mm(0, 0)})[0], "baseline")
chk("sub-pending", J.choose_submission({"A-T": "①", "B2": "판정 미완"}, {"A-T": mm(10, 5), "B2": None}, pending=["B2"])[0], "판정 미완")
print("모든 시험 통과:", ok)
