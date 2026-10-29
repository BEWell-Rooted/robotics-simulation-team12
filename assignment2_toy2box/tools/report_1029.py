"""사전등록 docs/prereg_1029.md §3 · §5.5 · §7 의 보고 항목 중 judge_1029.py 출력(verdict_1029.md)에 없는 것을 모은다.
판정(①–④ · 귀속 · §6 행 · 제출 후보)은 바꾸지 않고 읽지도 않는다 — 보고만.

  python -I tools/report_1029.py        # → results/judge_1029/report_1029.md (+ 귀속 비교의 평가 seed 별 txt·json)

담는 것: 평가 seed 별 원시 성공 수 · 귀속 비교의 평가 seed 별 Δ 와 상태 매개 지표 · 팔별 실패 원인 귀속 (ii)·(i)·(iii) ·
A-T 실패 이동 예측 대조(§3) · 층화 전 분포와 원본 선택 히스토그램(§3) · 데이터 md5 · 제출 확인(§7, play.py) 결과.
"""
import collections
import csv
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_1029 as J  # noqa: E402  (run_pc · paths · PATTERNS · fmt_cmp 를 그대로 쓴다)

A2 = J.A2
OUT = os.path.join(A2, "results", "judge_1029")
RUNS = os.path.expanduser("~/IsaacLab_RS/logs/rs_a2_runs")
DS = os.path.expanduser("~/IsaacLab_RS/datasets/team12_kgh")
ARMS = ("baseline", "A-T", "B2", "A-R", "B3'")
ATTR_KEYS = ("collision", "self_release", "slip", "other")
ATTR_KO = {"collision": "충돌", "self_release": "스스로 놓음", "slip": "미끄러짐", "other": "기타"}
# §3 A-T 의 실패 이동 예측 (baseline → A-T, 귀속 (ii), 200 rollout 당)
AT_PRED = {"collision": ("≤ 12", lambda x: x <= 12), "self_release": ("≤ 16 (약한 예측)", lambda x: x <= 16),
           "slip": ("8 ± 3 (불변)", lambda x: 5 <= x <= 11), "other": ("3 ± 3", lambda x: 0 <= x <= 6)}
MD5 = {  # 사전등록에 적은 앞자리
    "toy2box_kgh_annotated.hdf5": "8ddcba04", "plan_B/kgh_mimic_500_reach4.hdf5": "32f3c488",
    "plan_B/kgh_mimic_500_dirorig_magb2.hdf5": "a025e391", "kgh_mimic_500.hdf5": None,
    "plan_A/translate_strat500.hdf5": None, "plan_A/rotworld_strat500.hdf5": None,
}


def csvs(arm, ep=2000, es=(42, 43)):
    return [os.path.join(A2, p) for p in J.paths(J.PATTERNS[arm], es, ep)]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def succ(path):
    return sum(int(r["success"]) for r in read(path)) if os.path.exists(path) else None


def attr_counts(arm, es):
    """팔·평가 seed 묶음의 귀속 (ii)·(i)·(iii) 건수 (짝짓지 않은 원시 rollout 전체) + 들기 전 실패 + 시간 초과"""
    out = {k: collections.Counter() for k in ("attr_ii", "attr_i", "attr_iii")}
    n = pre = to = 0
    for p in csvs(arm, es=es):
        cp = p[:-4] + ".carry.csv"
        if not (os.path.exists(p) and os.path.exists(cp)):
            return None
        cls = {r["id"]: r for r in read(p)}
        for r in read(cp):
            n += 1
            for k in out:
                if r.get(k) in ATTR_KEYS:
                    out[k][r[k]] += 1
            pre += cls[r["id"]]["category"] in ("approach", "grasp")
            to += r.get("timeout") in ("1", 1)
    return dict(n=n, pre_lift=pre, timeout=to, **{k: dict(v) for k, v in out.items()})


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def src_hist(tag, n_shards):
    c = collections.Counter()
    for s in range(1, n_shards + 1):
        p = os.path.join(RUNS, "plan_A", f"{tag}_s{s}_src.jsonl")
        if not os.path.exists(p):
            return None, s - 1
        with open(p) as f:
            for line in f:
                c[json.loads(line)["src"]] += 1
    return c, n_shards


def main():
    L = ["# 사전등록 보고 보충 (docs/prereg_1029.md §3 · §5.5 · §7) — tools/report_1029.py 출력", "",
         "판정은 `verdict_1029.md` 에 있다. 이 문서는 판정에 쓰지 않는 보고 항목만 모은다.", ""]

    # 1. 원시 성공 수 (평가 seed 별)
    L += ["## 1. 원시 성공 수 (e2000, 100회 중, 학습 seed s101 · s111)", "",
          "| 팔 | 평가 seed 42 (공식) | 평가 seed 43 | 42 + 43 합 (/400) |", "| --- | --- | --- | --- |"]
    for arm in ARMS:
        row, tot = [], 0
        for e in (42, 43):
            v = [succ(p) for p in csvs(arm, es=(e,))]
            row.append(" · ".join("—" if x is None else str(x) for x in v))
            tot += sum(x for x in v if x is not None)
        L.append(f"| {arm} | {row[0]} | {row[1]} | {tot} |")
    extra = [("baseline e1900 (P0)", [os.path.join(A2, J.BASE_E1900.format(s=s, e=42)) for s in J.TRAIN_SEEDS]),
             ("A-T e1800", csvs("A-T", 1800, (42,))), ("A-T e1900", csvs("A-T", 1900, (42,))),
             ("baseline e1800", csvs("baseline", 1800, (42,))), ("B2 e1800", csvs("B2", 1800, (42,))),
             ("B2 e1900", csvs("B2", 1900, (42,)))]
    L += ["", "체크포인트 민감도용 (평가 seed 42): " + " · ".join(
        f"{name} " + " · ".join("진행 중" if succ(p) is None else str(succ(p)) for p in ps) for name, ps in extra), ""]

    # 2. 귀속 비교의 평가 seed 별 값과 상태 매개 지표
    L += ["## 2. 귀속 비교 — 평가 seed 별 Δ 와 상태 매개 지표 (§5.5)", ""]
    for arm, base, k in (("A-R", "A-T", "AR_vs_AT"), ("B3'", "B2", "B3p_vs_B2")):
        for e in (42, 43):
            j = J.run_pc(OUT, f"{k}_seed{e}", J.paths(J.PATTERNS[base], (e,)), J.paths(J.PATTERNS[arm], (e,)), base, arm)
            L.append(f"- {base} → {arm}, 평가 seed {e} 만: {J.fmt_cmp(j, base, arm)}")
        with open(os.path.join(OUT, f"{k}.json")) as f:
            m = json.load(f)["mediators"]
        L.append(f"- {base} → {arm} 상태 매개 지표 (42 + 43, A → B 차): " +
                 " · ".join(f"{kk} {v['a']:.4g} → {v['b']:.4g} ({v['diff']:+.4g})" for kk, v in m.items()))
        L.append("")

    # 3. 실패 원인 귀속 (ii)·(i)·(iii)
    L += ["## 3. 들기 이후 실패의 원인 귀속 (ii) · (i) · (iii) — 팔별 원시 rollout (짝짓지 않음)", "",
          "(ii) 가 판정·예측에 쓰는 정의, (i) · (iii) 은 민감도 (`carry_metrics.py`). 칸 = 충돌 · 스스로 놓음 · 미끄러짐 · 기타.", "",
          "| 팔 | 평가 seed | n | (ii) | (i) | (iii) | 들기 전 실패 | 시간 초과 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    AC = {}
    for arm in ARMS:
        for es, lab in (((42,), "42"), ((43,), "43"), ((42, 43), "42+43")):
            c = attr_counts(arm, es)
            AC[(arm, lab)] = c
            if not c:
                L.append(f"| {arm} | {lab} | 없음 | | | | | |")
                continue
            cell = lambda d: " · ".join(str(d.get(x, 0)) for x in ATTR_KEYS)
            L.append(f"| {arm} | {lab} | {c['n']} | {cell(c['attr_ii'])} | {cell(c['attr_i'])} | {cell(c['attr_iii'])} | "
                     f"{c['pre_lift']} | {c['timeout']} |")
    L.append("")

    # 4. A-T 실패 이동 예측 대조 (§3)
    L += ["## 4. A-T 의 실패 이동 예측 대조 (§3, 귀속 (ii), 200 rollout 당 — 보고만)", "",
          "예측은 평가 seed 42 (200 rollout) 기준으로 등록했다. 42 + 43 은 400 rollout 을 200 당으로 나눈 값.", "",
          "| 원인 | 등록 예측 (A-T) | baseline seed 42 | A-T seed 42 | 맞음 | baseline 42+43 (/200) | A-T 42+43 (/200) | 맞음 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in ATTR_KEYS:
        b42, a42 = AC[("baseline", "42")], AC[("A-T", "42")]
        bb, ab = AC[("baseline", "42+43")], AC[("A-T", "42+43")]
        if not all((b42, a42, bb, ab)):
            continue
        x42 = a42["attr_ii"].get(k, 0) * 200 / a42["n"]
        xab = ab["attr_ii"].get(k, 0) * 200 / ab["n"]
        L.append(f"| {ATTR_KO[k]} | {AT_PRED[k][0]} | {b42['attr_ii'].get(k, 0) * 200 / b42['n']:.0f} | {x42:.0f} | "
                 f"{'예' if AT_PRED[k][1](x42) else '아니오'} | {bb['attr_ii'].get(k, 0) * 200 / bb['n']:.1f} | {xab:.1f} | "
                 f"{'예' if AT_PRED[k][1](xab) else '아니오'} |")
    L += ["", "들기 전 실패는 음성 대조(§5.2, `verdict_1029.md`)로 본다.", ""]

    # 5. 층화 전 분포 · 원본 선택 히스토그램 (§3)
    L += ["## 5. 층화 전 분포와 원본 선택 호출 히스토그램 (§3)", ""]
    for tag, name, n in (("AT", "A-T", 6), ("AR", "A-R", 5)):
        p = os.path.join(A2, "results", "plan_A", f"stratify_{tag}500.txt")
        L.append(f"**{name}** — 층화 (`results/plan_A/stratify_{tag}500.txt`, '있음' 이 층화 전 후보 수):")
        L.append("")
        if os.path.exists(p):
            with open(p) as f:
                L += [ln.rstrip("\n") for ln in f if ln.startswith(("|", "baseline", "층화"))]
        c, got = src_hist(tag, n)
        L.append("")
        if c:
            tot = sum(c.values())
            L.append(f"원본 선택 호출 (`--src_log`, 샤드 {got}개, 시도 기준 {tot}회, 사람 시범 번호: 횟수): " +
                     " · ".join(f"{i}: {c.get(i, 0)}" for i in range(20)) +
                     f" — 최소 {min(c.get(i, 0) for i in range(20))} · 최대 {max(c.values())} (균등이면 {tot / 20:.0f})")
        L.append("")

    # 5b. 학습 데이터의 접근 높이 (A-R ↔ A-T 귀속 해석용, 보고만)
    dm = os.path.join(A2, "results", "plan_A", "data_metrics_1030")
    L += ["**학습 데이터 500 의 접근 높이** (`carry_metrics.py <데이터>.hdf5 --csv`, `results/plan_A/data_metrics_1030/`; "
          "p10 은 게이트 도구와 같은 numpy 선형 보간):", "",
          "| 데이터 | approach_z 중앙값 (m) | p10 (m) | < 0.15 m 시범 수 |", "| --- | --- | --- | --- |"]
    for name, lab in (("kgh_mimic_500", "baseline"), ("translate_strat500", "A-T"), ("rotworld_strat500", "A-R")):
        p = os.path.join(dm, f"{name}.csv")
        if not os.path.exists(p):
            continue
        v = sorted(float(r["approach_z"]) for r in read(p) if r["approach_z"] not in ("", "nan", "-"))
        k = 0.1 * (len(v) - 1)
        p10 = v[int(k)] + (v[min(int(k) + 1, len(v) - 1)] - v[int(k)]) * (k - int(k))
        med = (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2
        L.append(f"| {lab} (`{name}`) | {med:.3f} | {p10:.3f} | {sum(x < 0.15 for x in v)} / {len(v)} |")
    L.append("")

    # 6. 데이터 md5 · 평가 로그 위치
    L += ["## 6. 데이터 md5 · 평가 로그 위치", ""]
    for rel, want in MD5.items():
        p = os.path.join(DS, rel)
        if os.path.exists(p):
            h = md5(p)
            ok = "" if want is None else (" — 사전등록 값과 같음" if h.startswith(want) else f" — **사전등록 {want}… 와 다름**")
            L.append(f"- `datasets/team12_kgh/{rel}`: {h}{ok}")
    L += ["- 큐의 md5 검사(`run_queue_1029.sh md5_ok`)는 불일치만 status.log 에 적는다. 위 값이 일치의 근거다.",
          "- 평가 로그: `~/IsaacLab_RS/logs/rs_a2_runs/b3_eval_base_<tag>_s<s>_e<ep>_seed<e>.log` (요약 사본 `docs/run_logs/`). "
          "baseline 평가 seed 42 e2000 은 이전 판 `run_baseline.sh` 라 이름에 `_seed42` 가 없다(`b3_eval_base_kgh500_s{101,111}_e2000.log`). "
          "P0 e1900 은 `run_baseline.sh` 밖에서 돌려 `b4_null_a6000_s{101,111}_e1900.log` (rc 줄 없음, 기록 hdf5 의 attrs 로 확인).", ""]

    # 7. 제출 확인 (§7)
    L += ["## 7. 제출 확인 (§7, 수정하지 않은 play.py, 평가 seed 42, 100회, horizon 800)", ""]
    for p in sorted(glob.glob(os.path.join(RUNS, "submit_check_*.log"))):
        with open(p, errors="replace") as f:
            txt = f.read()
        m = re.findall(r"Successful trials: (\d+), out of (\d+)", txt)
        rc = re.findall(r"^rc=(\d+)", txt, re.M)
        ck = re.findall(r"ckpt (\S+) md5 (\w+)", txt)
        name = os.path.basename(p)[len("submit_check_"):-4]
        mm = re.match(r"(.+)_s(\d+)_e(\d+)_seed(\d+)", name)
        rec = None
        if mm:
            arm = {"AT": "A-T", "B2": "B2", "baseline": "baseline"}.get(mm.group(1), mm.group(1))
            if arm in J.PATTERNS:
                rec = succ(os.path.join(A2, J.PATTERNS[arm].format(s=mm.group(2), ep=mm.group(3), e=mm.group(4))))
        got = int(m[-1][0]) if m else None
        verdict = "진행 중" if not rc else ("일치" if got is not None and got == rec else f"**불일치 — 제출을 멈추고 원인을 찾는다**")
        L.append(f"- `{name}`: play.py {got if got is not None else '—'} / 100 · 기록(play_record.py) {rec} → {verdict}"
                 + (f" · rc {rc[-1]}" if rc else "") + (f" · ckpt md5 {ck[0][1]}" if ck else ""))
    L += ["- 이 확인과 평가 seed 43 은 모두 선택에 쓴 표본 안이라 '재현' 이 아니다. 조교의 미공개 seed 결과는 기대값 ± 약 5%p 흔들릴 수 있다.", ""]

    with open(os.path.join(OUT, "report_1029.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
