"""트랙 C 진단 실험 C3 와 위약: 같은 BC-RNN-GMM 체크포인트에서 '그리퍼 결정 규칙' 하나만 바꿔 평가한다 (play_record.py 위에 얹는 래퍼).

robomimic GMM 정책은 매 step 혼합 성분 하나를 뽑아(low_noise_eval: std 1e-4) 7차원 행동 전체를 쓴다.
이 래퍼는 샘플은 원래대로 한 번 뽑고(기본 난수 소비 동일), 그리퍼 차원만 규칙으로 덮어쓴다.

  --grip_rule sample    덮어쓰지 않음 (= baseline, 수동성 확인용)
  --grip_rule majority  C3: P(open) = Σ_k w_k · 1[μ_k,grip ≥ 0] ≥ 0.5 이면 +1(열기), 아니면 −1
  --grip_rule mean      Σ_k w_k μ_k,grip ≥ 0 이면 +1, 아니면 −1
  --grip_rule resample  위약: 별도 torch.Generator(--resample_seed, 기본 4242)로 그리퍼 부호를 Bernoulli(P(open))에서
                        다시 뽑는다. 주변 분포는 baseline 과 같고 난수만 새로 — "경계 상태를 다시 뽑는 효과"만 남긴다.

팔 6차원, 정책 가중치, RNN 초기화 주기, 평가 seed 는 그대로다. 조교 play.py 에는 이 규칙이 없으므로
제출 모델의 개선이 아니라 기전(소수 성분 샘플이 원인인가)을 가르는 진단 실험이다. C3 는 같은 기계에서 위약과 짝지어 판정한다.

바뀐 step 은 rollout · step · 단계 · P(open) · 원래 부호 · 새 부호를 CSV 로 남긴다(--change_log, 기본 <record_file>.gripchanges.csv).
단계는 그 step 의 관측으로 정한다: 트럭이 처음 높이 + 3 cm 위이고 손끝–트럭 < 8 cm 면 '들고 있음' →
트럭–상자 xy < 0.11 m 면 place, 아니면 transport. 들고 있지 않을 때: 한 번이라도 들었으면 released, 손끝–트럭 < 5 cm 면 grasp, 아니면 approach.

  ./isaaclab.sh -p <A2>/tools/plan_C/play_gripmix.py --grip_rule majority \
      --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint <pth> \
      --num_rollouts 100 --horizon 800 --seed 42 --headless --record_file <out.hdf5>
"""
import atexit
import csv
import os
import runpy
import sys
from collections import Counter

argv = sys.argv[1:]
rule = "sample"
resample_seed = 4242
change_log = None
out = []
i = 0
while i < len(argv):
    if argv[i] == "--grip_rule":
        rule = argv[i + 1]
        i += 2
    elif argv[i] == "--resample_seed":
        resample_seed = int(argv[i + 1])
        i += 2
    elif argv[i] == "--change_log":
        change_log = argv[i + 1]
        i += 2
    else:
        out.append(argv[i])
        i += 1
assert rule in ("sample", "majority", "mean", "resample"), rule
if change_log is None and "--record_file" in out:
    change_log = out[out.index("--record_file") + 1] + ".gripchanges.csv"

import gymnasium as gym  # noqa: E402

_prev_make = gym.make
_stat = {"steps": 0, "changed": 0, "to_open": 0, "to_close": 0, "patched": False, "reported": False}
_ep = {"k": -1, "t": 0, "z0": None, "lifted": False}
_rows = []
_phase_changed = Counter()
_phase_steps = Counter()


def _phase(obs):
    eef = obs["eef_pos"][0, -1]
    tr = obs["toy_truck_positions"][0, -1]
    bx = obs["box_positions"][0, -1]
    if _ep["z0"] is None:
        _ep["z0"] = float(tr[2])
    d3 = float((eef - tr).norm())
    held = float(tr[2]) > _ep["z0"] + 0.03 and d3 < 0.08
    if held:
        _ep["lifted"] = True
        return "place" if float((tr[:2] - bx[:2]).norm()) < 0.11 else "transport"
    if _ep["lifted"]:
        return "released"
    return "grasp" if d3 < 0.05 else "approach"


def _patch():
    if _stat["patched"]:
        return
    import torch
    from robomimic.algo.algo import RolloutPolicy
    from robomimic.models.policy_nets import RNNGMMActorNetwork

    orig_start = RolloutPolicy.start_episode

    def start_episode(self):
        _ep.update(k=_ep["k"] + 1, t=0, z0=None, lifted=False)
        return orig_start(self)

    RolloutPolicy.start_episode = start_episode
    orig_forward = RNNGMMActorNetwork.forward
    gen = {}

    def forward(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False):
        if self.training:
            return orig_forward(self, obs_dict, goal_dict=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)
        if rule == "sample":
            res = orig_forward(self, obs_dict, goal_dict=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)
            _stat["steps"] += 1
            _ep["t"] += 1
            return res
        o = self.forward_train(obs_dict=obs_dict, goal_dict=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)
        ad, state = (o if return_state else (o, None))
        a = ad.sample()  # 원래와 같은 기본 난수 소비 (팔 6차원은 이 샘플 그대로)
        with torch.no_grad():
            w = torch.softmax(ad.mixture_distribution.logits, dim=-1)       # (B, T, K)
            mu = ad.component_distribution.base_dist.loc[..., 6]             # (B, T, K)
            popen = (w * (mu >= 0).float()).sum(-1)                          # (B, T)
            if rule == "majority":
                new_open = popen >= 0.5
            elif rule == "mean":
                new_open = (w * mu).sum(-1) >= 0
            else:  # resample — 별도 Generator, 기본 난수 흐름은 건드리지 않음
                dev = popen.device
                if dev not in gen:
                    gen[dev] = torch.Generator(device=dev)
                    gen[dev].manual_seed(resample_seed)
                u = torch.rand(popen.shape, generator=gen[dev], device=dev)
                new_open = u < popen
            old_open = a[..., 6] >= 0
            ph = _phase(obs_dict)
            _phase_steps[ph] += 1
            _stat["steps"] += int(new_open.numel())
            ch = old_open != new_open
            n_ch = int(ch.sum())
            if n_ch:
                _stat["changed"] += n_ch
                _stat["to_open"] += int((ch & new_open).sum())
                _stat["to_close"] += int((ch & ~new_open).sum())
                _phase_changed[ph] += n_ch
                _rows.append(dict(rollout=_ep["k"], step=_ep["t"], phase=ph, popen=round(float(popen.flatten()[-1]), 4),
                                  old_open=int(bool(old_open.flatten()[-1])), new_open=int(bool(new_open.flatten()[-1]))))
            a = a.clone()
            a[..., 6] = torch.where(new_open, torch.ones_like(popen), -torch.ones_like(popen))
        _ep["t"] += 1
        return (a, state) if return_state else a

    RNNGMMActorNetwork.forward = forward
    _stat["patched"] = True
    print(f"[play_gripmix] grip_rule={rule} resample_seed={resample_seed} change_log={change_log}", flush=True)


def _report():
    if _stat["reported"]:
        return
    _stat["reported"] = True
    print(f"[play_gripmix] rule={rule} steps={_stat['steps']} gripper sign changed={_stat['changed']} "
          f"(→open {_stat['to_open']}, →close {_stat['to_close']})", flush=True)
    if rule != "sample":
        print("[play_gripmix] changed/steps by phase: " + ", ".join(
            f"{p} {_phase_changed[p]}/{_phase_steps[p]}" for p in ("approach", "grasp", "transport", "place", "released")), flush=True)
    if change_log and rule != "sample":
        os.makedirs(os.path.dirname(os.path.abspath(change_log)), exist_ok=True)
        with open(change_log, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["rollout", "step", "phase", "popen", "old_open", "new_open"])
            wr.writeheader()
            wr.writerows(_rows)
        print(f"[play_gripmix] wrote {len(_rows)} changed steps -> {change_log}", flush=True)


def _make(*a, **kw):
    _patch()
    env = _prev_make(*a, **kw)
    base = env.unwrapped
    oc = base.close

    def close(*ca, **ck):  # simulation_app.close() 가 atexit 전에 프로세스를 끝내므로 env.close 때 출력
        _report()
        return oc(*ca, **ck)

    base.close = close
    return env


gym.make = _make
atexit.register(_report)

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "play_record.py")
sys.argv = [os.path.abspath(script)] + out
runpy.run_path(os.path.abspath(script), run_name="__main__")
