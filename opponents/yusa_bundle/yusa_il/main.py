# 提出エントリ。2P/4P を obs から自動検出し、mode 別の best checkpoint で推論する。
#   2P = joint13_d384_best_2pbest.pth / 4P = joint13_d384_best_4pbest.pth
# 両 checkpoint とも同一アーキ (共有trunk + mode別head) の joint 学習から、
# 各 mode の評価 best となった epoch の重みを採用したもの。
import os
import sys

try:
    __file__
except NameError:                       # 提出環境の exec 対策
    __file__ = "main.py"
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch                                                       # noqa: E402
from yusa_il import model as M                                                  # noqa: E402

W2 = os.path.join("weights", "joint13_d384_best_2pbest.pth")
W4 = os.path.join("weights", "joint13_d384_best_4pbest.pth")
FIRE_TH = 0.9          # fire 確率のしきい値 (評価スイープで決定)


def _find(fname):
    for d in [_HERE, os.getcwd()] + [p for p in sys.path if p]:
        path = os.path.join(d, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(fname + " not found")


def _load_sd(path):
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                   # 古い torch: weights_only 引数なし
        sd = torch.load(path, map_location="cpu")
    # critic_head は学習時 (RL 併用設計) の遺物で推論では未使用 → 除外して strict load
    return {k: v for k, v in sd.items() if not k.startswith("critic_head.")}


def _net(fname):
    n = M.EntityTFNetDual()
    n.load_state_dict(_load_sd(_find(fname)), strict=True)
    n.eval()
    return n


_agent_2p = M.entity_tf_dual_agent(_net(W2), "cpu", fire_th=FIRE_TH, mode=2)
_agent_4p = M.entity_tf_dual_agent(_net(W4), "cpu", fire_th=FIRE_TH, mode=4)

# 2P/4P 判定: owner>=2 の惑星/艦隊を一度でも見たら 4P。
# step が巻き戻ったら新ゲームとみなしてリセット。
_state = {"seen_4p": False, "last_step": -1}


def _is_4p(obs):
    step = obs.get("step", 0) or 0
    if step < _state["last_step"]:
        _state["seen_4p"] = False
    _state["last_step"] = step
    if not _state["seen_4p"]:
        for p in (obs.get("planets") or []):
            o = p[1]
            if o is not None and o >= 2:
                _state["seen_4p"] = True
                break
        if not _state["seen_4p"]:
            for f in (obs.get("fleets") or []):
                o = f[1]
                if o is not None and o >= 2:
                    _state["seen_4p"] = True
                    break
    return _state["seen_4p"]


def agent(obs):
    act = _agent_4p if _is_4p(obs) else _agent_2p
    return act(obs) or []
