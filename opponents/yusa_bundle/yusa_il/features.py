"""features.py — 特徴抽出。1回の forward = 発射元惑星1つ。

入力ベクトル x[2385] = GLOBAL[45] + 候補60惑星 × [39]。
  GLOBAL      : 発射元と局面の全体情報 (45次元)
  per-target  : 候補惑星ごとの情報 (敵20 / 中立24 / 自16、距離順、39次元)

回転不変設計: 絶対座標・絶対 player 番号は使わない。
  - 方向は「発射元→太陽」を基準軸にした相対角 (radial, tangential) で表現
  - 発射元位置は太陽距離 + 最寄りエッジ距離に縮約
  - 候補の持ち主は絶対番号でなく「強さ順位」で相対化
物理は太陽中心でほぼ等方なので、絶対の向きはタクティクスに本質的でない。
角度そのものはルール側 (実座標 atan2 / 迎撃求解) で計算するため影響なし。
鏡映(左右反転)は公転方向が逆になるため拡張しない (tangential の符号で保持)。

彗星は標的候補から除外 (上位リプレイの観察から彗星はメタに無関係と判断)。
シェア系特徴の分子・分母も彗星を除外して一貫させる。

状態を持つ特徴 (前ターン差分): _PREV_STORE に player 別スナップショットを保持。
step 差がちょうど 1 のときだけ差分を出す (新ゲーム/欠測で嘘の差分を作らない)。
"""
import math

import numpy as np

from yusa_il.physics import (
    BOARD_DIAG, STATE_SPEED_ASSUMPTION,
    get_fleet_speed, predict_future_position, solve_intercept,
    sun_blocks_segment, ray_hits_sun,
)

# ===================== 次元定義 =====================

# 候補スロット数 (距離順・種別ごと)。実ラダー上位の実弾標的の取りこぼし
# (aiming coverage) を実測して広げた値。
N_ENEMIES = 20
N_NEUTRALS = 24
N_OWN = 16
N_TARGETS = N_ENEMIES + N_NEUTRALS + N_OWN          # 60

MAX_ENEMY_FACTIONS = 3                              # 4P で自分以外最大3勢力
N_RANK = 4                                          # 自分のスコア順位 one-hot

# GLOBAL 45 = 基本18 + 敵勢力別3×5 + 艦隊全体量4 + 発射元inbound時刻分解8
GLOBAL_FEAT = 45
# per-target 39 = 基本18 + 前ターン差分3 + inbound時刻分解8 + 最小ETA2
#               + 相対持ち主ID2 + 距離区間one-hot6
PER_TARGET_FEAT = 39
STATE_DIM = GLOBAL_FEAT + N_TARGETS * PER_TARGET_FEAT   # 2385

# amount ヘッド: 「発射元在庫の何割を送るか」の8ビン分類。
# 教師の攻撃量分布は num/在庫 が 0.5 と 0.95-1.0 に二峰集中しており、
# 在庫割合が教師の座標系 (「占領必要数の倍率」ではない)。
# num = round(倍率×在庫) ≤ 在庫 となりクランプが構造的に発生しない。
AMOUNT_VALUES = np.array(
    [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0], dtype=np.float32)
N_AMOUNT_BINS = len(AMOUNT_VALUES)                  # 8

ACTION_DIM = 1 + N_TARGETS + N_AMOUNT_BINS          # fire + target + amount = 69

# inbound 到着時刻バケットの境界 (ターン)
_INBT_NEAR = 8.0
_INBT_MID = 20.0
_INBT_ETA_SCALE = 30.0      # 最小ETA(連続値)の正規化スケール

# 候補距離 (dist/BOARD_DIAG) の区間 one-hot 境界。連続値と二重エンコードして
# 「射程の内/外」等のしきい値判断を使いやすくする (連続+ビニングの併用)。
_BIN_EDGES = (0.07, 0.13, 0.22, 0.35, 0.50)
N_BIN = len(_BIN_EDGES) + 1                          # 6

# 入射角しきい値 0.1rad の cos の二乗 (inbound コーン判定用)
_COS_INC2 = math.cos(0.1) ** 2

_SUN = (50.0, 50.0)
_LOG_REF = math.log1p(2000.0)


def _lognorm(v):
    return min(1.0, math.log1p(max(0.0, float(v))) / _LOG_REF)


def _slog(v):
    """sign付き lognorm (差分量用)。"""
    return math.copysign(_lognorm(abs(v)), v)


# ===================== amount =====================

def amount_base(source_ships):
    """amount ヘッドの倍率を掛ける基準値 = 発射元在庫 (在庫割合の座標系)。"""
    return float(max(1, source_ships))


def class_to_amount(cls):
    """クラス index → 在庫割合。"""
    return float(AMOUNT_VALUES[int(cls)])


# ===================== ターン共有の前計算 =====================

def _incoming(fleets, px, py):
    """点 (px,py) へ向かう fleet の ships 合計。fleet=(x,y,cos_ang,sin_ang,ships)。

    「進行方向」と「fleet→点」の角度差 < 0.1rad をコーン判定するが、
    cos(角度差) = 進行方向単位ベクトル・(fleet→点) / 距離 なので
      dot = dx*cos_ang + dy*sin_ang に対し dot>0 かつ dot² > cos(0.1)²·dist²
    と平方比較すれば atan2/sin/cos/sqrt なしで同一の判定になる。"""
    total = 0.0
    for fx, fy, fcos, fsin, fsh in fleets:
        dx = px - fx
        dy = py - fy
        dot = dx * fcos + dy * fsin
        if dot > 0.0 and dot * dot > _COS_INC2 * (dx * dx + dy * dy):
            total += fsh
    return total


def _inbound_timing(obs, me):
    """惑星ごとの inbound 艦隊を到着時刻バケットに分解 (1ターン1回)。
    返り値: pid -> [e_near, e_mid, e_far, e_n, m_near, m_mid, m_far, m_n,
                    e_min_eta, m_min_eta]
    (生値; 末尾2は最小到着ETA、inbound なし側は inf)。
    コーン判定は _incoming と同一基準。ETA = 現在距離 / 艦隊速度。"""
    out = {}
    fleets = obs.get("fleets", []) or []
    if not fleets:
        return out
    fl = [(f[2], f[3], math.cos(f[4]), math.sin(f[4]), f[6],
           get_fleet_speed(f[6]), 0 if f[1] == me else 1) for f in fleets]
    INF = float("inf")
    for p in obs.get("planets", []):
        pid, px, py = p[0], p[2], p[3]
        acc = [0.0] * 8
        e_min = m_min = INF
        for fx, fy, fc, fs, fsh, fsp, is_en in fl:
            dx = px - fx
            dy = py - fy
            dot = dx * fc + dy * fs
            if dot <= 0.0 or dot * dot <= _COS_INC2 * (dx * dx + dy * dy):
                continue
            eta = math.hypot(dx, dy) / fsp
            b = 0 if eta <= _INBT_NEAR else (1 if eta <= _INBT_MID else 2)
            base = 0 if is_en else 4   # 敵=index0-3 / 味方=index4-7
            acc[base + b] += fsh
            acc[base + 3] += 1
            if is_en:
                e_min = min(e_min, eta)
            else:
                m_min = min(m_min, eta)
        if any(acc):
            acc.append(e_min)   # index 8: 敵の最小到着ETA (なければ inf)
            acc.append(m_min)   # index 9: 味方の最小到着ETA
            out[pid] = acc
    return out


# 前ターン差分特徴の player 別ストア。
# me -> {"prev": snap|None, "cur": snap}  snap={"step","ships","owner","einc"}
_PREV_STORE = {}


def _prev_update(obs, me, ctx):
    """player別ストアを1ターン進め、前ターン snap (step差1のみ) を返す。
    同ターン再呼び出しは進めない (冪等)。"""
    step = int(obs.get("step", 0) or 0)
    snap = {"step": step,
            "ships": {p[0]: p[5] for p in obs.get("planets", [])},
            "owner": {p[0]: p[1] for p in obs.get("planets", [])},
            "einc": dict(ctx["einc"])}
    st = _PREV_STORE.get(me)
    if st is not None and st["cur"]["step"] == step:
        return st["prev"]                     # 同ターン再呼び出し: 進めない
    prev = (st["cur"] if st is not None and step - st["cur"]["step"] == 1
            else None)
    _PREV_STORE[me] = {"prev": prev, "cur": snap}
    return prev


def build_turn_context(obs, me=None, comet_ids=None):
    """ターン共有の前計算。「標的だけで決まる量」(einc/minc/mine_d/enemy_d/
    inbound時刻/前ターンsnap) を惑星ごとに1度だけ計算する。

    extract_state(..., ctx=...) に渡すと per-source の重複計算が消える (出力は不変)。
    呼び出し側はターン (=同一 obs) ごとに1度だけ作って全発射元に渡すこと。"""
    if me is None:
        me = obs.get("player", 0)
    planets = obs.get("planets", [])
    all_fleets = obs.get("fleets", []) or []
    # _incoming 用に進行方向の cos/sin をターン中1度だけ事前計算した tuple
    en_fleets = [(f[2], f[3], math.cos(f[4]), math.sin(f[4]), f[6])
                 for f in all_fleets if f[1] != me]
    my_fleets = [(f[2], f[3], math.cos(f[4]), math.sin(f[4]), f[6])
                 for f in all_fleets if f[1] == me]
    my_pl = [p for p in planets if p[1] == me]
    en_pl = [p for p in planets if p[1] != me and p[1] != -1]
    einc, minc, mine_d, enemy_d = {}, {}, {}, {}
    for p in planets:
        pid, px, py = p[0], p[2], p[3]
        einc[pid] = _incoming(en_fleets, px, py)
        minc[pid] = _incoming(my_fleets, px, py)
        mine_d[pid] = min((math.hypot(px - m[2], py - m[3])
                           for m in my_pl if m[0] != pid), default=BOARD_DIAG)
        enemy_d[pid] = min((math.hypot(px - e[2], py - e[3])
                            for e in en_pl if e[0] != pid), default=BOARD_DIAG)
    ctx = {"me": me, "en_fleets": en_fleets, "my_fleets": my_fleets,
           "my_pl": my_pl, "en_pl": en_pl,
           "einc": einc, "minc": minc, "mine_d": mine_d, "enemy_d": enemy_d}
    ctx["prev"] = _prev_update(obs, me, ctx)
    ctx["inbt"] = _inbound_timing(obs, me)
    return ctx


# ===================== 局面ヘルパ =====================

def detect_game_players(obs):
    """開始時の非中立 owner 数 = 実プレイヤー数 (2 or 4)。"""
    init = obs.get("initial_planets") or obs.get("planets") or []
    owners = {p[1] for p in init if len(p) > 1 and p[1] is not None and p[1] >= 0}
    return len(owners) if owners else 2


def _compute_rank(me, planets, fleets):
    """自分の総スコア順位 (1=top..4=bottom)。
    score = 所有惑星艦船 + 自軍艦隊艦船 合計。"""
    scores = [0.0] * 4
    for p in planets:
        o = p[1]
        if 0 <= o < 4:
            scores[o] += p[5]
    for f in fleets or []:
        o = f[1]
        if 0 <= o < 4:
            scores[o] += f[6]
    if me >= 4:
        return 4
    sorted_idx = sorted(range(4), key=lambda i: -scores[i])
    return sorted_idx.index(me) + 1


def _sun_axis(sx, sy):
    """発射元→太陽 の単位ベクトル (方向特徴の基準軸)。"""
    ax, ay = _SUN[0] - sx, _SUN[1] - sy
    n = math.hypot(ax, ay)
    if n < 1e-9:
        return 1.0, 0.0
    return ax / n, ay / n


# ===================== per-target 列レイアウト =====================

def _per_target_layout():
    """候補ブロックの (特徴名, 次元数) を append 順で返す。
    extract_state の features.extend 順と一致させること。"""
    return [("base", 18), ("prev_entity", 3), ("inbound_t", 8),
            ("eta", 2), ("factionid", 2), ("bin_onehot", N_BIN)]


def mode_drop_indices(names):
    """指定の per-target 特徴を全候補ぶんゼロにするための絶対列 index リスト。
    2P では 4P 専用特徴 (eta/factionid) を入力マスクで落とす用途。"""
    off = {}
    o = 0
    for nm, nd in _per_target_layout():
        off[nm] = (o, nd)
        o += nd
    assert o == PER_TARGET_FEAT, (o, PER_TARGET_FEAT)
    idx = []
    for ci in range(N_TARGETS):
        b = GLOBAL_FEAT + ci * PER_TARGET_FEAT
        for nm in names:
            if nm not in off:
                continue
            s, nd = off[nm]
            idx.extend(range(b + s, b + s + nd))
    return idx


# padding 候補の per-target ベクトル:
#   距離系 (先頭 dist / mine_d / enemy_d) = 1.0 (遠い), カウント系 = 0.0,
#   最小ETA = 1.0 (inbound なし=遠い), それ以外 = 0.0
_PAD = ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # base 18
        + [0.0, 0.0, 0.0]                                # 前ターン差分 3
        + [0.0] * 8                                      # inbound 時刻分解 8
        + [1.0, 1.0]                                     # 最小ETA 2
        + [0.0, 0.0]                                     # 相対持ち主ID 2
        + [0.0] * N_BIN)                                 # 距離区間 one-hot 6
assert len(_PAD) == PER_TARGET_FEAT


# ===================== 特徴抽出本体 =====================

def extract_state(obs, planet_id, comet_ids=None, ctx=None):
    """発射元惑星 planet_id から見た局面を (STATE_DIM vec, candidates) で返す。

    candidates[i] = {planet_id, x, y, is_padding, is_sun_blocked}
    is_sun_blocked は未来位置基準 (実発射は未来位置狙いのため。現在位置基準だと
    「現在貫通/未来クリア」の誤禁止と「現在クリア/未来貫通」のマスク漏れが出る)。
    ctx: build_turn_context() の結果 (None なら内部で構築)。"""
    if comet_ids is None:
        comet_ids = set(obs.get("comet_planet_ids", []) or [])

    planets = obs.get("planets", [])
    p_dict = {p[0]: p for p in planets}
    me = obs.get("player", 0)
    if planet_id not in p_dict:
        return None
    source_p = p_dict[planet_id]
    sx, sy = source_p[2], source_p[3]
    src_ships = source_p[5]

    # ctx 未指定 or me 不一致なら作り直す (誤った ctx 流用を防ぐ防御)
    if ctx is None or ctx.get("me") != me:
        ctx = build_turn_context(obs, me, comet_ids)
    my_pl, en_pl = ctx["my_pl"], ctx["en_pl"]
    my_ships = sum(p[5] for p in my_pl)
    en_ships = sum(p[5] for p in en_pl)
    my_prod = sum(p[6] for p in my_pl)
    en_prod = sum(p[6] for p in en_pl)
    # シェア系の分母・分子とも彗星除外で一貫
    total_pl = len([p for p in planets if p[0] not in comet_ids])

    all_fleets = obs.get("fleets", []) or []
    n_players = detect_game_players(obs)

    # 敵を勢力別集計 (ships/prod/最寄り距離/艦隊艦船/惑星数)
    fac = {}
    for p in en_pl:
        o = p[1]
        d = fac.setdefault(o, {"s": 0.0, "p": 0.0, "md": BOARD_DIAG,
                               "fl": 0.0, "pc": 0})
        d["s"] += p[5]
        d["p"] += p[6]
        d["pc"] += 0 if p[0] in comet_ids else 1
        dd = math.hypot(p[2] - sx, p[3] - sy)
        if dd < d["md"]:
            d["md"] = dd
    for f in all_fleets:
        if f[1] != me and f[1] != -1 and f[1] in fac:
            fac[f[1]]["fl"] += f[6]
    fac_sorted = sorted(fac.values(), key=lambda x: x["s"], reverse=True)
    # 持ち主→強さ順位 (0=最強)。候補の「相対持ち主ID」に使う (席番号非依存)
    fac_rank = {o: i for i, (o, _) in enumerate(
        sorted(fac.items(), key=lambda kv: -kv[1]["s"]))}
    alive_en = len(fac)
    total_fleet_ships = sum(f[6] for f in all_fleets)

    my_rank = _compute_rank(me, planets, all_fleets)
    rank_oh = [1.0 if my_rank == k + 1 else 0.0 for k in range(N_RANK)]

    top_threat_ships = fac_sorted[0]["s"] if fac_sorted else 0.0
    threat_diff = (my_ships - top_threat_ships) / \
                  (my_ships + top_threat_ships + 1.0)

    src_einc = ctx["einc"][planet_id]

    # 発射元位置の回転不変表現: 太陽距離 + 最寄りエッジ距離
    src_sun_dist = math.hypot(sx - _SUN[0], sy - _SUN[1])
    src_edge_min = min(sx, 100.0 - sx, sy, 100.0 - sy)

    # ---------- GLOBAL (45) ----------
    g = [
        min(1.0, src_sun_dist / BOARD_DIAG),
        min(1.0, src_edge_min / 50.0),
        _lognorm(src_ships),
        source_p[6] / 5.0,
        source_p[4] / 5.0,
        obs.get("step", 0) / 500.0,
        (n_players - 2) / 2.0,
        min(1.0, alive_en / 3.0),
        my_ships / (my_ships + en_ships + 1.0),
        len([p for p in my_pl if p[0] not in comet_ids]) / (total_pl + 1.0),
        my_prod / (my_prod + en_prod + 1.0),
        (obs.get("angular_velocity", 0.0) or 0.0) / 0.05,
        src_einc / (src_einc + src_ships + 1.0),
        threat_diff,
        rank_oh[0], rank_oh[1], rank_oh[2], rank_oh[3],
    ]
    for k in range(MAX_ENEMY_FACTIONS):
        if k < len(fac_sorted):
            d = fac_sorted[k]
            g += [d["s"] / (my_ships + en_ships + 1.0),
                  d["p"] / (my_prod + en_prod + 1.0),
                  min(1.0, d["md"] / BOARD_DIAG),
                  d["fl"] / (total_fleet_ships + 1.0),    # fleets_share
                  d["pc"] / (total_pl + 1.0)]             # planets_share
        else:
            g += [0.0, 0.0, 1.0, 0.0, 0.0]
    # 飛行中艦隊の全体量 (駐留のみシェアの盲点補正):
    #   艦隊込み戦力シェア / 自展開率 / 敵展開率 / 空中戦力の優劣
    myf = sum(f[6] for f in all_fleets if f[1] == me)
    enf = sum(f[6] for f in all_fleets if f[1] != me and f[1] != -1)
    g += [
        (my_ships + myf) / (my_ships + myf + en_ships + enf + 1.0),
        myf / (my_ships + myf + 1.0),
        enf / (en_ships + enf + 1.0),
        (myf - enf) / (myf + enf + 1.0),
    ]
    # 発射元自身への inbound 時刻分解 (「いつ襲われる/援軍いつ」→撃つか守るか)
    s = ctx["inbt"].get(planet_id)
    if s is None:
        g += [0.0] * 8
    else:
        g += [
            _lognorm(s[0]), _lognorm(s[1]), _lognorm(s[2]),
            min(1.0, s[3] / 4.0),
            _lognorm(s[4]), _lognorm(s[5]), _lognorm(s[6]),
            min(1.0, s[7] / 4.0),
        ]
    assert len(g) == GLOBAL_FEAT, (len(g), GLOBAL_FEAT)

    # 基準軸 (発射元→太陽)。標的方向の相対角に使う
    ax, ay = _sun_axis(sx, sy)

    # ---------- 候補 (距離順・種別ごと、彗星除外) ----------
    others = []
    for p in planets:
        if p[0] == planet_id or p[0] in comet_ids:
            continue
        dx, dy = p[2] - sx, p[3] - sy
        others.append((math.hypot(dx, dy), dx, dy, p))
    others.sort(key=lambda x: x[0])
    enemies = [o for o in others if o[3][1] != me and o[3][1] != -1][:N_ENEMIES]
    neutrals = [o for o in others if o[3][1] == -1][:N_NEUTRALS]
    own = [o for o in others if o[3][1] == me][:N_OWN]

    features = list(g)
    candidates = []

    def _push(lst, count):
        for i in range(count):
            if i >= len(lst):
                features.extend(_PAD)
                candidates.append({"planet_id": -1, "x": 0.0, "y": 0.0,
                                   "is_padding": True, "is_sun_blocked": False})
                continue
            dist, dx, dy, p = lst[i]
            inv = 1.0 / dist if dist > 1e-6 else 0.0
            blocked = sun_blocks_segment(sx, sy, p[2], p[3])
            tt = dist / STATE_SPEED_ASSUMPTION if dist > 1e-6 else 0.0
            fut = predict_future_position(p[0], tt, obs, _pdict=p_dict)
            fx, fy = (p[2], p[3]) if fut is None else fut
            fdx, fdy = fx - sx, fy - sy
            fdist = math.hypot(fdx, fdy)
            finv = 1.0 / fdist if fdist > 1e-6 else 0.0
            fblk = sun_blocks_segment(sx, sy, fx, fy)

            # 標的方向を太陽軸基準の相対角へ (回転不変):
            #   radial = cos(標的方向, 太陽軸) / tangential = signed sin (公転方向の符号)
            ux, uy = dx * inv, dy * inv
            radial = ux * ax + uy * ay
            tangential = ux * ay - uy * ax
            fux, fuy = fdx * finv, fdy * finv
            fradial = fux * ax + fuy * ay
            ftangential = fux * ay - fuy * ax

            pid = p[0]
            mine_d = ctx["mine_d"][pid]
            enemy_d = ctx["enemy_d"][pid]
            einc = ctx["einc"][pid]
            minc = ctx["minc"][pid]

            owner = p[1]
            is_enemy = 1.0 if (owner != me and owner != -1) else 0.0
            is_neutral = 1.0 if owner == -1 else 0.0
            # ships_needed: 占領に必要な最小数 (敵は到着時生産込み、自は概念なし)
            if owner == me:
                sn = 0.0
            elif owner == -1:
                sn = p[5] + 1.0
            else:
                sn = p[5] + tt * p[6] + 1.0
            net = max(0.0, sn + einc - minc)

            # --- base 18 ---
            features.extend([
                dist / BOARD_DIAG,
                radial,
                tangential,
                _lognorm(p[5]),
                p[6] / 5.0,
                1.0 if blocked else 0.0,
                fradial,
                ftangential,
                1.0 if fblk else 0.0,
                min(1.0, mine_d / BOARD_DIAG),
                min(1.0, enemy_d / BOARD_DIAG),
                einc / (einc + p[5] + 1.0),
                minc / (minc + sn + 1.0),
                _lognorm(sn),
                min(1.0, net / (src_ships + 1.0)),
                _lognorm(net),
                is_enemy,
                is_neutral,
            ])
            # --- 前ターン差分 3 (現在 obs から導出できないイベント情報のみ) ---
            #   Δ艦数 / 所有変化 / Δ敵inbound。step差1でなければ全0
            pv = ctx.get("prev")
            if pv is None or pid not in pv["ships"]:
                features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([
                    _slog(p[5] - pv["ships"][pid]),
                    1.0 if p[1] != pv["owner"][pid] else 0.0,
                    _slog(einc - pv["einc"].get(pid, 0.0)),
                ])
            # --- inbound 時刻分解 8 (いつ・何waveで来るか) ---
            ib = ctx["inbt"].get(pid)
            if ib is None:
                features.extend([0.0] * 8)
            else:
                features.extend([
                    _lognorm(ib[0]), _lognorm(ib[1]), _lognorm(ib[2]),
                    min(1.0, ib[3] / 4.0),
                    _lognorm(ib[4]), _lognorm(ib[5]), _lognorm(ib[6]),
                    min(1.0, ib[7] / 4.0),
                ])
            # --- 最小ETA 2 (バケットと連続値の二重エンコード) ---
            if ib is None:
                features.extend([1.0, 1.0])     # inbound なし = 遠い
            else:
                features.extend([
                    min(1.0, ib[8] / _INBT_ETA_SCALE),   # 敵最小ETA (inf→1.0)
                    min(1.0, ib[9] / _INBT_ETA_SCALE),   # 味方最小ETA
                ])
            # --- 相対持ち主ID 2 (4P用: どの敵か。強さ順位で席番号非依存) ---
            if owner != me and owner != -1 and owner in fac_rank:
                features.extend([
                    fac[owner]["s"] / (my_ships + en_ships + 1.0),
                    (MAX_ENEMY_FACTIONS - fac_rank[owner])
                    / float(MAX_ENEMY_FACTIONS),
                ])
            else:
                features.extend([0.0, 0.0])   # 中立/自分
            # --- 距離区間 one-hot 6 ---
            dn = dist / BOARD_DIAG
            b = 0
            for e in _BIN_EDGES:
                if dn >= e:
                    b += 1
            oh = [0.0] * N_BIN
            oh[b] = 1.0
            features.extend(oh)

            candidates.append({"planet_id": p[0], "x": p[2], "y": p[3],
                               "is_padding": False,
                               "is_sun_blocked": bool(fblk)})

    _push(enemies, N_ENEMIES)
    _push(neutrals, N_NEUTRALS)
    _push(own, N_OWN)

    vec = np.array(features, dtype=np.float32)
    assert vec.shape[0] == STATE_DIM, (vec.shape[0], STATE_DIM)
    return vec, candidates


# ===================== move 生成 (モデル出力 → 環境 action) =====================

def move_from_choice(obs, p, cand, amount_cls, me, future_aim=True):
    """モデルの選択 (候補 index / amount ビン) を環境の move [id, angle, num] に変換。

    ここから先は全て物理: 量 = 倍率×在庫、迎撃求解で未来位置へ atan2、
    最終角度の太陽レイチェック。"""
    sx, sy = p[2], p[3]
    source_ships = p[5]
    source_radius = p[4]
    tid = cand["planet_id"]
    target_p = next((q for q in obs.get("planets", []) if q[0] == tid), None)
    if target_p is None:
        return None
    target_radius = target_p[4]

    num = int(round(class_to_amount(int(amount_cls)) * amount_base(source_ships)))
    num = min(num, source_ships)
    if num < 1:
        return None

    speed = get_fleet_speed(num)
    cx, cy = cand["x"], cand["y"]
    if future_aim:
        tx, ty, _ = solve_intercept(
            obs, tid, sx, sy, source_radius, target_radius, speed, cx, cy)
    else:
        tx, ty = cx, cy
    ang = math.atan2(ty - sy, tx - sx)
    if ray_hits_sun(sx, sy, ang):
        return None
    return [p[0], ang, num]
