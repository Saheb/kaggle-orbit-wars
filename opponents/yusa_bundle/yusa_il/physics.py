"""physics.py — 純物理計算(学習しないルール部分)。

設計方針: 「本番でも厳密に計算できる物理は学習させない」。
モデルは fire/target/amount の価値判断のみを学習し、以下はルールが担う:
  - 公転惑星・彗星の未来位置予測 (軌道力学 = 環境ダイナミクスそのもの)
  - 発射元→公転標的の最早迎撃時刻の求解
  - 太陽ブロック判定 (太陽横断 = 艦隊即死の回避)
"""
import math

# 太陽の中心座標 (ボードは 100x100、太陽は中央固定)
SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS_SAFETY = 10.0

# ボードの対角線。距離正規化の分母
BOARD_DIAG = 142.0         # 100 * sqrt(2) ≈ 141.4

# 特徴量構築時の固定速度仮定 (num 未定の段階で未来位置を粗く見積もる用)
STATE_SPEED_ASSUMPTION = 4.0

# 迎撃時刻 (travel_time) の求解パラメータ。
#   既定は 3-pass の不動点反復。速い公転 × 遅い艦隊 (|g'|=r·ω/speed≥1) では
#   反復が収束しないため、収束しないと判定したら符号反転走査+二分法に落とす。
INTERCEPT_PICARD_PASSES = 3       # 既定の不動点反復回数 (多数派ケースで十分)
INTERCEPT_TOL = 0.1               # 収束/二分の許容 (ターン)。位置誤差 ≤ r·ω·0.1
INTERCEPT_MAX_T = 250.0           # 二分走査の上限 (ターン)。これ超は迎撃不能扱い
INTERCEPT_SCAN_DT = 1.0           # 符号反転走査の刻み (ターン粒度=自然)


def get_fleet_speed(ships):
    """艦隊速度: log カーブで 1.0〜6.0 (env のルールと同式)。"""
    if ships <= 1:
        return 1.0
    try:
        speed = 1.0 + 5.0 * (math.log(ships) / math.log(1000)) ** 1.5
    except ValueError:
        return 1.0
    return min(speed, 6.0)


def predict_future_position(planet_id, steps, obs, _pdict=None):
    """obs 内 planet_id の `steps` ターン後位置 (x, y)。公転は角速度で先読み、
    静止は現在位置、彗星は軌道 path(path_index+steps) で先読み。

    _pdict={id:planet} を渡すと線形スキャンが O(1) dict 参照になる (出力は不変)。"""
    if _pdict is not None:
        planet = _pdict.get(planet_id)
    else:
        planet = None
        for p in obs.get('planets', []):
            if p[0] == planet_id:
                planet = p
                break
    if planet is None:
        return None

    px, py, pr = planet[2], planet[3], planet[4]

    # 彗星: 楕円軌道 path に沿って先読み (obs.comets の paths/path_index を使用)
    if planet_id in (obs.get('comet_planet_ids') or []):
        for g in (obs.get('comets', []) or []):
            ids = g.get('planet_ids', [])
            if planet_id in ids:
                i = ids.index(planet_id)
                paths = g.get('paths', [])
                if i < len(paths):
                    path = paths[i]
                    idx = g.get('path_index', 0) + int(round(steps))
                    if 0 <= idx < len(path):
                        return path[idx][0], path[idx][1]
                break
        return px, py   # path 範囲外(離脱間際) → 現在位置

    dist_to_sun = math.sqrt((px - SUN_X) ** 2 + (py - SUN_Y) ** 2)
    if dist_to_sun + pr < 50.0:
        ang_vel = obs.get('angular_velocity', 0) or 0
        current_angle = math.atan2(py - SUN_Y, px - SUN_X)
        future_angle = current_angle + ang_vel * steps
        future_x = SUN_X + dist_to_sun * math.cos(future_angle)
        future_y = SUN_Y + dist_to_sun * math.sin(future_angle)
        return future_x, future_y
    return px, py


def _intercept_actual_dist(obs, tid, sx, sy, src_r, tgt_r, t, fallback):
    """時刻 t での標的位置と、発射元からの縁-縁距離 (min 0.1 でクランプ)。
    返り値: (actual_dist, (px, py))。"""
    pred = predict_future_position(tid, t, obs)
    px, py = fallback if pred is None else pred
    d = math.hypot(px - sx, py - sy) - src_r - tgt_r
    return (d if d > 0.1 else 0.1), (px, py)


def solve_intercept(obs, tid, sx, sy, src_r, tgt_r, speed, init_x, init_y):
    """発射元→公転/静止標的の「最早迎撃時刻 t*」と、その時刻の標的位置を返す。

    既定は 3-pass の不動点反復 (travel_time = actual_dist(t)/speed)。
    多数派 (speed > r·ω) はこれで収束。収束しない領域
    (遅い艦隊 × 速い/遠い公転, |g'|=r·ω/speed ≥ 1) を検知したら、
    F(t)=speed·t − actual_dist(t) の最小正根を符号反転走査+二分法で厳密に解く。
    返り値: (target_x, target_y, travel_time)
    """
    speed = speed if speed > 0.1 else 0.1
    fallback = (init_x, init_y)

    # --- Picard: 3-pass まで、収束/発散を監視 ---
    d0, _ = _intercept_actual_dist(obs, tid, sx, sy, src_r, tgt_r, 0.0, fallback)
    t = d0 / speed
    prev_step = None
    converged = False
    for _ in range(INTERCEPT_PICARD_PASSES):
        d, _pos = _intercept_actual_dist(obs, tid, sx, sy, src_r, tgt_r, t, fallback)
        t_new = d / speed
        step = abs(t_new - t)
        t = t_new
        if step < INTERCEPT_TOL:
            converged = True
            break
        if prev_step is not None and step >= prev_step:
            break   # ステップが縮まない = 発散域 → Picard 中断
        prev_step = step

    if converged:
        _, pos = _intercept_actual_dist(obs, tid, sx, sy, src_r, tgt_r, t, fallback)
        return pos[0], pos[1], t

    # --- 収束保証フォールバック: F(t)=speed·t − actual_dist(t) の最小正根 ---
    def _F(tt):
        d, _ = _intercept_actual_dist(obs, tid, sx, sy, src_r, tgt_r, tt, fallback)
        return speed * tt - d

    a, fa = 0.0, _F(0.0)        # F(0) = -d0 < 0
    b = INTERCEPT_SCAN_DT
    bracket = None
    while b <= INTERCEPT_MAX_T:
        fb = _F(b)
        if fa <= 0.0 < fb:
            bracket = (a, b)
            break
        a, fa = b, fb
        b += INTERCEPT_SCAN_DT
    if bracket is None:
        # 上限内に迎撃点なし → 最後の Picard 推定で返す (実質迎撃不能)
        _, pos = _intercept_actual_dist(obs, tid, sx, sy, src_r, tgt_r, t, fallback)
        return pos[0], pos[1], (t if t > 0.0 else 0.0)

    lo, hi = bracket
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _F(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < INTERCEPT_TOL:
            break
    t_star = 0.5 * (lo + hi)
    _, pos = _intercept_actual_dist(obs, tid, sx, sy, src_r, tgt_r, t_star, fallback)
    return pos[0], pos[1], t_star


def sun_blocks_segment(x1, y1, x2, y2):
    """発射元→ターゲットの線分が太陽 (安全半径) を貫くか。"""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return False
    t = ((SUN_X - x1) * dx + (SUN_Y - y1) * dy) / seg_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    px = x1 + t * dx
    py = y1 + t * dy
    return ((px - SUN_X) ** 2 + (py - SUN_Y) ** 2) < (SUN_RADIUS_SAFETY * SUN_RADIUS_SAFETY)


def ray_hits_sun(sx, sy, ang, board=100.0):
    """角度 ang で撃った艦隊の直進レイ(盤面端まで)が太陽を貫くか。
    迎撃補正後の最終角度に対する安全チェック用。"""
    dx, dy = math.cos(ang), math.sin(ang)
    ts = []
    if dx > 1e-12:
        ts.append((board - sx) / dx)
    elif dx < -1e-12:
        ts.append((0.0 - sx) / dx)
    if dy > 1e-12:
        ts.append((board - sy) / dy)
    elif dy < -1e-12:
        ts.append((0.0 - sy) / dy)
    t = min([v for v in ts if v > 0], default=board * 2.0)
    ex, ey = sx + dx * t, sy + dy * t
    return sun_blocks_segment(sx, sy, ex, ey)
