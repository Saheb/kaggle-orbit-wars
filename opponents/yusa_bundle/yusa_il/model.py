"""model.py — 共有 trunk + 2P/4P 別 head の Entity Transformer。

構成 (7.21M params):
  ctx_proj(45→384) + persona sub_emb ─┐
  cand_proj(39→384) ×60候補          ─┴→ 61 tokens
    → TransformerEncoder ×6 (d=384, 8 heads, FF 768)   … 2P/4P 共有 trunk (98.5%)
    → fire / target / amount head (mode別 '2'/'4')     … 0.2%

設計意図:
  - 物理・幾何の理解 (trunk) は 2P/4P 共通 → 両モードのデータで学習しデータ効率を上げる。
    戦略の価値づけ (head) は敵1体/3体で異なる → mode 別に分離。
    trunk は両モードの勾配で更新、各 head は自モードの勾配のみで更新。
  - persona sub_emb: 教師チーム(全提出sub)の ID 埋め込みを文脈トークンに加算。
    複数チームの矛盾するラベル (同じ盤面で違う手) を「平均」せず、チーム別の
    条件付き方策として分離する。推論時は最強チームの ID を固定 (=学習されたプロンプト)。
  - 2P では 4P 専用特徴 (eta/factionid) を入力マスクでゼロ化 (mode 別特徴)。
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                   # 提出環境の exec 対策
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from yusa_il import features as feat                                            # noqa: E402

# ---- アーキテクチャ定数 (checkpoint と一体) ----
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 6
FF = 768
DROPOUT = 0.1          # 学習時の値。eval() では無効 (構成の記録として保持)

# 2P 入力でゼロ化する per-target 特徴 (4P 専用特徴を 2P から外す)
MODE2P_DROP = ("eta", "factionid")

# persona sub_emb の語彙 = 学習に使った全教師 submission ID (153個)。
# checkpoint の sub_emb.weight (153×384) の行順と一対一対応なので変更不可。
SUB_EMBED_IDS = [
    52449013, 52476017, 52504445, 52514115, 52514126, 52526940, 52558368,
    52558371, 52737137, 52754953, 52814820, 52825439, 52826010, 52832131,
    52903099, 52903587, 52951805, 52952321, 52967053, 52982631, 52984926,
    52999924, 53019594, 53032640, 53032642, 53039344, 53039522, 53064255,
    53064261, 53084172, 53084181, 53094091, 53094527, 53100676, 53114654,
    53114660, 53116971, 53120522, 53121896, 53122039, 53163196, 53203890,
    53204000, 53210820, 53232000, 53236656, 53255109, 53264211, 53265042,
    53291341, 53291917, 53304314, 53305666, 53319627, 53340181, 53345805,
    53348314, 53356055, 53365802, 53366684, 53387296, 53402231, 53402535,
    53403511, 53403518, 53406662, 53432034, 53433159, 53433867, 53434004,
    53436020, 53439389, 53450554, 53458575, 53486556, 53503159, 53503178,
    53505965, 53507583, 53507926, 53507928, 53507942, 53509496, 53511969,
    53513895, 53513997, 53517858, 53521407, 53522127, 53526382, 53540104,
    53543377, 53545513, 53546944, 53550281, 53560488, 53566715, 53570126,
    53571839, 53572155, 53572498, 53573102, 53579476, 53605769, 53614754,
    53617842, 53618545, 53632494, 53642762, 53655226, 53655239, 53662527,
    53663545, 53673475, 53685430, 53685469, 53688190, 53699976, 53709383,
    53710259, 53711102, 53717801, 53720816, 53720830, 53721352, 53721629,
    53724336, 53729375, 53730795, 53733351, 53733796, 53735461, 53744623,
    53749015, 53750716, 53755383, 53755733, 53755916, 53768374, 53769902,
    53771674, 53775422, 53775429, 53779025, 53780051, 53785550, 53789220,
    53848103, 53848114, 53860711, 53860828, 53864291, 53864310,
]

# 推論時に条件付けする人格 (=最強だった教師 sub の ID)
SUB_INFER_ID = 53204000

_MODES = ("2", "4")


def _mode_key(mode):
    """mode(int/str) → '2' or '4'。3以上は 4P 扱い。"""
    try:
        m = int(mode)
    except (TypeError, ValueError):
        m = 2
    return "2" if m <= 2 else "4"


class EntityTFNetDual(nn.Module):
    """共有 trunk + 2P/4P 別 head。forward は「発射元惑星1つ」単位。"""

    def __init__(self):
        super().__init__()
        dm = D_MODEL
        # ---- 共有 trunk ----
        self.ctx_proj = nn.Linear(feat.GLOBAL_FEAT, dm)
        self.cand_proj = nn.Linear(feat.PER_TARGET_FEAT, dm)
        self.type_emb = nn.Embedding(2, dm)          # 0=文脈トークン / 1=候補トークン
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dm, nhead=N_HEADS, dim_feedforward=FF,
            dropout=DROPOUT, batch_first=True, activation="gelu")
        # enable_nested_tensor=False: nested-tensor 高速パスを無効化。
        # CUDA + batch=1 の eval forward で C 層 crash する prototype パスの回避。
        # 重み不変・出力等価 (padding は src_key_padding_mask で除外)。
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=N_LAYERS, enable_nested_tensor=False)
        # persona sub_emb (checkpoint の行順 = SUB_EMBED_IDS の順)
        self.sub_emb = nn.Embedding(len(SUB_EMBED_IDS), dm)
        self._sub_infer_ix = SUB_EMBED_IDS.index(SUB_INFER_ID)
        # 2P 入力で落とす列 = 0, それ以外 1 のマスク (buffer = device に追従)
        m = torch.ones(feat.STATE_DIM)
        m[feat.mode_drop_indices(MODE2P_DROP)] = 0.0
        self.register_buffer("_mode2p_mask", m)
        # ---- 2P/4P 別 head (ModuleDict, キー '2'/'4') ----
        self.fire_head = nn.ModuleDict(
            {k: nn.Linear(2 * dm, 1) for k in _MODES})
        self.target_head = nn.ModuleDict(
            {k: nn.Linear(dm, 1) for k in _MODES})
        self.amount_head = nn.ModuleDict(
            {k: nn.Linear(2 * dm, feat.N_AMOUNT_BINS) for k in _MODES})

    def _encode(self, x, mask, sub_idx=None):
        """x[B,2385], mask[B,60] → cand_out[B,60,D], ctx_out[B,D], pooled[B,D]。
        sub_idx[B](long): 学習時はサンプルの教師 sub index。
        None (推論) = 固定 ID で条件付け (学習されたプロンプト)。"""
        B = x.shape[0]
        ctx = x[:, :feat.GLOBAL_FEAT]
        cand = x[:, feat.GLOBAL_FEAT:].reshape(
            B, feat.N_TARGETS, feat.PER_TARGET_FEAT)
        ctx_tok = self.ctx_proj(ctx).unsqueeze(1)
        cand_tok = self.cand_proj(cand)
        ctx_tok = ctx_tok + self.type_emb(
            torch.zeros(B, 1, dtype=torch.long, device=x.device))
        if sub_idx is None:
            sub_idx = torch.full((B,), self._sub_infer_ix,
                                 dtype=torch.long, device=x.device)
        ctx_tok = ctx_tok + self.sub_emb(sub_idx).unsqueeze(1)
        cand_tok = cand_tok + self.type_emb(
            torch.ones(B, feat.N_TARGETS, dtype=torch.long, device=x.device))
        tokens = torch.cat([ctx_tok, cand_tok], dim=1)
        ctx_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        cand_pad = mask < 0.5
        key_pad = torch.cat([ctx_pad, cand_pad], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=key_pad)
        ctx_out = out[:, 0]
        cand_out = out[:, 1:]
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = cand_out.masked_fill(m < 0.5, 0.0).sum(dim=1) / denom
        return cand_out, ctx_out, pooled

    def policy_logits(self, x, mask, mode=2, sub_idx=None):
        k = _mode_key(mode)
        if k == "2":
            x = x * self._mode2p_mask       # 2P: 4P専用特徴をゼロ化
        cand_out, ctx_out, pooled = self._encode(x, mask, sub_idx)
        ctx_pool = torch.cat([ctx_out, pooled], dim=-1)
        fire = self.fire_head[k](ctx_pool)
        amount = self.amount_head[k](ctx_pool)
        target = self.target_head[k](cand_out).squeeze(-1)
        return torch.cat([fire, target, amount], dim=-1)

    def forward(self, x, mask, mode=2, sub_idx=None):
        return self.policy_logits(x, mask, mode, sub_idx)


def entity_tf_dual_agent(net, device="cpu", fire_th=0.5, mode=None):
    """agent(obs) -> moves のクロージャを返す。decode は greedy。

    mode=None なら obs からプレイヤー数を検出して 2P/4P head を自動選択。
    fire 確率がしきい値 fire_th を超えた発射元だけが発射する。"""
    net.eval()
    nt = feat.N_TARGETS
    nb = feat.N_AMOUNT_BINS
    fire_logit_th = float(np.log(fire_th / (1.0 - fire_th)))

    def act(obs):
        m = mode if mode is not None else feat.detect_game_players(obs)
        me = obs.get("player", 0)
        cids = set(obs.get("comet_planet_ids", []) or [])
        ctx = feat.build_turn_context(obs, me, cids)
        # ① 全所有惑星の (vec, cands, mask) を集める
        rows = []   # (p, cands, mask_np)
        vecs = []
        masks_np = []
        for p in obs["planets"]:
            if p[1] != me or p[5] < 1:
                continue
            res = feat.extract_state(obs, p[0], cids, ctx)
            if res is None:
                continue
            vec, cands = res
            mask_np = np.array(
                [0.0 if (c["is_padding"] or c["is_sun_blocked"]) else 1.0
                 for c in cands], dtype=np.float32)
            if mask_np.sum() < 0.5:
                continue
            rows.append((p, cands, mask_np))
            vecs.append(vec)
            masks_np.append(mask_np)
        if not rows:
            return []
        # ② 全惑星を 1 batch で forward (各行は独立 = per-planet 実行と数値同一)
        X = torch.tensor(np.stack(vecs), dtype=torch.float32, device=device)
        M = torch.tensor(np.stack(masks_np), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = net.policy_logits(X, M, m)
        out = out.cpu().numpy()
        # ③ 行ごとに greedy decode → 物理で move へ変換
        moves = []
        for i, (p, cands, mask_np) in enumerate(rows):
            o = out[i]
            if o[0] <= fire_logit_th:
                continue
            tgt = o[1:1 + nt].copy()
            tgt[mask_np < 0.5] = -1e30
            ti = int(np.argmax(tgt))
            amt = int(np.argmax(o[1 + nt:1 + nt + nb]))
            mv = feat.move_from_choice(obs, p, cands[ti], amt, me,
                                       future_aim=True)
            if mv is not None:
                moves.append(mv)
        return moves
    return act
