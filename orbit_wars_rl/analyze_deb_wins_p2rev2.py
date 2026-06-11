"""Find + profile our WINS vs debatreya at p2rev2 @4.19M (the 3 panel wins).
Runs the full 256-seed panel (WIN_FULL_PANEL=1, default) — or the 3-archetype subset — saves each win's HTML
replay, and profiles OUR play: planets@N trajectory, material-share curve, whether we
were ever CONTESTED (Deb led) then recovered = a real mid-game HOLD, vs an uncontested
snowball. Goal: is the agent learning to fix the mid-game collapse, or are these lucky
easy seeds?  Run:  CUDA_VISIBLE_DEVICES="" python analyze_deb_wins_p2rev2.py
"""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "opponents")
import torch
from config import Config
from model import EntityTransformer
from eval import load_checkpoint, build_agent_fn, game_conversion, new_conversion_acc, add_conversion, _fmt_conversion
from kaggle_environments import make

CKPT = os.environ.get("WIN_CKPT",
    "gpu_run_artifacts/p2rev2/checkpoints/torch_step_8912896_p2rev2_20260611_105500.pt")
OUTDIR = os.environ.get("WIN_OUTDIR", "/tmp/p2rev2_deb_wins_8.9M"); os.makedirs(OUTDIR, exist_ok=True)
FULL_PANEL = os.environ.get("WIN_FULL_PANEL", "1") == "1"   # always full panel (all 256)
SEAT_COLOR = {0: "BLUE", 1: "ORANGE"}   # orbit wars renderer convention (seat 0 = blue)


def render_banner_html(env, seat, seed, is_win, n_steps, final_lead):
    """env.render(mode=html) + a fixed top banner showing which color WE are (the
    annotation replay_wins.py adds; absent from a plain env.render). seat 0 = BLUE."""
    color = SEAT_COLOR[seat]
    html = env.render(mode="html")
    html = html.replace("<title>", f"<title>[You={color} seat{seat}] ", 1)
    bg = "#1a5276" if color == "BLUE" else "#7d3c00"
    border = "#3498db" if color == "BLUE" else "#e67e22"
    res = "WIN ✓" if is_win else "LOSS ✗"
    banner = f"""
<script>
window.addEventListener('load', function() {{
  var b = document.createElement('div');
  b.innerHTML = 'YOU = <b>{color}</b> (seat {seat}) &nbsp;|&nbsp; Seed {seed} &nbsp;|&nbsp; {res} in {n_steps} steps &nbsp;|&nbsp; Final planet lead: {final_lead:+d}';
  b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:2147483647;background:{bg};color:white;padding:7px 14px;font-size:14px;font-family:monospace;border-bottom:3px solid {border};pointer-events:none;';
  document.body.appendChild(b);
}});
</script>"""
    return html.replace("</body>", banner + "</body>", 1)
from eval_panel import BY_ARCHETYPE
# full panel (all archetypes, all seeds, both seats = 256 games) so we find EVERY win
ARCHES = dict(BY_ARCHETYPE) if FULL_PANEL else {
    "low_prod__mostly_static__big_rotating": [1, 1725, 1948, 2419],
    "low_prod__mostly_rotating__big_static": [417, 451, 2663, 8782],
    "high_prod__mostly_static__big_static":  [1030, 1153, 2726, 8686],
}

cfg = Config(); cfg.env.num_players = 2
sd, _ = load_checkpoint(CKPT, cfg)
device = torch.device(cfg.device)
model = EntityTransformer(cfg.model).to(device)
model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
model.reinforce_gate_min_planets = 3; model.reinforce_forward_only = True; model.reinforce_garrison_floor = 10.0
model.load_state_dict(sd, strict=False); model.eval()
agent_fn = build_agent_fn(model, device, fire_threshold=0.5, target_decode=True)
spec = importlib.util.spec_from_file_location("deb", "opponents/candidate_debatreya_1300.py")
deb = importlib.util.module_from_spec(spec); sys.modules["deb"] = deb; spec.loader.exec_module(deb)


def planet_share(obs, seat):
    pl = obs["planets"]
    me = sum(1 for p in pl if int(p[1]) == seat)
    op = sum(1 for p in pl if int(p[1]) == (1 - seat))
    return me, op


def material(obs, seat):
    pl, fl = obs["planets"], obs["fleets"]
    me = sum(p[5] for p in pl if int(p[1]) == seat) + sum(f[6] for f in fl if int(f[1]) == seat)
    op = sum(p[5] for p in pl if int(p[1]) == (1 - seat)) + sum(f[6] for f in fl if int(f[1]) == (1 - seat))
    return me, op


def profile(steps, seat):
    """Per-step trajectory → contestation/hold signature."""
    T = len(steps)
    pl_at = {}
    mat_share = []           # our material share each step
    deb_ever_led = False; deb_peak_share = 0.0
    our_min_planets_after16 = 99; recovered = False
    for t in range(T):
        obs = steps[t][seat].observation
        me_p, op_p = planet_share(obs, seat)
        me_m, op_m = material(obs, seat)
        tot = me_m + op_m
        sh = me_m / tot if tot > 0 else 0.5
        mat_share.append(sh)
        if t in (16, 32, 50, 100):
            pl_at[t] = me_p
        if sh < 0.5:
            deb_ever_led = True
            deb_peak_share = max(deb_peak_share, 1 - sh)
        if t > 16:
            our_min_planets_after16 = min(our_min_planets_after16, me_p)
    # recovery = Deb led at some point but we ended up winning (we already filter to wins)
    recovered = deb_ever_led
    final = steps[-1]
    rew = [s.reward if s.reward is not None else 0 for s in final]
    deb_dead_step = None
    for t in range(T):
        _, op_m = material(steps[t][seat].observation, seat)
        if op_m == 0:
            deb_dead_step = t; break
    return dict(T=T, pl_at=pl_at, deb_ever_led=deb_ever_led, deb_peak_share=deb_peak_share,
                deb_dead_step=deb_dead_step, mat_share=mat_share, rew=rew)


wins = []
all_acc = new_conversion_acc()
win_acc = new_conversion_acc()
_ngames = sum(len(s) for s in ARCHES.values()) * 2
print(f"checkpoint: {os.path.basename(CKPT)}  (running {_ngames} games vs debatreya)\n")
for arch, seeds in ARCHES.items():
    for seed in seeds:
        for our in (0, 1):
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run([agent_fn, deb.agent] if our == 0 else [deb.agent, agent_fn])
            steps = env.steps
            rew = [s.reward if s.reward is not None else 0 for s in steps[-1]]
            won = rew[our] > rew[1 - our]
            conv = game_conversion(steps, our)
            add_conversion(all_acc, conv)
            if won:
                add_conversion(win_acc, conv)
                p = profile(steps, our)
                p.update(arch=arch, seed=seed, seat=our, conv=conv)
                wins.append(p)
                final_lead = p["pl_at"].get(100, 0)
                me_p, op_p = planet_share(steps[-1][our].observation, our)
                fn = f"{OUTDIR}/WIN_{arch}_seed{seed}_seat{our}.html"
                open(fn, "w").write(
                    render_banner_html(env, our, seed, True, len(steps), me_p - op_p))
                print(f"  WIN  {arch:42s} seed {seed:5d} seat {our} ({SEAT_COLOR[our]})  -> {fn}")

print(f"\n=== {len(wins)} wins found ===")
for p in wins:
    pa = p["pl_at"]
    plstr = "/".join(str(pa.get(m, "-")) for m in (16, 32, 50, 100))
    contest = (f"CONTESTED (Deb led, peak {p['deb_peak_share']:.0%}) then HELD"
               if p["deb_ever_led"] else "uncontested snowball (Deb never led)")
    print(f"\n  {p['arch']} seed {p['seed']} seat {p['seat']}")
    print(f"    game len {p['T']}   Deb eliminated @step {p['deb_dead_step']}")
    print(f"    planets@16/32/50/100 = {plstr}   (top ref 2/6/9/10; loss profile 2/4/6/3)")
    print(f"    {contest}")
    c = p["conv"]
    print(f"    caps {c['captures']}  atk-launch {c['attack_launches']}  "
          f"cap/atk {c['captures']/max(c['attack_launches'],1):.2f}  redundant {c['redundant']/max(c['attack_launches'],1):.2f}")

print("\n=== conversion: WINS only ===")
print(_fmt_conversion(win_acc))
print(f"\n=== conversion: ALL {_ngames} games ===")
print(_fmt_conversion(all_acc))
print(f"\nHTML replays in {OUTDIR}/")
