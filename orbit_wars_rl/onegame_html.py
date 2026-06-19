"""Render a replay JSON (from onegame_deep.py) into a self-contained HTML viewer."""
import argparse
import json
import os

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Orbit Wars replay — seed {seed} ({result})</title>
<style>
 body{{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.5 ui-monospace,Menlo,monospace}}
 .wrap{{display:flex;gap:16px;padding:16px;flex-wrap:wrap}}
 canvas{{background:#05080d;border:1px solid #21262d;border-radius:8px}}
 .panel{{min-width:300px;flex:1}}
 h1{{font-size:15px;margin:0 0 8px}}
 .stat{{display:flex;gap:14px;margin:6px 0;flex-wrap:wrap}}
 .stat b{{font-weight:700}}
 .us{{color:#4ea1ff}} .opp{{color:#ff5b5b}} .neu{{color:#8b949e}}
 .controls{{display:flex;gap:10px;align-items:center;margin:10px 0}}
 input[type=range]{{flex:1}}
 button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:5px 12px;cursor:pointer}}
 #log{{height:260px;overflow:auto;border:1px solid #21262d;border-radius:6px;padding:8px;background:#0a0e14}}
 .ev{{margin:2px 0}} .now{{background:#1f2937;border-radius:4px}}
 .cap{{color:#56d364}} .loss{{color:#ff7b72}}
 .ratio{{font-size:20px;font-weight:700}}
</style></head><body>
<div class="wrap">
 <div>
  <canvas id="cv" width="600" height="600"></canvas>
  <div class="controls">
   <button id="play">▶ play</button>
   <input id="slider" type="range" min="0" max="{maxt}" value="0">
   <span id="tlabel">t 0 / {maxt}</span>
   <select id="spd"><option value="120">1×</option><option value="60">2×</option><option value="30">4×</option></select>
  </div>
 </div>
 <div class="panel">
  <h1>seed {seed} — <span class="{rcls}">{result}</span> (len {n})</h1>
  <div class="stat"><span>planets <b class="us" id="op">–</b> us / <b class="opp" id="pp">–</b> opp / <b class="neu" id="np">–</b> neutral</span></div>
  <div class="stat"><span>mass <b class="us" id="om">–</b> us / <b class="opp" id="pm">–</b> opp</span> <span>ratio <b class="ratio" id="rt">–</b></span></div>
  <canvas id="chart" width="300" height="80" style="margin:6px 0"></canvas>
  <div id="log"></div>
 </div>
</div>
<script>
const R = {data};
const cv=document.getElementById('cv'),g=cv.getContext('2d');
const W=cv.width,P=24,S=(W-2*P)/100;            // board 0..100 -> pixels
const col=o=>o===0?'#4ea1ff':o===1?'#ff5b5b':'#6b7280';
function X(x){{return P+x*S}} function Y(y){{return P+y*S}}
function draw(t){{
 const f=R.frames[t];
 g.clearRect(0,0,W,W);
 // sun
 g.fillStyle='#3a2f0b'; g.beginPath(); g.arc(X(50),Y(50),10*S,0,7); g.fill();
 // fleets (dot + faint line to target)
 const pById={{}}; f.planets.forEach(p=>pById[p.id]=p);
 f.fleets.forEach(fl=>{{
   const tp=pById[fl.tgt];
   if(tp){{g.strokeStyle=col(fl.own)+'55';g.lineWidth=1;g.beginPath();g.moveTo(X(fl.x),Y(fl.y));g.lineTo(X(tp.x),Y(tp.y));g.stroke();}}
   g.fillStyle=col(fl.own); g.beginPath(); g.arc(X(fl.x),Y(fl.y),Math.max(2,Math.sqrt(fl.ships)*0.7),0,7); g.fill();
 }});
 // planets
 f.planets.forEach(p=>{{
   const r=Math.max(6,p.r*S);
   g.fillStyle=col(p.own); g.globalAlpha=0.28; g.beginPath(); g.arc(X(p.x),Y(p.y),r,0,7); g.fill();
   g.globalAlpha=1; g.strokeStyle=col(p.own); g.lineWidth=2; g.beginPath(); g.arc(X(p.x),Y(p.y),r,0,7); g.stroke();
   g.fillStyle='#e6edf3'; g.font='11px monospace'; g.textAlign='center'; g.textBaseline='middle';
   g.fillText(Math.round(p.ships),X(p.x),Y(p.y));
   g.fillStyle='#8b949e'; g.font='8px monospace'; g.fillText('p'+p.id+' ·'+p.prod,X(p.x),Y(p.y)+r+7);
 }});
 // stats
 document.getElementById('op').textContent=f.our_p;
 document.getElementById('pp').textContent=f.opp_p;
 document.getElementById('np').textContent=f.neu_p;
 document.getElementById('om').textContent=Math.round(f.our_m);
 document.getElementById('pm').textContent=Math.round(f.opp_m);
 const rt=f.our_m/Math.max(f.opp_m,1);
 const re=document.getElementById('rt'); re.textContent=rt.toFixed(2);
 re.className='ratio '+(rt>=1?'us':'opp');
 document.getElementById('tlabel').textContent='t '+t+' / {maxt}';
 document.getElementById('slider').value=t;
 chart(t); renderLog(t);
}}
const ch=document.getElementById('chart'),cg=ch.getContext('2d');
function chart(t){{
 cg.clearRect(0,0,300,80);
 cg.strokeStyle='#30363d'; cg.beginPath(); cg.moveTo(0,40); cg.lineTo(300,40); cg.stroke(); // ratio=1
 cg.strokeStyle='#58a6ff'; cg.lineWidth=1.5; cg.beginPath();
 R.frames.forEach((f,i)=>{{
   const rt=Math.min(2,f.our_m/Math.max(f.opp_m,1));
   const x=i/{maxt}*300, y=80-rt/2*80;
   i?cg.lineTo(x,y):cg.moveTo(x,y);
 }});
 cg.stroke();
 const x=t/{maxt}*300; cg.strokeStyle='#f0f6fc'; cg.beginPath(); cg.moveTo(x,0); cg.lineTo(x,80); cg.stroke();
}}
function renderLog(t){{
 const L=document.getElementById('log');
 const who=o=>o===0?'US':o===1?'OPP':'neutral';
 let h='';
 R.events.filter(e=>e.t<=t).forEach(e=>{{
   const cls=e.to===0?'cap':(e.frm===0?'loss':'');
   const tag=e.to===0?'CAPTURE':(e.frm===0?'LOST':'flip');
   const now=e.t===t?' now':'';
   h+=`<div class="ev ${{cls}}${{now}}">t${{e.t}} <b>${{tag}}</b> p${{e.pid}} ${{who(e.frm)}}→${{who(e.to)}} `+
      `garr ${{e.garr}} inb us/opp ${{e.inb_us}}/${{e.inb_opp}}</div>`;
 }});
 L.innerHTML=h;
 const nowEl=L.querySelector('.now'); if(nowEl) nowEl.scrollIntoView({{block:'nearest'}});
}}
let playing=false,timer=null;
const slider=document.getElementById('slider'),btn=document.getElementById('play');
slider.oninput=()=>{{stop();draw(+slider.value);}};
function step(){{let t=+slider.value; t=t>={maxt}?0:t+1; draw(t);}}
function stop(){{playing=false;btn.textContent='▶ play';if(timer)clearInterval(timer);}}
btn.onclick=()=>{{if(playing){{stop();}}else{{playing=true;btn.textContent='⏸ pause';
  timer=setInterval(step,+document.getElementById('spd').value);}}}};
document.getElementById('spd').onchange=()=>{{if(playing){{clearInterval(timer);
  timer=setInterval(step,+document.getElementById('spd').value);}}}};
draw(0);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.json) as fh:
        R = json.load(fh)
    m = R["meta"]
    html = HTML.format(
        seed=m["seed"], n=m["n"], maxt=m["n"] - 1,
        result="WIN" if m["win"] else "LOSS",
        rcls="us" if m["win"] else "opp",
        data=json.dumps(R, separators=(",", ":")),
    )
    with open(args.out, "w") as fh:
        fh.write(html)
    print("wrote", args.out, f"({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
