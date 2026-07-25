#!/usr/bin/env python
"""Build the confuser annotation gallery HTML from rendered PNGs + manifest.json.
Embeds every thumbnail as a data URI (self-contained Artifact)."""
import os, json, base64, html, io
from PIL import Image

IMGDIR = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs"
OUT = "/Users/jonah/.claude/jobs/dab08409/tmp/confuser_gallery.html"

def embed(path):
    im = Image.open(path).convert("RGB").quantize(colors=64, method=Image.MEDIANCUT)
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

REGION = {  # site -> (region label, order)
    "wb_front_mid": "front waistband", "wb_front_l": "front waistband", "wb_front_r": "front waistband",
    "flank_l": "flank", "flank_r": "flank",
    "wb_back_mid": "back waistband", "wb_back_l": "back waistband", "wb_back_r": "back waistband",
    "wb_l": "side", "wb_r": "side",
    "belly": "midline", "chest": "midline", "small_of_back": "midline", "upper_back": "midline",
    "thigh_inner_l": "thigh", "thigh_inner_r": "thigh", "thigh_outer_l": "thigh", "thigh_outer_r": "thigh",
    "ankle_l": "ankle", "ankle_r": "ankle",
    "upper_arm_l": "arm", "upper_arm_r": "arm", "forearm_l": "arm", "forearm_r": "arm",
    "shin_l": "shin", "shin_r": "shin",
}
REGION_ORDER = ["front waistband","flank","back waistband","side","midline","thigh","ankle","arm","shin"]

# reviewer's latest picks -> seed the page (status only). Source of truth = confuser_site_map.json
PRIOR = json.load(open("/Users/jonah/.claude/jobs/dab08409/tmp/confuser_site_map.json"))
REFURI = embed("/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs/_objects_reference.png")
PRELOAD = {}
for o, d in PRIOR.items():
    for st in ("ok", "no", "maybe"):
        for s in d.get(st, []):
            PRELOAD[o + "|" + s] = {"status": st}

man = json.load(open(os.path.join(IMGDIR, "manifest.json")))
# embed images
for r in man:
    r["uri"] = embed(os.path.join(IMGDIR, r["img"]))
    r["region"] = REGION.get(r["site"], "other")

objects = sorted({r["object"] for r in man})
DATA = json.dumps(man)

CSS = """
:root{
  --bg:#f4f5f7; --panel:#ffffff; --ink:#1b1e24; --muted:#6b7280; --line:#e2e5ea;
  --neutral:#9aa2ad; --accent:#c07a00; --accent-soft:#f2e6cc;
  --ok:#2f9e6f; --no:#d94a3d; --maybe:#c98a00;
  --steel:#d94a3d; --poly:#4a4a8a;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#12151a; --panel:#1a1e25; --ink:#e8ebf0; --muted:#8a93a0; --line:#2a2f38;
    --neutral:#5b636e; --accent:#e8a800; --accent-soft:#3a3218; }
}
:root[data-theme=dark]{ --bg:#12151a; --panel:#1a1e25; --ink:#e8ebf0; --muted:#8a93a0; --line:#2a2f38;
  --neutral:#5b636e; --accent:#e8a800; --accent-soft:#3a3218; }
:root[data-theme=light]{ --bg:#f4f5f7; --panel:#ffffff; --ink:#1b1e24; --muted:#6b7280; --line:#e2e5ea;
  --neutral:#9aa2ad; --accent:#c07a00; --accent-soft:#f2e6cc; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.4}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
header{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--panel) 92%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 22px;
  display:flex;align-items:center;gap:20px;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650;letter-spacing:-.01em}
h1 small{display:block;font-weight:400;color:var(--muted);font-size:11.5px;letter-spacing:.02em;margin-top:2px}
.spacer{flex:1}
.prog{font-size:12.5px;color:var(--muted)}
.prog b{color:var(--ink)}
.bar{height:5px;width:150px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:4px}
.bar i{display:block;height:100%;background:var(--accent);width:0%}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--panel);color:var(--ink);
  border-radius:7px;padding:7px 13px;font-size:12.5px}
button:hover{border-color:var(--neutral)}
.btn-accent{background:var(--accent);border-color:var(--accent);color:#1b1400;font-weight:600}
:root[data-theme=light] .btn-accent,:root:not([data-theme]) .btn-accent{color:#fff}
@media (prefers-color-scheme:light){:root:not([data-theme]) .btn-accent{color:#fff}}
main{max-width:1220px;margin:0 auto;padding:22px}
.objnav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.objnav a{text-decoration:none;font-size:12px;padding:5px 11px;border:1px solid var(--line);
  border-radius:20px;color:var(--muted)}
.objnav a:hover{color:var(--ink);border-color:var(--neutral)}
.refbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:16px;padding:6px 12px}
.refbox summary{cursor:pointer;font-size:12.5px;font-weight:600;padding:6px 2px;color:var(--ink)}
.refbox img{width:100%;max-width:100%;display:block;border-radius:6px;margin:8px 0 6px;background:#fff}
.legend{font-size:11.5px;color:var(--muted);margin:2px 0 20px;display:flex;gap:16px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:5px}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block}
section.obj{margin-bottom:34px;scroll-margin-top:82px}
.objhead{display:flex;align-items:baseline;gap:12px;border-bottom:1px solid var(--line);
  padding-bottom:7px;margin-bottom:14px}
.objhead h2{font-size:19px;margin:0;text-transform:capitalize;letter-spacing:-.01em}
.objhead .mat{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.objhead .tally{margin-left:auto;font-size:12px;color:var(--muted)}
.regionlab{font-size:10.5px;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);
  margin:16px 0 8px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(196px,1fr));gap:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;
  display:flex;flex-direction:column;transition:border-color .12s}
.card[data-st=ok]{border-color:var(--ok)}
.card[data-st=no]{border-color:var(--no);opacity:.72}
.card[data-st=maybe]{border-color:var(--maybe)}
.card img{width:100%;display:block;background:#fff}
.cbody{padding:8px 9px 10px;display:flex;flex-direction:column;gap:7px}
.site{font-size:12.5px;font-weight:600}
.seg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px}
.seg button{padding:5px 0;font-size:11.5px;border-radius:6px}
.seg button.on[data-v=ok]{background:var(--ok);border-color:var(--ok);color:#fff}
.seg button.on[data-v=no]{background:var(--no);border-color:var(--no);color:#fff}
.seg button.on[data-v=maybe]{background:var(--maybe);border-color:var(--maybe);color:#1b1400}
.note{width:100%;font:inherit;font-size:11.5px;padding:5px 7px;border:1px solid var(--line);
  border-radius:6px;background:transparent;color:var(--ink)}
.note::placeholder{color:var(--neutral)}
dialog{border:1px solid var(--line);border-radius:12px;background:var(--panel);color:var(--ink);
  max-width:640px;width:92%;padding:0}
dialog::backdrop{background:rgba(0,0,0,.45)}
.dlg-h{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
.dlg-b{padding:16px 18px}
textarea{width:100%;height:230px;font-family:ui-monospace,monospace;font-size:11.5px;
  border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);padding:10px;resize:vertical}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

JS = """
const DATA = __DATA__;
const KEY = 'confuser_annot_v2';
const PRELOAD = __PRELOAD__;                                 // round-1 picks, seeds a fresh page
let state = JSON.parse(localStorage.getItem(KEY) || 'null') || JSON.parse(JSON.stringify(PRELOAD));
const k = (o,s)=>o+'|'+s;
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); progress(); }
function set(o,s,v){
  const key=k(o,s); const cur=(state[key]||{}); cur.status=(cur.status===v?null:v);
  state[key]=cur; save();
  const card=document.querySelector(`[data-card="${key}"]`);
  card.dataset.st=cur.status||'';
  card.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on', b.dataset.v===cur.status));
  tallies();
}
function note(o,s,val){ const key=k(o,s); const cur=(state[key]||{}); cur.note=val; state[key]=cur; save(); }
function progress(){
  const tot=DATA.length; const done=DATA.filter(d=>{const c=state[k(d.object,d.site)];return c&&c.status;}).length;
  document.getElementById('done').textContent=done;
  document.getElementById('tot').textContent=tot;
  document.getElementById('barfill').style.width=(100*done/tot).toFixed(1)+'%';
}
function tallies(){
  document.querySelectorAll('section.obj').forEach(sec=>{
    const o=sec.dataset.obj;
    const rows=DATA.filter(d=>d.object===o);
    const ok=rows.filter(d=>(state[k(o,d.site)]||{}).status==='ok').length;
    sec.querySelector('.tally').innerHTML=`<span class="mono">${ok}</span> sites OK`;
  });
}
function exportJSON(){
  const out={};
  DATA.forEach(d=>{ const c=state[k(d.object,d.site)]||{};
    if(!out[d.object]) out[d.object]={ok:[],no:[],maybe:[],notes:{}};
    if(c.status) out[d.object][c.status].push(d.site);
    if(c.note) out[d.object].notes[d.site]=c.note;
  });
  const txt=JSON.stringify(out,null,2);
  document.getElementById('exp').value=txt;
  document.getElementById('dlg').showModal();
}
function copyExp(){ navigator.clipboard.writeText(document.getElementById('exp').value);
  document.getElementById('copybtn').textContent='Copied'; setTimeout(()=>document.getElementById('copybtn').textContent='Copy',1200); }
function downloadExp(){ const b=new Blob([document.getElementById('exp').value],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download='confuser_site_map.json'; a.click(); }
function toggleTheme(){ const r=document.documentElement;
  const cur=r.getAttribute('data-theme')|| (matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  r.setAttribute('data-theme', cur==='dark'?'light':'dark'); }

// build DOM
function render(){
  const REG=__REGORD__;
  const objs=[...new Set(DATA.map(d=>d.object))].sort();
  const nav=document.getElementById('nav'); const root=document.getElementById('root');
  objs.forEach(o=>{
    const a=document.createElement('a'); a.href='#s-'+o; a.textContent=o; nav.appendChild(a);
    const rows=DATA.filter(d=>d.object===o);
    const mat=rows[0].material;
    const sec=document.createElement('section'); sec.className='obj'; sec.id='s-'+o; sec.dataset.obj=o;
    sec.innerHTML=`<div class="objhead"><h2>${o}</h2><span class="mat">${mat}</span><span class="tally"></span></div>`;
    REG.forEach(rg=>{
      const rr=rows.filter(d=>d.region===rg); if(!rr.length) return;
      const lab=document.createElement('div'); lab.className='regionlab'; lab.textContent=rg; sec.appendChild(lab);
      const grid=document.createElement('div'); grid.className='grid';
      rr.forEach(d=>{
        const key=k(o,d.site); const st=(state[key]||{});
        const card=document.createElement('div'); card.className='card'; card.dataset.card=key; card.dataset.st=st.status||'';
        card.innerHTML=`<img loading="lazy" src="${d.uri}" alt="${o} at ${d.site}">
          <div class="cbody"><div class="site mono">${d.site}</div>
          <div class="seg">
            <button data-v="ok" class="${st.status==='ok'?'on':''}">OK</button>
            <button data-v="no" class="${st.status==='no'?'on':''}">No</button>
            <button data-v="maybe" class="${st.status==='maybe'?'on':''}">?</button>
          </div>
          <input class="note" placeholder="note (orientation, why not)" value="${(st.note||'').replace(/"/g,'&quot;')}"></div>`;
        card.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>set(o,d.site,b.dataset.v));
        card.querySelector('.note').oninput=(e)=>note(o,d.site,e.target.value);
        grid.appendChild(card);
      });
      sec.appendChild(grid);
    });
    root.appendChild(sec);
  });
  progress(); tallies();
}
render();
"""

body = f"""
<header>
  <h1>Confuser placement review<small>THz concealed-threat sim · which body sites each benign object may occupy</small></h1>
  <div class="spacer"></div>
  <div class="prog">Decided <b id="done">0</b> / <span id="tot">0</span><div class="bar"><i id="barfill"></i></div></div>
  <button onclick="toggleTheme()">Theme</button>
  <button class="btn-accent" onclick="exportJSON()">Export map</button>
</header>
<main>
  <details open class="refbox">
    <summary>Size &amp; orientation reference — every object at true scale vs the body</summary>
    <img src="{REFURI}" alt="objects size and orientation reference at true scale">
  </details>
  <div class="objnav" id="nav"></div>
  <div class="legend">
    <span class="chip"><span class="dot" style="background:var(--steel)"></span>metal (steel/alloy)</span>
    <span class="chip"><span class="dot" style="background:var(--poly)"></span>dielectric (polymer/plastic)</span>
    <span class="chip"><span class="dot" style="background:var(--ok)"></span>OK here</span>
    <span class="chip"><span class="dot" style="background:var(--no)"></span>not here</span>
    <span class="chip"><span class="dot" style="background:var(--maybe)"></span>maybe</span>
    <span class="chip">yellow ring = object location · zoom panel shows scale/material</span>
  </div>
  <div id="root"></div>
</main>
<dialog id="dlg">
  <div class="dlg-h"><b>Site map</b><span class="spacer" style="flex:1"></span>
    <button id="copybtn" onclick="copyExp()">Copy</button>
    <button onclick="downloadExp()">Download JSON</button>
    <button onclick="document.getElementById('dlg').close()">Close</button></div>
  <div class="dlg-b"><textarea id="exp" readonly></textarea></div>
</dialog>
"""

htmlout = ("<style>" + CSS + "</style>\n" + body +
           "<script>" + JS.replace("__DATA__", DATA)
                          .replace("__REGORD__", json.dumps(REGION_ORDER))
                          .replace("__PRELOAD__", json.dumps(PRELOAD)) + "</script>")
with open(OUT, "w") as fh:
    fh.write(htmlout)
print(f"wrote {OUT}  ({len(htmlout)/1e6:.1f} MB, {len(man)} cards)")
