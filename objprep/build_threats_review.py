#!/usr/bin/env python
"""Build a compact threats-review artifact: mapping table + the two render figures."""
import base64, io
from PIL import Image
IMG = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs"
OUT = "/Users/jonah/.claude/jobs/dab08409/tmp/threats_review.html"
def embed(p):
    im = Image.open(p).convert("RGB").quantize(colors=64, method=Image.MEDIANCUT)
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
GUNS = embed(f"{IMG}/_threats_guns.png"); KNIVES = embed(f"{IMG}/_knives_all.png")
CLOSED = embed(f"{IMG}/_knives_selected.png")

ROWS = [
 ("gun1","Glock Gen2","glock17_ext.stl (dataset4 gun)","anchor","Reused from the 3,920-scan set for cross-set comparison."),
 ("gun2","Glock Gen5","Glock 19 (glock_cgtrader)","good","Modern Glock; Gen2/Gen5 are near-identical in mesh."),
 ("gun3","Hellcat","FMK 9C1","approx","No Hellcat in pack; FMK is a compact striker 9mm — closest style."),
 ("gun4","S&W M&P9","Grand Power K100","approx","No M&P in library; full-size polymer striker, closest look. (MAC-10 was wrong — that's the B&T MP9 SMG.)"),
 ("gun5","derringer (small)","Tanfoglio GT27","weak","Pocket .25 semi-auto, NOT a 2-barrel derringer. No derringer in library."),
 ("knife1","pocket knife (folded)","folding_pocket_knife (AH)","SELECTED","Pocket Knifes A rigged FBX, posed closed + baked. → pocketknife_a/selected/"),
 ("+","switchblade","switchblade (AF)","SELECTED","closed folder, from the same rigged FBX set."),
 ("+","corkscrew / multitool","corkscrew (AI)","SELECTED","closed multitool (3504v, most detailed)."),
 ("knife2",'6" knife',"hunting_knife","SELECTED","fixed blade ~6in, from the military set."),
 ("knife3","dielectric knife","any selected + ceramic","good","Reuse a blade with material s=0.35 (dielectric) — shows blue."),
 ("knife4","carving knife","(none)","gap","No carving/kitchen blade in library. Needs sourcing."),
]
Q = {"anchor":"#2c7fb8","good":"#2f9e6f","approx":"#c98a00","weak":"#d94a3d","gap":"#d94a3d","filler":"#8a93a0","SELECTED":"#238b45"}
trs = "".join(
 f'<tr><td class=mono>{r[0]}</td><td>{r[1]}</td><td class=mono>{r[2]}</td>'
 f'<td><span class=pill style="background:{Q[r[3]]}">{r[3]}</span></td><td class=note>{r[4]}</td></tr>'
 for r in ROWS)

HTML = f"""<style>
:root{{--bg:#f4f5f7;--panel:#fff;--ink:#1b1e24;--muted:#6b7280;--line:#e2e5ea}}
@media(prefers-color-scheme:dark){{:root{{--bg:#12151a;--panel:#1a1e25;--ink:#e8ebf0;--muted:#8a93a0;--line:#2a2f38}}}}
:root[data-theme=dark]{{--bg:#12151a;--panel:#1a1e25;--ink:#e8ebf0;--muted:#8a93a0;--line:#2a2f38}}
:root[data-theme=light]{{--bg:#f4f5f7;--panel:#fff;--ink:#1b1e24;--muted:#6b7280;--line:#e2e5ea}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.45}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{max-width:1180px;margin:0 auto;padding:26px 22px 60px}}
h1{{font-size:20px;margin:0 0 4px}}h1 small{{display:block;font-weight:400;color:var(--muted);font-size:13px;margin-top:3px}}
h2{{font-size:15px;margin:30px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.note{{color:var(--muted);font-size:12.5px}}
.pill{{color:#fff;font-size:11px;padding:2px 9px;border-radius:20px;font-weight:600;text-transform:uppercase;letter-spacing:.03em}}
img{{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff;margin-top:8px}}
.flags{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:14px;font-size:13px}}
.flags b{{color:var(--ink)}}.flags li{{margin:4px 0;color:var(--muted)}}
</style>
<main>
<h1>Threat picks review<small>Closest library meshes to the spreadsheet's gun1–5 / knife1–5 · shapes only (scale set later)</small></h1>
<table><thead><tr><th>id</th><th>spreadsheet</th><th>chosen mesh</th><th>match</th><th>note</th></tr></thead><tbody>{trs}</tbody></table>
<div class=flags><b>Status — 1 gap left (was 3)</b><ul>
<li><b>✓ folded pocket knife — SOLVED for real.</b> The Pocket Knifes A FBX collection = 10 rigged folders; each posed to its closed animation frame and baked → 10 clean closed pocket knives (below). No procedural hack.</li>
<li><b>carving / kitchen knife</b> — still none (knife4). Our blades are all combat-style. Source one.</li>
<li><b>derringer</b> — still none (gun5); Tanfoglio pocket pistol is a weak stand-in.</li>
<li><i>gun4 M&P9 → Grand Power K100</i> is the closest look (no actual M&P in the pack).</li>
<li><b>Sizes are normalized</b> to compare shape; real sizes get set when we scale for seating.</li>
</ul></div>
<h2>Selected knives (4)</h2>
<p class=note>Final picks: 3 closed pocket-tools from the rigged Pocket Knifes A FBX (switchblade AF, folding knife AH, corkscrew AI — posed closed + baked) plus the hunting fixed blade from the military set. Saved in threats/knife/pocketknife_a/selected/.</p>
<img src="{CLOSED}" alt="selected knives: switchblade, folding knife, corkscrew, hunting blade">
<h2>Guns</h2><img src="{GUNS}" alt="threat gun candidates">
<h2>Knives — full library inventory</h2>
<p class=note>All 6 knife meshes (profile top / plan bottom) — knife1 now uses the folded textured folder; carving knife still missing.</p>
<img src="{KNIVES}" alt="all knife meshes in library">
</main>"""
open(OUT,"w").write(HTML)
print(f"wrote {OUT} ({len(HTML)/1e6:.1f} MB)")
