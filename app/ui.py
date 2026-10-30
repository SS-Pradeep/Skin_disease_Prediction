"""Look and feel of the DermaScan app: page CSS, HTML pieces and image helpers.

app.py holds the page flow and calls these functions; nothing here touches the model.
Colours, fonts and corner radius for Streamlit's own widgets are in .streamlit/config.toml.
"""
import base64
import html
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

RISK = {"ACK": ("Pre-cancerous", "amber"), "BCC": ("Skin cancer", "red"), "MEL": ("Skin cancer", "red"),
        "SCC": ("Skin cancer", "red"), "NEV": ("Benign", "green"), "SEK": ("Benign", "green")}
SYMPTOM_LABEL = {"itch": "Itch", "grew": "Grew", "hurt": "Hurt", "changed": "Changed", "bleed": "Bleed",
                 "elevation": "Raised"}
ANSWER_TEXT = {"TRUE": "Yes", "FALSE": "No", "UNK": "Unknown"}
ANSWER_CODE = {v: k for k, v in ANSWER_TEXT.items()}
ESC = html.escape

CSS = """
<style>
@property --ds-p { syntax: '\\3c number\\3e'; inherits: false; initial-value: 0; }
:root {
  --ds-bg:#0A1014; --ds-card:#0F181D; --ds-card-2:#142027; --ds-track:#1A2630;
  --ds-line:#1E2C34; --ds-line-2:#2A3B45;
  --ds-text:#E6EEF2; --ds-soft:#C5D2D8; --ds-muted:#8FA1AB; --ds-faint:#62757F;
  --ds-teal:#2DD4BF; --ds-teal-2:#14B8A6; --ds-teal-ink:#042F2E; --ds-teal-bg:rgba(45,212,191,.12);
  --ds-red:#F87171; --ds-red-bg:rgba(248,113,113,.13);
  --ds-amber:#FBBF24; --ds-amber-bg:rgba(251,191,36,.13);
  --ds-green:#4ADE80; --ds-green-bg:rgba(74,222,128,.12);
}
.stApp { background: radial-gradient(1100px 420px at 50% -160px, rgba(45,212,191,.10), transparent 70%), var(--ds-bg); }
[data-testid="stHeader"] { display:none; }
[data-testid="stMainBlockContainer"] { max-width:1260px; padding:26px 32px 40px; }

/* cards: st.container(key="card_...") */
[class*="st-key-card_"] { background:var(--ds-card); border:1px solid var(--ds-line); border-radius:18px;
  padding:20px 22px; box-shadow:0 24px 48px -30px rgba(0,0,0,.8); }
.ds-ch { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.ds-h { font-size:11.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--ds-muted); }
.ds-aside { font-size:12px; color:var(--ds-faint); }
.ds-sub { font-size:13px; color:var(--ds-muted); line-height:1.55; margin-top:4px; }

/* top bar + stepper */
.ds-top { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }
.ds-logo { display:flex; align-items:center; gap:12px; }
.ds-mark { width:40px; height:40px; border-radius:12px; display:grid; place-items:center;
  background:linear-gradient(135deg,#2DD4BF,#0E7490); box-shadow:0 10px 24px -10px rgba(45,212,191,.7); }
.ds-logo b { display:block; font-size:17px; font-weight:800; letter-spacing:-.01em; color:var(--ds-text); }
.ds-logo small { display:block; font-size:12px; color:var(--ds-muted); }
.ds-steps { display:flex; gap:4px; background:var(--ds-card); padding:5px; border-radius:999px; border:1px solid var(--ds-line); }
.ds-step { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; color:var(--ds-faint);
  padding:7px 16px; border-radius:999px; }
.ds-step i { font-style:normal; width:20px; height:20px; border-radius:50%; display:grid; place-items:center;
  font-size:11px; font-weight:700; background:var(--ds-track); color:var(--ds-muted); }
.ds-step.done { color:var(--ds-soft); }
.ds-step.done i { background:var(--ds-teal-bg); color:var(--ds-teal); }
.ds-step.on { background:var(--ds-teal-2); color:var(--ds-teal-ink); }
.ds-step.on i { background:rgba(4,47,46,.2); color:var(--ds-teal-ink); }
.ds-step.busy i { animation:ds-pulse 1s ease-in-out infinite; }
.ds-pill { font-size:12px; font-weight:700; color:var(--ds-amber); background:var(--ds-amber-bg); padding:6px 12px; border-radius:999px; }
.ds-title { margin:18px 0 2px; }
.ds-title h1 { font-size:30px; font-weight:800; letter-spacing:-.02em; margin:0; padding:0; line-height:1.2; color:var(--ds-text); }
.ds-title p { margin:6px 0 0; color:var(--ds-muted); font-size:15px; }

/* buttons, uploader, pills, segmented controls */
[data-testid="stBaseButton-primary"] { background:var(--ds-teal-2); border-color:var(--ds-teal-2); color:var(--ds-teal-ink);
  font-weight:700; box-shadow:0 10px 24px -12px rgba(45,212,191,.75); }
[data-testid="stBaseButton-primary"]:hover { background:var(--ds-teal); border-color:var(--ds-teal); color:var(--ds-teal-ink); }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p { font-weight:700; }
[data-testid="stBaseButton-secondary"] { background:var(--ds-card-2); border-color:var(--ds-line-2); }
[data-testid="stFileUploaderDropzone"] { background:var(--ds-card-2); border:1.5px dashed var(--ds-line-2);
  border-radius:14px; padding:26px 18px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--ds-teal-2); }
[data-testid="stButtonGroup"] button[data-variant="pills"] { flex:0 0 auto; background:var(--ds-card-2); border-color:var(--ds-line); color:var(--ds-muted); }
[data-testid="stButtonGroup"] button[data-variant="pills"][data-selected="true"] { background:var(--ds-teal-bg); border-color:var(--ds-teal-2); color:var(--ds-teal); }
[data-testid="stButtonGroup"] button[data-variant="pills"][data-selected="true"] p { font-weight:700; }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] { background:var(--ds-card); color:var(--ds-muted); }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][data-selected="true"] { background:#25363F; border-color:#3A4F5A; color:var(--ds-text); }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][data-selected="true"] p { font-weight:700; }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:first-child[data-selected="true"] {
  background:var(--ds-teal-2); border-color:var(--ds-teal-2); color:var(--ds-teal-ink); }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:nth-child(3)[data-selected="true"] {
  background:var(--ds-amber-bg); border-color:rgba(251,191,36,.55); color:var(--ds-amber); }
[data-testid="stExpander"] details { background:var(--ds-card); border:1px solid var(--ds-line); border-radius:14px; }
[data-testid="stExpander"] summary p { font-weight:700; }

/* step 1: upload tips, selected photo, example tiles */
.ds-tips { display:grid; gap:8px; margin-top:2px; font-size:13px; color:var(--ds-muted); }
.ds-tips div { display:flex; gap:10px; align-items:center; }
.ds-tips div::before { content:""; width:6px; height:6px; border-radius:50%; background:var(--ds-teal-2); flex:none; }
.ds-sel { display:flex; gap:14px; align-items:center; padding:12px; border-radius:14px; background:var(--ds-card-2); border:1px solid var(--ds-line); }
.ds-sel img { width:64px; height:64px; border-radius:10px; object-fit:cover; }
.ds-sel b { display:block; font-size:14px; color:var(--ds-text); }
.ds-sel span { font-size:12.5px; color:var(--ds-muted); }
.ds-ok { margin-left:auto; font-size:12px; font-weight:700; color:var(--ds-teal); background:var(--ds-teal-bg); padding:4px 10px; border-radius:999px; }
[class*="st-key-tile_"] { position:relative; gap:0; border-radius:14px; border:1px solid var(--ds-line); background:var(--ds-card-2);
  overflow:hidden; transition:border-color .15s, transform .15s, box-shadow .15s; }
[class*="st-key-tile_"]:hover { border-color:var(--ds-teal-2); transform:translateY(-2px); }
[class*="st-key-tile_"][class*="_sel"] { border-color:var(--ds-teal-2); box-shadow:0 0 0 3px rgba(45,212,191,.2); }
[class*="st-key-tile_"] [data-testid="stElementContainer"]:has(button) { position:absolute; inset:0; z-index:3; margin:0; }
[class*="st-key-tile_"] [data-testid="stElementContainer"]:has(button) div,
[class*="st-key-tile_"] [data-testid="stElementContainer"]:has(button) button { width:100%; height:100%; }
[class*="st-key-tile_"] [data-testid="stElementContainer"]:has(button) button { opacity:0; cursor:pointer; }
.ds-tile img { display:block; width:100%; aspect-ratio:1; object-fit:cover; }
.ds-tile-cap { padding:9px 12px 11px; }
.ds-tile-cap b { display:block; font-size:13px; color:var(--ds-text); }
.ds-tile-cap span { font-size:12px; color:var(--ds-muted); }
.ds-tick { position:absolute; top:8px; right:8px; width:24px; height:24px; border-radius:50%; background:var(--ds-teal-2);
  color:var(--ds-teal-ink); display:none; place-items:center; font-size:13px; font-weight:800; }
[class*="st-key-tile_"][class*="_sel"] .ds-tick { display:grid; }

/* step 2: photo, symptom rows */
.ds-photo img { display:block; width:100%; aspect-ratio:1; object-fit:cover; border-radius:14px; border:1px solid var(--ds-line); }
.ds-photo span { display:block; font-size:12.5px; color:var(--ds-muted); margin-top:8px; }
[class*="st-key-sym_"] { background:var(--ds-card-2); border:1px solid var(--ds-line); border-radius:12px; padding:10px 12px; }
[class*="st-key-sym_"] [data-testid="stElementContainer"]:has(.ds-q) { flex:1 1 0; min-width:0; width:auto; }
.ds-q b { display:block; font-size:14px; color:var(--ds-text); }
.ds-q span { font-size:12.5px; color:var(--ds-muted); }
.ds-err { font-size:13px; color:var(--ds-red); background:var(--ds-red-bg); border-radius:10px; padding:8px 12px; }

/* analysing screen */
.ds-scan { position:relative; aspect-ratio:1; width:100%; max-width:360px; margin:0 auto; border-radius:16px; overflow:hidden; border:1px solid var(--ds-line-2); }
.ds-scan img { width:100%; height:100%; object-fit:cover; display:block; }
.ds-scan::before { content:""; position:absolute; inset:0; z-index:1;
  background-image:linear-gradient(rgba(45,212,191,.10) 1px,transparent 1px),linear-gradient(90deg,rgba(45,212,191,.10) 1px,transparent 1px);
  background-size:24px 24px; }
.ds-scan-line { position:absolute; left:0; right:0; height:80px; z-index:2;
  background:linear-gradient(180deg,transparent,rgba(45,212,191,.25) 82%,rgba(94,234,212,.95) 100%);
  animation:ds-scan 1.9s cubic-bezier(.45,0,.55,1) infinite; }
.ds-br { position:absolute; width:26px; height:26px; border-color:var(--ds-teal); border-style:solid; z-index:3; }
.ds-br.a { top:12px; left:12px; border-width:3px 0 0 3px; border-top-left-radius:8px; }
.ds-br.b { top:12px; right:12px; border-width:3px 3px 0 0; border-top-right-radius:8px; }
.ds-br.c { bottom:12px; left:12px; border-width:0 0 3px 3px; border-bottom-left-radius:8px; }
.ds-br.d { bottom:12px; right:12px; border-width:0 3px 3px 0; border-bottom-right-radius:8px; }
.ds-scan-tag { position:absolute; left:50%; bottom:16px; transform:translateX(-50%); z-index:3; font-size:12px; font-weight:700;
  color:var(--ds-teal-ink); background:var(--ds-teal); padding:5px 12px; border-radius:999px; animation:ds-pulse 1.2s ease-in-out infinite; }
.ds-checks { display:grid; gap:10px; }
.ds-check { display:flex; align-items:center; gap:12px; padding:12px 14px; border-radius:12px; border:1px solid var(--ds-line);
  background:var(--ds-card-2); font-size:14px; color:var(--ds-faint); }
.ds-check i { flex:none; width:22px; height:22px; border-radius:50%; border:2px solid var(--ds-line-2); display:grid; place-items:center;
  font-style:normal; font-size:12px; font-weight:800; box-sizing:border-box; }
.ds-check small { margin-left:auto; font-size:12px; color:var(--ds-faint); font-variant-numeric:tabular-nums; }
.ds-check.on { color:var(--ds-text); border-color:var(--ds-teal-2); }
.ds-check.on i { border-color:var(--ds-teal); border-top-color:transparent; animation:ds-spin .8s linear infinite; }
.ds-check.done { color:var(--ds-soft); }
.ds-check.done i { background:var(--ds-teal-2); border-color:var(--ds-teal-2); color:var(--ds-teal-ink); }

/* results: heat-map slider */
.ds-cmp { position:relative; width:100%; aspect-ratio:1; border-radius:14px; overflow:hidden; --x:50%; user-select:none; border:1px solid var(--ds-line); }
.ds-cmp > img, .ds-cmp .top img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; pointer-events:none; }
.ds-cmp .top { position:absolute; inset:0; clip-path:inset(0 0 0 var(--x)); }
.ds-cmp .line { position:absolute; top:0; bottom:0; left:var(--x); width:2px; margin-left:-1px; background:#fff;
  box-shadow:0 0 12px rgba(0,0,0,.5); pointer-events:none; z-index:2; }
.ds-cmp .knob { position:absolute; top:50%; left:50%; width:36px; height:36px; margin:-18px 0 0 -18px; border-radius:50%; background:#fff;
  display:grid; place-items:center; box-shadow:0 4px 14px rgba(0,0,0,.45); }
.ds-cmp .knob img { width:18px; height:18px; }
.ds-cmp .tag { position:absolute; top:10px; z-index:2; font-size:11.5px; font-weight:700; padding:4px 9px; border-radius:8px;
  background:rgba(10,16,20,.75); color:#fff; pointer-events:none; }
.ds-cmp .tag.l { left:10px; } .ds-cmp .tag.r { right:10px; }
.ds-cmp input { position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0; cursor:ew-resize; z-index:5; }
.ds-tip { font-size:12.5px; color:var(--ds-soft); background:var(--ds-teal-bg); border-radius:10px; padding:9px 12px; line-height:1.55; }

/* results: prediction card */
.ds-res { display:flex; gap:18px; align-items:center; }
.ds-ring { flex:none; width:120px; height:120px; border-radius:50%; display:grid; place-items:center; position:relative;
  background:conic-gradient(var(--ds-teal) calc(var(--ds-p) * 1%), var(--ds-track) 0); animation:ds-ring 1.1s cubic-bezier(.2,.8,.2,1); }
.ds-ring::before { content:""; position:absolute; inset:11px; background:var(--ds-card); border-radius:50%; }
.ds-ring div { position:relative; text-align:center; line-height:1; }
.ds-ring b { font-size:27px; font-weight:800; color:var(--ds-text); }
.ds-ring span { display:block; font-size:10.5px; color:var(--ds-muted); font-weight:600; margin-top:4px; }
.ds-code { font-size:12px; font-weight:700; color:var(--ds-teal); letter-spacing:.08em; }
.ds-name { font-size:21px; font-weight:800; line-height:1.2; margin:2px 0 10px; color:var(--ds-text); }
.ds-badge { display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:700; padding:5px 11px; border-radius:999px; }
.ds-badge::before { content:""; width:7px; height:7px; border-radius:50%; background:currentColor; }
.ds-badge.red { background:var(--ds-red-bg); color:var(--ds-red); }
.ds-badge.amber { background:var(--ds-amber-bg); color:var(--ds-amber); }
.ds-badge.green { background:var(--ds-green-bg); color:var(--ds-green); }
.ds-truth { margin-top:16px; font-size:12.5px; padding:9px 12px; border-radius:10px; background:var(--ds-card-2);
  border:1px solid var(--ds-line); color:var(--ds-muted); }
.ds-truth b { color:var(--ds-text); }
.ds-truth em { font-style:normal; font-weight:700; color:var(--ds-teal); }
.ds-truth em.no { color:var(--ds-amber); }
.ds-bars { display:grid; gap:10px; margin-top:18px; }
.ds-bar .l { display:flex; justify-content:space-between; font-size:12.5px; color:var(--ds-muted); margin-bottom:5px; }
.ds-bar .l b { color:var(--ds-soft); font-weight:700; margin-right:6px; }
.ds-bar .l span:last-child { font-variant-numeric:tabular-nums; }
.ds-bar p { margin:0; height:7px; border-radius:99px; background:var(--ds-track); overflow:hidden; }
.ds-bar p i { display:block; height:100%; border-radius:99px; background:#2B6F68; animation:ds-grow .9s cubic-bezier(.2,.8,.2,1); }
.ds-bar.top .l, .ds-bar.top .l b { color:var(--ds-text); }
.ds-bar.top p i { background:linear-gradient(90deg,var(--ds-teal-2),var(--ds-teal)); }

/* results: patient card */
.ds-chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.ds-chips span { font-size:12.5px; font-weight:600; background:var(--ds-card-2); border:1px solid var(--ds-line); border-radius:9px;
  padding:5px 11px; color:var(--ds-soft); }
.ds-syms { display:grid; gap:11px; margin-top:12px; }
.ds-sym { display:grid; grid-template-columns:116px 1fr 42px; gap:10px; align-items:center; font-size:12.5px; }
.ds-ans { border-radius:8px; padding:4px 8px; font-weight:700; text-align:center; font-size:12px; white-space:nowrap; }
.ds-ans.y { background:var(--ds-teal-bg); color:var(--ds-teal); }
.ds-ans.n { background:var(--ds-track); color:var(--ds-muted); }
.ds-ans.u { background:var(--ds-amber-bg); color:var(--ds-amber); }
.ds-eff { height:6px; background:var(--ds-track); border-radius:99px; overflow:hidden; }
.ds-eff i { display:block; height:100%; background:var(--ds-teal-2); border-radius:99px; animation:ds-grow .9s cubic-bezier(.2,.8,.2,1); }
.ds-eff i.neg { background:var(--ds-red); }
.ds-sym b { text-align:right; font-variant-numeric:tabular-nums; color:var(--ds-text); }
.ds-note { font-size:12px; color:var(--ds-faint); line-height:1.55; margin-top:14px; }
.ds-gap { height:14px; }

.ds-foot { margin-top:30px; padding-top:16px; border-top:1px solid var(--ds-line); display:flex; justify-content:space-between;
  gap:8px 24px; flex-wrap:wrap; font-size:12.5px; color:var(--ds-faint); }

@keyframes ds-grow { from { width:0; } }
@keyframes ds-ring { from { --ds-p:0; } }
@keyframes ds-spin { to { transform:rotate(360deg); } }
@keyframes ds-pulse { 50% { opacity:.55; } }
@keyframes ds-scan { from { top:-80px; } to { top:100%; } }
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] { padding:18px 16px 32px; }
  .ds-steps { order:3; width:100%; justify-content:space-between; }
  .ds-step { padding:7px 10px; }
}
</style>
"""

# While a script run is in progress Streamlit keeps the previous page's elements on screen, faded ("stale").
HIDE_PREVIOUS_PAGE = ('<style>[data-stale="true"], '
                      '[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]:not(:has([data-stale="false"])), '
                      '[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"]:not(:has([data-stale="false"])) '
                      '{ display:none !important; }</style>')


def _svg_uri(svg: str) -> str:
    """st.html drops inline <svg>, so icons are embedded as small SVG images instead."""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


LOGO = _svg_uri('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" '
                'stroke="#042F2E" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5a1 1 '
                '0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/>'
                '<circle cx="12" cy="12" r="3.2"/></svg>')
ARROWS = _svg_uri('<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" '
                  'stroke="#0A1014" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">'
                  '<path d="M9 6l-6 6 6 6M15 6l6 6-6 6"/></svg>')


# ------------------------------------------------------------------ images
def read_rgb(data: bytes, pipeline: str) -> np.ndarray:
    """File bytes -> full-size RGB array, decoded the same way the model's pipeline decodes it."""
    if pipeline == "ck":                       # skin_core.decode_image: OpenCV, no EXIF rotation
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if bgr is None:
            raise ValueError("Could not read this file as an image.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:                                       # multimodal pipeline: PIL with the camera rotation applied
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception as err:
        raise ValueError("Could not read this file as an image.") from err


def square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size without cropping - the same framing the model gets."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def thumbnail(img: np.ndarray, size: int) -> np.ndarray:
    """Centre-cropped square thumbnail (for display only)."""
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)


def to_b64(img_rgb: np.ndarray, quality: int = 90) -> str:
    """RGB array -> JPEG data URI for <img src=...>."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ------------------------------------------------------------------ page chrome
def top_bar(step) -> str:
    """Logo, 3-step progress indicator and prototype label. step: 1, 2, 'scan' or 3."""
    pos = {1: 1, 2: 2, "scan": 3, 3: 3}[step]
    items = []
    for n, label in enumerate(["Photo", "Patient", "Results"], start=1):
        if n < pos:
            cls, mark = "done", "&#10003;"
        elif n == pos:
            cls, mark = "on" + (" busy" if step == "scan" else ""), str(n)
        else:
            cls, mark = "", str(n)
        items.append(f'<div class="ds-step {cls}"><i>{mark}</i>{label}</div>')
    return (f'<div class="ds-top"><div class="ds-logo"><span class="ds-mark"><img src="{LOGO}" alt=""></span><div><b>DermaScan</b>'
            f'<small>Explainable skin-lesion classifier</small></div></div><div class="ds-steps">{"".join(items)}</div>'
            f'<div class="ds-pill">Research prototype</div></div>')


def page_title(title: str, subtitle: str) -> str:
    return f'<div class="ds-title"><h1>{ESC(title)}</h1><p>{ESC(subtitle)}</p></div>'


def card_head(title: str, aside: str = "") -> str:
    side = f'<span class="ds-aside">{ESC(aside)}</span>' if aside else ""
    return f'<div class="ds-ch"><span class="ds-h">{ESC(title)}</span>{side}</div>'


def footer(metrics) -> str:
    right = ""
    if metrics:
        right = (f"Feature-fusion model · test set (unseen patients): accuracy {metrics['Accuracy']:.1%} · "
                 f"macro-F1 {metrics['Macro F1']:.2f} · macro AUC {metrics['Macro AUC']:.2f}")
    return (f'<div class="ds-foot"><span>Student research prototype trained on PAD-UFES-20. Not a medical device: '
            f'always consult a dermatologist.</span><span>{right}</span></div>')


# ------------------------------------------------------------------ step 1: photo
def tips() -> str:
    rows = ["Close-up, with the lesion in the centre", "Good light and in focus", "PNG or JPG"]
    return '<div class="ds-tips">' + "".join(f"<div>{r}</div>" for r in rows) + "</div>"


def selected_photo(thumb_uri: str, name: str, detail: str) -> str:
    return (f'<div class="ds-sel"><img src="{thumb_uri}" alt=""><div><b>{ESC(name)}</b><span>{ESC(detail)}</span></div>'
            f'<span class="ds-ok">Ready</span></div>')


def example_tile(thumb_uri: str, number: int, region: str, age: int) -> str:
    return (f'<div class="ds-tile"><img src="{thumb_uri}" alt="Example {number}"><span class="ds-tick">&#10003;</span>'
            f'<div class="ds-tile-cap"><b>Example {number}</b><span>{ESC(region.title())} · {age} yrs</span></div></div>')


# ------------------------------------------------------------------ step 2: patient details
def photo_card(uri: str, detail: str) -> str:
    return f'<div class="ds-photo"><img src="{uri}" alt="Selected photo"><span>{ESC(detail)}</span></div>'


def question(label: str, text: str) -> str:
    return f'<div class="ds-q"><b>{ESC(label)}</b><span>{ESC(text)}</span></div>'


def error(text: str) -> str:
    return f'<div class="ds-err">{ESC(text)}</div>'


# ------------------------------------------------------------------ analysing screen
def scan(uri: str) -> str:
    return (f'<div class="ds-scan"><img src="{uri}" alt="Photo being analysed"><div class="ds-scan-line"></div>'
            '<i class="ds-br a"></i><i class="ds-br b"></i><i class="ds-br c"></i><i class="ds-br d"></i>'
            '<span class="ds-scan-tag">Analysing</span></div>')


def checklist(steps, current: int, seconds) -> str:
    """steps before `current` are done (with their time), `current` is running, the rest are waiting."""
    rows = []
    for i, label in enumerate(steps):
        if i < current:
            rows.append(f'<div class="ds-check done"><i>&#10003;</i>{label}<small>{seconds[i]:.2f} s</small></div>')
        elif i == current:
            rows.append(f'<div class="ds-check on"><i></i>{label}</div>')
        else:
            rows.append(f'<div class="ds-check"><i></i>{label}</div>')
    return '<div class="ds-checks">' + "".join(rows) + "</div>"


# ------------------------------------------------------------------ results
def compare_slider(photo_uri: str, heat_uri: str, label: str, uid: str) -> str:
    """Before/after slider: drag to reveal the Grad-CAM over the photo. Needs unsafe_allow_javascript=True."""
    return (f'<div class="ds-cmp" id="cmp-{uid}"><img src="{photo_uri}" alt="Photo">'
            f'<div class="top"><img src="{heat_uri}" alt="Grad-CAM heat-map"></div>'
            f'<div class="line"><span class="knob"><img src="{ARROWS}" alt=""></span></div>'
            f'<span class="tag l">Photo</span><span class="tag r">Grad-CAM · {ESC(label)}</span>'
            f'<input type="range" min="0" max="100" value="50" step="0.5" aria-label="Drag to compare photo and heat-map">'
            f'</div><script>(function(){{var r=document.getElementById("cmp-{uid}");if(!r)return;'
            f'var i=r.querySelector("input");var s=function(){{r.style.setProperty("--x",i.value+"%");}};'
            f'i.addEventListener("input",s);s();}})();</script>')


def evidence_tip(code: str, predicted: bool) -> str:
    which = "the predicted class" if predicted else "this class"
    return (f'<div class="ds-tip">Drag across the photo. Red and yellow areas raised the score for <b>{ESC(code)}</b> '
            f'({which}); blue areas had little influence. A trustworthy result focuses on the lesion, not on hair, '
            f'pen marks or the photo edges.</div>')


def result_card(names, full, probs, top: int, truth=None) -> str:
    code = names[top]
    risk, colour = RISK.get(code, ("-", "green"))
    head = (f'<div class="ds-res"><div class="ds-ring" style="--ds-p:{probs[top] * 100:.1f}"><div>'
            f'<b>{probs[top]:.0%}</b><span>confidence</span></div></div><div><div class="ds-code">{code}</div>'
            f'<div class="ds-name">{ESC(full[code])}</div><span class="ds-badge {colour}">{risk}</span></div></div>')
    check = ""
    if truth:
        verdict = '<em>matches</em>' if truth == code else '<em class="no">differs from the prediction</em>'
        check = f'<div class="ds-truth">Dataset label for this example: <b>{truth}</b> · {verdict}</div>'
    bars = []
    for k in np.argsort(probs)[::-1]:
        cls = "ds-bar top" if k == top else "ds-bar"
        bars.append(f'<div class="{cls}"><div class="l"><span><b>{names[k]}</b>{ESC(full[names[k]])}</span>'
                    f'<span>{probs[k]:.1%}</span></div><p><i style="width:{probs[k] * 100:.1f}%"></i></p></div>')
    return head + check + '<div class="ds-bars">' + "".join(bars) + "</div>"


def patient_card(age: int, region: str, effects, code: str) -> str:
    chips = f'<div class="ds-chips"><span>Age {age}</span><span>{ESC(region.title())}</span></div>'
    biggest = max([abs(e["effect"]) for e in effects] + [1e-9])
    rows = []
    for e in sorted(effects, key=lambda e: e["effect"], reverse=True):
        ans = e["answer"]
        cls = {"TRUE": "y", "FALSE": "n", "UNK": "u"}[ans]
        pts = e["effect"] * 100
        width = abs(e["effect"]) / biggest * 100
        rows.append(f'<div class="ds-sym"><span class="ds-ans {cls}">{SYMPTOM_LABEL[e["symptom"]]} · {ANSWER_TEXT[ans]}</span>'
                    f'<div class="ds-eff"><i class="{"neg" if pts < 0 else ""}" style="width:{width:.1f}%"></i></div>'
                    f'<b>{pts:+.1f}</b></div>')
    note = (f'<div class="ds-note">Change in the probability of {code}, in percentage points, compared with the '
            f'opposite answer (No and Unknown are compared with Yes). Photo and other answers stay the same.</div>')
    return (chips + '<div class="ds-gap"></div>' + card_head(f"How each answer moved {code}") +
            '<div class="ds-syms">' + "".join(rows) + "</div>" + note)
