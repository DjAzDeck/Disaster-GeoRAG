import os, base64, uuid, io
from typing import List, Dict, Any


# ---------- IO helpers ----------
OVERLAY_DIR = os.path.abspath("runs/overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)

def save_png(pil_img, prefix="overlay"):
    fn = f"{prefix}_{uuid.uuid4().hex}.png"
    fp = os.path.join(OVERLAY_DIR, fn)
    pil_img.convert("RGB").save(fp, format="PNG")
    return fp

def to_data_uri(pil_img):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"<img src='data:image/png;base64,{b64}' style='max-width:100%;border:1px solid #ccc'/>"

# ---------- labeling utils ----------
LABELS = ["flood","wildfire","storm_wind","earthquake","landslide","volcano","drought","industrial"]
TAG_CANON = {
    "flood":"flood","floods":"flood","inundation":"flood",
    "wildfire":"wildfire","wildfires":"wildfire","forest fire":"wildfire","brush fire":"wildfire","bushfire":"wildfire","burn scar":"wildfire",
    "severe storms":"storm_wind","storm":"storm_wind","snowstorm":"storm_wind","blizzard":"storm_wind","cyclone":"storm_wind","hurricane":"storm_wind","typhoon":"storm_wind","wind damage":"storm_wind","cold waves":"storm_wind",
    "volcano":"volcano","eruption":"volcano","volcanic eruption":"volcano",
    "earthquake":"earthquake","seismic activity":"earthquake",
    "landslide":"landslide","mudslide":"landslide","debris flow":"landslide","rockfall":"landslide",
    "drought":"drought","water stress":"drought",
    "industrial":"industrial","industrial accident":"industrial","oil spill":"industrial","chemical leak":"industrial","refinery fire":"industrial"
}
HAZARD_KEYS = sorted(set(TAG_CANON.keys()), key=len, reverse=True)

def _norm_tag(t: str) -> str: return str(t).strip().lower()

def infer_label_from_text(text: str) -> str | None:
    t = (text or "").lower()
    for k in HAZARD_KEYS:
        if k in t: return TAG_CANON[k]
    return None

def extract_gaia_tags(meta: Dict[str, Any]) -> List[str]:
    if not isinstance(meta, dict): return []
    tags = meta.get("tag") if meta.get("tag") is not None else meta.get("tags")
    if tags is None: return []
    if isinstance(tags, str): return [x.strip() for x in tags.replace(";", ",").split(",") if x.strip()]
    if isinstance(tags, list): return [str(x) for x in tags]
    return []

def map_tags_to_label(tags: List[str]) -> str | None:
    for raw in tags:
        key = _norm_tag(raw)
        if key in TAG_CANON: return TAG_CANON[key]
    return None

def weak_label_from_meta(meta: Dict[str, Any]) -> str | None:
    lab = map_tags_to_label(extract_gaia_tags(meta))
    if lab: return lab
    for field in ("id","image_alt"):
        lab = infer_label_from_text(str(meta.get(field,"")))
        if lab: return lab
    for c in (meta.get("captions") or []):
        lab = infer_label_from_text(str(c))
        if lab: return lab
    return None

def gsd_from_meta(gaia_meta: dict) -> float|None:
    sat = (gaia_meta or {}).get("satellite")
    sen = (gaia_meta or {}).get("sensor")
    if isinstance(sat, list): sat = sat[0]
    if isinstance(sen, list): sen = sen[0]
    if str(sat).lower().startswith("sentinel-2") or "MSI" in str(sen): return 10.0
    return None

LOCALIZE_TRIGGERS = ("where","highlight","show","box","boxes","mask","outline","segment","draw","extent","area","km2","coverage")
def wants_localization(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(k in t for k in LOCALIZE_TRIGGERS)

def to_messages(history):
    out = []
    for m in (history or []):
        if isinstance(m, dict) and "role" in m:
            if "content" in m:
                c = m["content"]
                if isinstance(c, str): out.append({"role": m["role"], "content": c})
                elif isinstance(c, dict) and "text" in c: out.append({"role": m["role"], "content": c["text"]})
                elif isinstance(c, list):
                    txts = [x.get("text","") for x in c if isinstance(x, dict) and x.get("type")=="text"]
                    out.append({"role": m["role"], "content": " ".join([t for t in txts if t])})
                else: out.append({"role": m["role"], "content": str(c)})
            elif "text" in m: out.append({"role": m["role"], "content": str(m["text"])})
    return out