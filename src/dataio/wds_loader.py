from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional, Sequence, List
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
import webdataset as wds
import json, re, math

from .metadata import EOImage

@dataclass
class WDSRecord:
    key: str
    shard: str | None
    image_pil: Image.Image
    text: str
    meta_dict: Dict[str, Any]
    eo: EOImage

def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out={}
    for k,v in d.items():
        lk = k.lower() if isinstance(k,str) else k
        out[lk] = _lower_keys(v) if isinstance(v,dict) else v
    return out

def _ensure_json(x) -> Dict[str, Any]:
    if isinstance(x, dict): return _lower_keys(x)
    if isinstance(x, bytes):
        try: return _lower_keys(json.loads(x.decode("utf-8",errors="ignore")))
        except Exception: return {"raw": x.decode("utf-8",errors="ignore")}
    if isinstance(x, str):
        try: return _lower_keys(json.loads(x))
        except Exception: return {"raw": x}
    return {"raw": str(x)}

def _first_or_none(v): return (v[0] if isinstance(v,list) and v else v)

def _norm_float(v) -> Optional[float]:
    try:
        f=float(v);  return f if math.isfinite(f) else None
    except: return None

SENSOR_HINTS = [
    (r"sentinel[- ]?2|(^|[^A-Za-z])s2([^A-Za-z]|$)", "Sentinel-2"),
    (r"sentinel[- ]?1|(^|[^A-Za-z])s1([^A-Za-z]|$)", "Sentinel-1"),
    (r"landsat[- ]?8|(^|[^A-Za-z])l8([^A-Za-z]|$)|oli", "Landsat-8"),
    (r"landsat[- ]?9|(^|[^A-Za-z])l9([^A-Za-z]|$)", "Landsat-9"),
    (r"planetscope|^ps\\b", "PlanetScope"),
    (r"worldview|wv[ -]?\\d", "WorldView"),
]

def _guess_satellite(text: str) -> Optional[str]:
    t=text.lower()
    for pat,tag in SENSOR_HINTS:
        if re.search(pat,t): return tag
    return None

def _compose_text(meta: Dict[str, Any]) -> str:
    caps = [c for c in meta.get("captions") or [] if isinstance(c,str)]
    alt= meta.get("image_alt") or ""
    loc = meta.get("location") or ""
    tags = [t for t in (meta.get("tag") or []) if isinstance(t,str)]
    sat = _first_or_none(meta.get("satellite"))
    sen = _first_or_none(meta.get("sensor"))
    mod = _first_or_none(meta.get("modalities"))
    res = meta.get("resolution") or ""
    parts=[]
    if caps: parts.append(" ".join(caps))
    if isinstance(alt,str) and alt: parts.append(alt)
    if tags: parts.append("Tags: "+", ".join(tags))
    if isinstance(loc,str) and loc: parts.append(f"Location: {loc}")
    side=[x for x in [sat,sen,mod,res] if x]
    if side: parts.append(" / ".join(side))
    return " | ".join(parts)[:1000]

def _eo_from_gaia(meta: Dict[str,Any]) -> EOImage:
    sat=_first_or_none(meta.get("satellite"))
    sen=_first_or_none(meta.get("sensor"))
    mod=_first_or_none(meta.get("modalities"))
    res=meta.get("resolution")
    loc=meta.get("location")
    lat=_norm_float(meta.get("lat")); lon=_norm_float(meta.get("lon"))
    text_hay=" ".join([str(x) for x in [
        meta.get("image_alt",""), " ".join(meta.get("captions") or []), str(meta.get("tag") or "")
    ] if x])
    if not sat: sat=_guess_satellite(text_hay)
    return EOImage(path="<wds>", sensor=sen, satellite=sat, modality=mod,
                   resolution=res, date=None, cloud=None, location=loc, lat=lat, lon=lon)

def resolve_shards(spec: str|Path) -> Sequence[str]:
    p=Path(spec)
    if p.exists() and p.is_dir():
        files=sorted(str(x) for x in p.glob("*.tar"))
        if not files: raise FileNotFoundError(f"No .tar files in {p}")
        return files
    return [str(spec)]

def _basename_from_url(u: str) -> str:
    try:
        parsed = urlparse(u)
        path = parsed.path if parsed.scheme else u
        return Path(path).name
    except Exception:
        return str(u)

def make_wds(shards_or_dir: str, shardshuffle: bool=False, keep_key: bool=False):
    shards = resolve_shards(shards_or_dir)
    ds = wds.WebDataset(shards, shardshuffle=shardshuffle).decode("pil")
    # keep both __key__ and __url__ so we know the source shard per sample
    # (__key__/__url__ are supported special fields in WebDataset)
    if keep_key:
        return ds.to_tuple("__key__", "__url__", "png;jpg;jpeg", "txt", "json")
    else:
        return ds.to_tuple("png;jpg;jpeg", "txt", "json")

def iter_wds(shards_or_dir: str, limit: Optional[int]=None) -> Iterator[WDSRecord]:
    cnt=0
    for key, url, img, txt, meta in make_wds(shards_or_dir, keep_key=True):
        if not isinstance(img, Image.Image): img = Image.fromarray(img).convert("RGB")
        else: img = img.convert("RGB")
        md = _ensure_json(meta)
        text = _compose_text(md)
        eo = _eo_from_gaia(md)
        yield WDSRecord(key=key, shard=_basename_from_url(url), image_pil=img, text=text, meta_dict=md, eo=eo)
        cnt+=1
        if limit is not None and cnt>=limit: break

def build_catalog(shards_or_dir: str, limit: Optional[int]=None) -> List[Dict[str,Any]]:
    cat=[]
    for i,rec in enumerate(iter_wds(shards_or_dir, limit=limit)):
        md=rec.meta_dict
        cat.append({
            "idx": i,
            "shard": rec.shard or "",
            "key": rec.key,
            "id": md.get("id") or rec.key,
            "title": (md.get("image_alt") or (md.get("captions") or [""])[0] or ""),
            "location": md.get("location") or "",
            "lat": md.get("lat"), "lon": md.get("lon"),
            "tags": ", ".join(md.get("tag") or []),
            "satellite": _first_or_none(md.get("satellite")) or "",
            "sensor": _first_or_none(md.get("sensor")) or "",
            "modalities": _first_or_none(md.get("modalities")) or "",
            "resolution": md.get("resolution") or "",
        })
    return cat

def get_sample_by_index(shards_or_dir: str, index: int) -> WDSRecord:
    for i,rec in enumerate(iter_wds(shards_or_dir, limit=None)):
        if i==index: return rec
    raise IndexError(f"Index {index} out of range")

def list_shards(dir_path: str) -> List[Dict[str,Any]]:
    """
    Return [{'shard': name, 'size_mb': float, 'path': '/abs/path/to.tar'}, ...]
    for all .tar files in a directory. If dir doesn't exist or is empty, return [].
    """
    p = Path(dir_path)
    if not (p.exists() and p.is_dir()):
        return []
    rows=[]
    for f in sorted(p.glob("*.tar")):
        try:
            rows.append({
                "shard": f.name,
                "size_mb": round(f.stat().st_size/1e6, 1),
                "path": str(f.resolve())
            })
        except Exception:
            rows.append({"shard": f.name, "size_mb": None, "path": str(f)})
    return rows
