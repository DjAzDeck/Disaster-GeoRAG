import json, faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

class DisasterKB:
    def __init__(self, index_dir="geokb/.index", model="sentence-transformers/all-MiniLM-L6-v2"):
        self.index = faiss.read_index(f"{index_dir}/kb.index")
        self.items = json.load(open(f"{index_dir}/kb_items.json","r",encoding="utf-8"))
        self.enc = SentenceTransformer(model)
    def query(self, text, k=3):
        q = self.enc.encode([text], normalize_embeddings=True).astype('float32')
        D, I = self.index.search(q, k)
        out=[]
        for d,i in zip(D[0], I[0]):
            item = self.items[i]
            out.append({"id": item["id"], "type": item["type"], "cue": item["cue"], "score": float(d)})
        return out

def compose_retrieval_query_gaia(text: str, meta: Dict[str, Any], taxonomy_map: Dict[str,str]) -> str:
    """
    Build a retrieval string using GAIA fields + disaster hints.
    taxonomy_map maps GAIA tag -> our taxonomy label (e.g., "Floods"->"flood").
    """
    parts = [text]
    for k in ("location", "resolution"):
        v = meta.get(k)
        if isinstance(v, str) and v: parts.append(f"{k}: {v}")
    for k in ("satellite", "sensor", "modalities", "tag"):
        v = meta.get(k)
        if isinstance(v, list) and v:
            parts.append(f"{k}: " + ", ".join([str(x) for x in v]))
        elif isinstance(v, str) and v:
            parts.append(f"{k}: {v}")
    lat, lon = meta.get("lat"), meta.get("lon")
    if isinstance(lat,(int,float)) and isinstance(lon,(int,float)):
        parts.append(f"lat:{lat:.3f} lon:{lon:.3f}")
    hints = []
    for t in meta.get("tag", []) or []:
        label = taxonomy_map.get(str(t), None)
        if label:
            hints.extend([label]*2)
    if hints:
        parts.append("hints: " + " ".join(hints))
    return " | ".join(parts)[:1200]

def calibrate_confidence(
    js: dict,
    snips: list,
    meta: dict,
    taxonomy_map: dict | None = None,
    *,
    w_model: float = 0.5,
    w_support: float = 0.4,
    tag_boost: float = 0.2,
    contradict_penalty: float = 0.15,
) -> dict:
    """
    Post-hoc confidence calibration for EO disaster triage.

    Combines:
    - model-reported confidence (from the VLM JSON),
    - retrieval support: similarity of KB cues matching the predicted type,
    - GAIA tag priors: when dataset tags align/contradict the predicted type.

    Returns a *new* JSON dict with 'confidence' in [0,1].
    # Inspired from: 
    - https://arxiv.org/pdf/1706.04599
    - https://openreview.net/pdf?id=nNQmZGjEVe


    Notes / rationale:
      • Post-hoc calibration (like temperature scaling) is widely used to align
        probabilities with observed correctness; here we use an interpretable
        heuristic because we lack per-class logits and labeled val data.
      • Reliability diagrams are recommended to verify the effect offline.
      • Retrieval-weighting ideas in RAG indicate evidence-strength should inform trust.
    """
    import copy
    out = copy.deepcopy(js)

    pred_is_disaster = out.get("is_disaster")
    pred_type = out.get("type")
    try:
        base = float(out.get("confidence", 0.0) or 0.0)
    except Exception:
        base = 0.0
    base = max(0.0, min(1.0, base))

    if isinstance(pred_is_disaster, str):
        pred_is_disaster = pred_is_disaster.lower()
    if pred_is_disaster in (False, "false"):
        out["confidence"] = min(base, 0.10)
        return out
    if pred_is_disaster == "uncertain" or pred_type in (None, "", "none", "null"):
        out["confidence"] = min(base, 0.30)
        return out

    # Retrieval support for the predicted type.
    #    FAISS is cosine/IP on normalized embeddings -> scores in [-1,1].
    #    Map to [0,1] and average across matching snippets.
    matches = [s.get("score", 0.0) for s in snips if s.get("type") == pred_type]
    if matches:
        # rescale cosine/IP from [-1,1] -> [0,1]
        support = sum((m + 1.0) * 0.5 for m in matches) / len(matches)
    else:
        support = 0.0

    # GAIA tag prior (weak label)
    tag_prior = 0.0
    contradict = False
    if taxonomy_map is not None:
        tags = meta.get("tag") or []
        mapped = {taxonomy_map.get(str(t)) for t in tags if taxonomy_map.get(str(t))}
        if pred_type in mapped:
            tag_prior += tag_boost
        elif len(mapped) > 0 and pred_type not in mapped:
            contradict = True

    # Blend and penalize contradictions
    conf = (w_model * base) + (w_support * support) + tag_prior
    if contradict:
        conf -= contradict_penalty

    # Final clamp
    out["confidence"] = float(max(0.0, min(1.0, conf)))
    return out

