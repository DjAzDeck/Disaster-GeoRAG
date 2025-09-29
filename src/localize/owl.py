from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection

_PROCESSOR = None
_MODEL = None

def _load_owlv2(model_id: str = "google/owlv2-base-patch16-ensemble"):
    global _PROCESSOR, _MODEL
    if _PROCESSOR is None:
        _PROCESSOR = Owlv2Processor.from_pretrained(model_id)
    if _MODEL is None:
        _MODEL = Owlv2ForObjectDetection.from_pretrained(model_id).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        ).eval()
    return _PROCESSOR, _MODEL

def _iou_vec(b: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b[0], others[:,0]); y1 = np.maximum(b[1], others[:,1])
    x2 = np.minimum(b[2], others[:,2]); y2 = np.minimum(b[3], others[:,3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (b[2]-b[0])*(b[3]-b[1]); area_o = (others[:,2]-others[:,0])*(others[:,3]-others[:,1])
    union = area_b + area_o - inter + 1e-9
    return inter / union

def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> List[int]:
    if len(boxes) == 0: return []
    idxs = scores.argsort()[::-1]; keep = []
    while len(idxs):
        i = idxs[0]; keep.append(i)
        if len(idxs) == 1: break
        ious = _iou_vec(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thr]
    return keep

def _expand_prompts(phrases: List[str]) -> List[str]:
    # Improve recall in EO by adding RS-friendly phrasings:
    base = [p.strip() for p in phrases if p and p.strip()]
    extras = [f"satellite image of {p}" for p in base] + [f"aerial view of {p}" for p in base]
    return base + extras

def detect_boxes(image: Image.Image, phrases: List[str],
                 score_thresh: float = 0.10, iou_thr: float = 0.5, topk_per_query: int = 6
                 ) -> Tuple[List[Dict[str,Any]], Tuple[int,int]]:
    """
    Text-conditioned zero-shot detection with OWLv2.
    Returns: list of {xyxy, score, label}, and (H, W).
    """
    proc, model = _load_owlv2()
    pil = image.convert("RGB")
    H, W = pil.height, pil.width

    # IMPORTANT: nested list + grounded post-processing (per HF docs)
    phrases = _expand_prompts(phrases)
    text_labels = [phrases]  # nested
    inputs = proc(text=text_labels, images=pil, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(H, W)], device=model.device)
    results = proc.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=score_thresh, text_labels=text_labels
    )[0]

    boxes = results["boxes"].detach().cpu().numpy()   # xyxy in pixels
    scores = results["scores"].detach().cpu().numpy()
    labels = results["text_labels"] # list[str], already mapped

    # Keep top-k per original (non-expanded) query by grouping on label text
    keep = np.argsort(scores)[::-1]
    boxes, scores, labels = boxes[keep], scores[keep], [labels[i] for i in keep]

    # Global NMS
    keep = _nms_xyxy(boxes, scores, iou_thr=iou_thr)
    boxes, scores, labels = boxes[keep], scores[keep], [labels[i] for i in keep]

    out=[]
    for b,s,lbl in zip(boxes, scores, labels):
        x1,y1,x2,y2 = [int(v) for v in b.tolist()]
        out.append({"xyxy": (x1,y1,x2,y2), "score": float(s), "label": lbl})
    return out, (H, W)

def draw_overlay(image: Image.Image, boxes: List[Dict[str,Any]]) -> Image.Image:
    im = image.convert("RGB").copy()
    dr = ImageDraw.Draw(im, "RGBA")
    for b in boxes:
        x1,y1,x2,y2 = b["xyxy"]; sc=b["score"]; lbl=b["label"]
        dr.rectangle([x1,y1,x2,y2], outline=(255,0,0,220), width=3)
        # compact label box
        dr.rectangle([x1, max(0,y1-16), min(im.width, x1+160), y1], fill=(255,0,0,180))
        dr.text((x1+2, y1-14), f"{lbl[:18]} {sc:.2f}", fill=(255,255,255,255))
    return im

def measure_area(boxes: List[Dict[str,Any]], H: int, W: int, gsd_m: float | None = None):
    if not boxes: return {"coverage":0.0, "pixels":0, "area_km2":None}
    canvas = np.zeros((H,W), dtype=np.uint8)
    for b in boxes:
        x1,y1,x2,y2 = b["xyxy"]
        canvas[max(0,y1):max(0,y2), max(0,x1):max(0,x2)] = 1
    px = int(canvas.sum())
    coverage = px / float(H*W)
    area_km2 = (px * (gsd_m**2) / 1e6) if gsd_m else None
    return {"coverage": coverage, "pixels": px, "area_km2": area_km2}

PROMPTS = {
    "wildfire": ["burn scar","wildfire scar","smoke plume"],
    "flood":["floodwater","inundation","standing water","overflowed river"],
    "storm_wind": ["wind damage","downed trees","debris"],
    "earthquake": ["collapsed building","rubble"],
    "landslide": ["landslide","mudslide","slope failure"],
    "volcano":["lava flow","ash plume"],
    "drought": ["dry riverbed","parched field"],
    "industrial": ["oil spill","industrial fire","chemical plume"],
}
def prompts_for_type(dtype: str) -> List[str]:
    return PROMPTS.get(dtype or "", []) or ["disaster footprint","damage","impact"]
