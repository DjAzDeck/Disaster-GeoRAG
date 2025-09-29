from typing import List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw
import torch
from io import BytesIO


def _lazy_grounding():
    from groundingdino.util.inference import load_model, predict, load_image
    return load_model, predict, load_image

def _lazy_sam():
    from segment_anything import SamPredictor, sam_model_registry
    return SamPredictor, sam_model_registry

# NOTE You need to download these weights MANUALY
CFG = {
    "gdino_cfg": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "gdino_wts": "weights/groundingdino_swint_ogc.pth",
    "sam_wts":"weights/sam_vit_h.pth",
}

PROMPTS = {
    "wildfire": ["burn scar", "wildfire scar", "smoke plume"],
    "flood": ["floodwater", "inundation", "standing water"],
    "storm_wind":["wind damage","debris","downed trees"],
    "earthquake":["collapsed building","rubble"],
    "landslide":["landslide","mudslide","slope failure"],
    "volcano": ["lava flow","ash plume"],
    "drought": ["dry riverbed","parched field"],
    "industrial":["oil spill","industrial fire","chemical plume"],
}

def detect_boxes(image: Image.Image, phrases: List[str], box_thresh=0.25, text_thresh=0.25):
    load_model, predict, load_image = _lazy_grounding()
    model = load_model(CFG["gdino_cfg"], CFG["gdino_wts"])

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image)).convert("RGB")
    else:
        image = image.convert("RGB")

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # im_tensor = torch.Tensor(im)
    image_source, processed_image = load_image(buf)

    text = " . ".join(phrases)
    boxes, logits, phrases_out = predict(model=model, image=processed_image, 
                                        caption=text, box_threshold=box_thresh, 
                                        text_threshold=text_thresh)
    # boxes: xyxy in absolute pixels
    results=[]
    H, W = image_source.shape[:2]
    for b, s, p in zip(boxes, logits, phrases_out):
        x1,y1,x2,y2 = [int(v) for v in b]
        x1 = max(0,min(W-1,x1)); x2 = max(0,min(W-1,x2))
        y1 = max(0,min(H-1,y1)); y2 = max(0,min(H-1,y2))
        if x2>x1 and y2>y1:
            results.append({"xyxy":(x1,y1,x2,y2), "score":float(s), "label":p})
    return results, (H,W)

def refine_mask(image: Image.Image, box_xyxy) -> np.ndarray:
    try:
        SamPredictor, sam_model_registry = _lazy_sam()
    except Exception:
        return None  # NO SAM
    sam = sam_model_registry["vit_h"](checkpoint=CFG["sam_wts"])
    predictor = SamPredictor(sam)
    img = np.array(image.convert("RGB"))
    predictor.set_image(img)
    box = np.array([box_xyxy], dtype=np.float32)
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    return masks[0].astype(np.uint8)  # 0/1

def draw_overlay(image: Image.Image, boxes: List[Dict[str,Any]], mask: np.ndarray|None=None):
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im, "RGBA")
    for b in boxes:
        x1,y1,x2,y2 = b["xyxy"]; s=b["score"]; lbl=b["label"]
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0,200), width=3)
        draw.rectangle([x1,y1,x1+6,y1+6], fill=(255,0,0,200))
    if mask is not None:
        # semi-transparent fill
        overlay = Image.fromarray((mask*120).astype(np.uint8))
        im = Image.composite(Image.new("RGB", im.size, (0,255,0)), im, overlay.resize(im.size))
    return im

def measure_area(mask: np.ndarray|None, boxes: List[Dict[str,Any]], H: int, W: int, gsd_m: float|None=None):
    if mask is not None:
        px = int(mask.sum())
    elif boxes:
        # union of boxes as rough proxy
        canvas = np.zeros((H,W), dtype=np.uint8)
        for b in boxes:
            x1,y1,x2,y2=b["xyxy"]
            canvas[y1:y2, x1:x2]=1
        px = int(canvas.sum())
    else:
        return {"coverage":0.0, "pixels":0, "area_km2":None}
    coverage = px / float(H*W)
    area_km2 = (px * (gsd_m**2) / 1e6) if gsd_m else None
    return {"coverage":coverage, "pixels":px, "area_km2":area_km2}

def prompts_for_type(dtype: str) -> List[str]:
    return PROMPTS.get(dtype or "", []) or ["disaster footprint","damage","impact"]
