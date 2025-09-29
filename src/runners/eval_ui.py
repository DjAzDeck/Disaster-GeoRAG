import yaml
from io import StringIO
from src.dataio.wds_loader import iter_wds
from src.georag.retriever import DisasterKB, compose_retrieval_query_gaia, calibrate_confidence
from src.georag.prompting import build_user_prompt, enforce_json
from src.vlm.qwen2vl import Qwen2VL
from src.metrics.disaster_metrics import f1_binary, macro_f1_types

GAIA2DISASTER = {
    "Wildfire":"wildfire","Wildfires":"wildfire","Floods":"flood","Flood":"flood",
    "Severe Storms":"storm_wind","Storm":"storm_wind","Cyclone":"storm_wind","Hurricane":"storm_wind",
    "Volcano":"volcano","Eruption":"volcano","Earthquake":"earthquake","Landslide":"landslide",
    "Drought":"drought","Industrial":"industrial"
}
LABELS = ["flood","wildfire","storm_wind","earthquake","landslide","volcano","drought","industrial"]

def tag_to_label(tags):
    if not tags: return None
    for t in tags:
        lab = GAIA2DISASTER.get(str(t))
        if lab: return lab
    return None

def run_eval(shards_dir: str, limit: int = 200, topk: int = 3):
    cfg_prompt = yaml.safe_load(open("configs/prompts.yaml"))
    system = cfg_prompt["system"]; user_tpl = cfg_prompt["user_template"]
    kb = DisasterKB("geokb/.index")
    model = Qwen2VL()

    rows=[]
    for i,rec in enumerate(iter_wds(shards_dir, limit=limit if limit>0 else None)):
        retrieval_query = compose_retrieval_query_gaia(rec.text, rec.meta_dict, GAIA2DISASTER)
        snips = kb.query(retrieval_query, k=topk)
        prompt = build_user_prompt(user_tpl, rec.eo, snips)
        full_prompt = system + "\n\n" + prompt
        raw = model.infer_json(rec.image_pil, full_prompt)
        js = enforce_json(raw) or {"is_disaster":"uncertain","type":None,"confidence":0.0,"rationale":raw,"evidence_ids":[]}
        js = calibrate_confidence(js, snips, rec.meta_dict, GAIA2DISASTER)

        tags = rec.meta_dict.get("tag") or []
        gaia_type = tag_to_label(tags)
        gaia_is_disaster = bool(gaia_type)
        pred_type = js.get("type")
        pred_is_disaster = (js.get("is_disaster") is True) or (isinstance(js.get("is_disaster"),str) and js["is_disaster"].lower()=="true")

        rows.append({"pred_d": int(pred_is_disaster), "gt_d": int(gaia_is_disaster),
                     "pred_t": pred_type, "gt_t": gaia_type, "conf": float(js.get("confidence") or 0.0)})

    # Binary F1
    binm = f1_binary([r["pred_d"]==1 for r in rows], [r["gt_d"]==1 for r in rows])

    # Type macro-F1 (only where weak GT exists)
    typed_rows = [r for r in rows if r["gt_t"] is not None]
    macro = macro_f1_types([r["pred_t"] for r in typed_rows], [r["gt_t"] for r in typed_rows], LABELS)

    bins=[{"n":0,"ok":0} for _ in range(10)]
    for r in typed_rows:
        b=min(9, int(r["conf"]*10))
        bins[b]["n"]+=1
        if r["pred_t"]==r["gt_t"]: bins[b]["ok"]+=1
    rel = []
    for i,b in enumerate(bins):
        rel.append({"bin":i, "count":b["n"], "accuracy": (b["ok"]/b["n"]) if b["n"] else None, "conf_center": (i+0.5)/10})

    md = StringIO()
    md.write("## Evaluation (weak labels from GAIA tags)\n")
    md.write(f"- **Binary (disaster) F1**: {binm['f1']:.3f}  &nbsp;&nbsp; precision: {binm['precision']:.3f}  recall: {binm['recall']:.3f}\n")
    md.write(f"- **Type macro-F1**: {macro:.3f}  (labels: {', '.join(LABELS)})\n")
    md.write("\n**Reliability (confidence vs accuracy):**\n\n")
    md.write("| bin | conf_center | count | accuracy |\n|---:|---:|---:|---:|\n")
    for r in rel:
        acc = "-" if r["accuracy"] is None else f"{r['accuracy']:.2f}"
        md.write(f"| {r['bin']} | {r['conf_center']:.2f} | {r['count']} | {acc} |\n")
    return md.getvalue()
