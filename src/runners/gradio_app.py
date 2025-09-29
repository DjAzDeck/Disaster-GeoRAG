import os, json, yaml
import gradio as gr
from typing import List, Dict, Any
import numpy as np
from PIL import Image

from src.dataio.metadata import EOImage
from src.dataio.wds_loader import build_catalog, get_sample_by_index, list_shards
from src.georag.retriever import DisasterKB, compose_retrieval_query_gaia, calibrate_confidence
from src.georag.prompting import build_user_prompt, enforce_json
from src.vlm.qwen2vl import Qwen2VL
from src.metrics.disaster_metrics import f1_binary, macro_f1_types
from src.localize.owl import detect_boxes, draw_overlay, measure_area, prompts_for_type
from src.runners.utils import OVERLAY_DIR, LABELS, TAG_CANON
from src.runners.utils import save_png, to_data_uri, extract_gaia_tags, weak_label_from_meta, gsd_from_meta, wants_localization, to_messages


cfg = yaml.safe_load(open("configs/prompts.yaml"))
system = cfg["system"]; user_tpl = cfg["user_template"]
kb = DisasterKB("geokb/.index")
qwen = Qwen2VL()

def on_chat_send(history, user_text, pil_image, meta_json, last_pred_type):
    if pil_image is None: raise gr.Error("Load a sample first.")
    history = to_messages(history)
    reply_text, overlay_pil = "", pil_image
    loc_flag = wants_localization(user_text)

    try:
        if loc_flag:
            phrases = prompts_for_type(last_pred_type) or ["disaster footprint","damage","impact"]
            boxes, (H,W) = detect_boxes(pil_image, phrases)
            boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)[:3]
            overlay_pil = draw_overlay(pil_image, boxes)
            gsd = gsd_from_meta(json.loads(meta_json) if meta_json else {})
            stats = measure_area(boxes, H, W, gsd_m=gsd)
            reply_text = f"Outlined {len(boxes)} region(s). Coverage: {stats['coverage']*100:.1f}%."
            if stats.get("area_km2") is not None:
                reply_text += f" ~{stats['area_km2']:.2f} km² (assuming {gsd:.0f} m/px)."
        else:
            qhist = [{"role": m["role"], "text": m["content"]} for m in history]
            reply_text = qwen.chat_once(pil_image, qhist, user_text, max_new_tokens=192)
    except Exception as e:
        reply_text = f"{'Localization' if loc_flag else 'Chat'} error: {e}"

    new_messages = history + [
        {"role":"user","content":user_text},
        {"role":"assistant","content":reply_text},
    ]
    return new_messages, new_messages, to_data_uri(overlay_pil), save_png(overlay_pil, prefix="overlay")

# ---------- data actions ----------
def do_list_shards(shards_dir: str):
    rows = list_shards(shards_dir)
    headers = ["shard","size_mb","path"]
    table = [[r[h] for h in headers] for r in rows]
    choices = ["ALL"] + [r["shard"] for r in rows]
    return table, gr.update(choices=choices, value="ALL"), f"Found {len(rows)} shards"

def do_scan(shards_dir: str, shard_filter: str, limit: int):
    target = shards_dir if (not shard_filter or shard_filter=="ALL") else os.path.join(shards_dir, shard_filter)
    cat = build_catalog(target, limit=limit if limit>0 else None)
    headers = ["idx","shard","id","title","location","lat","lon","tags","satellite","sensor","modalities","resolution"]
    table = [[row.get(h,"") for h in headers] for row in cat]
    choices = [f"{row['idx']} | {row['title'] or row['id']}" for row in cat]
    return table, gr.update(choices=choices, value=None), json.dumps(cat, ensure_ascii=False)

def load_selected(shards_dir: str, selector: str, shard_filter: str):
    if not selector: raise gr.Error("Select a sample first.")
    idx = int(selector.split("|",1)[0].strip())
    target = shards_dir if (not shard_filter or shard_filter=="ALL") else os.path.join(shards_dir, shard_filter)
    rec = get_sample_by_index(target, index=idx)
    meta_str = json.dumps(rec.meta_dict, indent=2, ensure_ascii=False)
    return (
        rec.image_pil, rec.eo.sensor or "", rec.eo.satellite or "", rec.eo.modality or "",
        rec.eo.resolution or "", rec.eo.location or "", rec.eo.lat, rec.eo.lon,
        meta_str, meta_str, to_data_uri(rec.image_pil), save_png(rec.image_pil, prefix="loaded")
    )

# ---------- inference / triage ----------
def triage(img, sensor, satellite, modality, resolution, location, lat, lon, date, cloud, k, meta_json: str):
    meta_eo = EOImage(
        path="<ui>", sensor=(sensor or None), satellite=(satellite or None),
        modality=(modality or None), resolution=(resolution or None),
        location=(location or None), lat=(lat if lat not in (None,"") else None),
        lon=(lon if lon not in (None,"") else None), date=(date or None),
        cloud=(int(cloud) if cloud not in (None,"") else None)
    )
    try: gaia_meta = json.loads(meta_json) if meta_json else {}
    except Exception: gaia_meta = {}

    text_for_retrieval = compose_retrieval_query_gaia(
        " ".join([str(x) for x in [gaia_meta.get("image_alt",""),
                                   " ".join(gaia_meta.get("captions") or []),
                                   gaia_meta.get("location","")] if x]),
        gaia_meta if isinstance(gaia_meta, dict) else {},
        {k: v for k, v in TAG_CANON.items()}
    )
    snips = kb.query(text_for_retrieval, k=int(k))
    prompt = build_user_prompt(user_tpl, meta_eo, snips)
    full = system + "\n\n" + prompt

    raw = qwen.infer_json(img, full)
    js = enforce_json(raw) or {"is_disaster":"uncertain","type":None,"confidence":0.0,"rationale":raw,"evidence_ids":[]}
    pred_type = js.get("type")
    support_scores = [s["score"] for s in snips if pred_type and s.get("type")==pred_type]
    support = sum(support_scores)/len(support_scores) if support_scores else 0.0
    js = calibrate_confidence(js, snips, gaia_meta if isinstance(gaia_meta, dict) else {}, {k: v for k, v in TAG_CANON.items()})
    pred_is_disaster = (js.get("is_disaster") is True) or (isinstance(js.get("is_disaster"), str) and js["is_disaster"].lower()=="true")
    gaia_type = weak_label_from_meta(gaia_meta)
    evidence = "\n".join([f"- {s['id']} ({s['type']}): {s['cue']}" for s in snips])

    # keep preview in sync (show current image)
    html_img = to_data_uri(img if isinstance(img, Image.Image) else Image.fromarray(np.array(img)).convert("RGB"))
    saved = save_png(img if isinstance(img, Image.Image) else Image.fromarray(np.array(img)), prefix="triage")

    return (
        json.dumps(js, indent=2, ensure_ascii=False),
        evidence,
        full,
        html_img,
        pred_type or "", int(pred_is_disaster), float(js.get("confidence") or 0.0), float(support),
        gaia_type or "",
        saved
    )

# ---------- UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 Disaster-GeoRAG (DGRAG)")

    st_catalog_json = gr.State("")
    st_meta_json = gr.State("")
    st_eval_rows = gr.State([])

    with gr.Tab("Analyst Chat"):
        gr.Markdown("Ask about the **current image** (e.g., *Where is the flooding? Outline and area*).")
        chat_history = gr.State([])
        chatbox = gr.Chatbot(type="messages", height=300) # expects [{role,content}]
        chat_input = gr.Textbox(placeholder="Ask about this image…")
        btn_send = gr.Button("Send")

    with gr.Row():
        with gr.Column(scale=1):
            shards_dir = gr.Textbox(label="Folder with .tar shards (or brace pattern)", value=os.environ.get("TE_TRACK3_TEST_WDS",""))
            btn_list = gr.Button("List shards in folder")
            shards_tbl = gr.Dataframe(headers=["shard","size_mb","path"], interactive=False, max_height=160)
            shard_filter = gr.Dropdown(label="Filter by shard (optional)", choices=["ALL"], value="ALL")
            status_md = gr.Markdown()
            limit = gr.Slider(0, 5000, step=1, value=0, label="Scan limit (0 = all)")
            btn_scan = gr.Button("Scan → build catalog")
            catalog_tbl = gr.Dataframe(
                headers=["idx","shard","id","title","location","lat","lon","tags","satellite","sensor","modalities","resolution"],
                datatype=["number","str","str","str","str","number","number","str","str","str","str","str"],
                interactive=False, max_height=240
            )
            selector = gr.Dropdown(label="Select sample (idx | title)", choices=[], interactive=True)
            btn_load = gr.Button("Load selected")

        with gr.Column(scale=1):
            img = gr.Image(type="pil", image_mode="RGB", label="Satellite image", height=384)
            sensor = gr.Textbox(label="Sensor (e.g., MSI)")
            satellite = gr.Textbox(label="Satellite (e.g., Sentinel-2)")
            modality = gr.Textbox(label="Modality (e.g., Optical/SAR)")
            resolution = gr.Textbox(label="Resolution (e.g., High)")
            location = gr.Textbox(label="Location")
            lat = gr.Number(label="Lat", precision=6)
            lon = gr.Number(label="Lon", precision=6)
            date = gr.Textbox(label="Date (YYYY-MM-DD)")
            cloud = gr.Slider(0,100,step=1,label="Cloud %", value=0)
            k = gr.Slider(1,6,step=1,label="KB Top-k", value=3)
            btn_run = gr.Button("Run triage")

        with gr.Column(scale=1):
            img_html = gr.HTML(label="Preview (always renders)")
            out_json = gr.Code(label="Prediction (JSON)", language="json")
            out_ev   = gr.Markdown(label="Retrieved evidence")
            out_pr   = gr.Textbox(label="Full prompt (debug)")
            cur_meta = gr.Code(label="Current GAIA metadata (JSON)", language="json")
            img_file = gr.File(label="Saved PNG (download)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Evaluation (only samples you add here)")
            btn_add = gr.Button("+ Add current prediction to evaluation set")
            eval_tbl = gr.Dataframe(
                headers=["gaia_id","gaia_tag_label","gaia_is_disaster","pred_type","pred_is_disaster","confidence","support"],
                interactive=False, max_height=240
            )
            with gr.Row():
                btn_compute = gr.Button("Compute metrics on selected set")
                btn_clear   = gr.Button("Clear evaluation set")
        with gr.Column(scale=1):
            md_eval = gr.Markdown()

    # events
    btn_list.click(do_list_shards, [shards_dir], [shards_tbl, shard_filter, status_md])
    btn_scan.click(do_scan, [shards_dir, shard_filter, limit], [catalog_tbl, selector, st_catalog_json])

    btn_load.click(
        load_selected, [shards_dir, selector, shard_filter],
        [img, sensor, satellite, modality, resolution, location, lat, lon,
         st_meta_json, cur_meta, img_html, img_file]
    )

    last_type = gr.State("")
    last_is_d = gr.State(0)
    last_conf= gr.State(0.0)
    last_sup = gr.State(0.0)
    last_gaia_t = gr.State("")

    btn_run.click(
        triage,
        [img, sensor, satellite, modality, resolution, location, lat, lon, date, cloud, k, st_meta_json],
        [out_json, out_ev, out_pr, img_html, last_type, last_is_d, last_conf, last_sup, last_gaia_t, img_file]
    )

    def add_to_eval(eval_rows: List[Dict[str,Any]], meta_json: str,
                    pred_type: str, pred_is_disaster: int, confidence: float, support: float, json_pred_str: str):
        try: meta = json.loads(meta_json) if meta_json else {}
        except Exception: meta = {}
        tags = extract_gaia_tags(meta)
        gaia_label = weak_label_from_meta(meta)
        row = {
            "gaia_id": meta.get("id"),
            "gaia_tags": tags,
            "gaia_tag_label": gaia_label,
            "gaia_is_disaster": int(bool(gaia_label)),
            "pred_type": pred_type or None,
            "pred_is_disaster": int(pred_is_disaster),
            "confidence": float(confidence),
            "support": float(support),
            "json": json_pred_str,
        }
        eval_rows = (eval_rows or []) + [row]
        headers = ["gaia_id","gaia_tag_label","gaia_is_disaster","pred_type","pred_is_disaster","confidence","support"]
        table = [[r.get(h,"") for h in headers] for r in eval_rows]
        return eval_rows, table
    def clear_eval(_rows): return [], []
    def compute_eval(eval_rows: List[Dict[str,Any]]):
        if not eval_rows: return "No rows in evaluation set. Add some predictions first."
        y_pred = [r["pred_is_disaster"]==1 for r in eval_rows if r.get("gaia_is_disaster") is not None]
        y_true = [r["gaia_is_disaster"]==1 for r in eval_rows if r.get("gaia_is_disaster") is not None]
        binm = f1_binary(y_pred, y_true) if y_true else {"precision":0.0, "recall":0.0, "f1":0.0}
        typed = [r for r in eval_rows if r.get("gaia_tag_label") is not None]
        macro = macro_f1_types([r.get("pred_type") for r in typed], [r.get("gaia_tag_label") for r in typed], LABELS) if typed else 0.0
        bins=[{"n":0,"ok":0} for _ in range(10)]
        for r in typed:
            b=min(9, int(float(r.get("confidence") or 0.0)*10))
            bins[b]["n"]+=1
            if r.get("pred_type")==r.get("gaia_tag_label"): bins[b]["ok"]+=1
        lines = ["## Evaluation (selected set only)"]
        lines.append(f"- **Binary F1**: {binm['f1']:.3f}  (P={binm['precision']:.3f}, R={binm['recall']:.3f})")
        lines.append(f"- **Type macro-F1**: {macro:.3f}  (labels: {', '.join(LABELS)})")
        lines.append("\n**Reliability (confidence vs. accuracy)**")
        lines.append("| bin | conf_center | count | accuracy |")
        lines.append("|---:|---:|---:|---:|")
        for i,b in enumerate(bins):
            acc = (b["ok"]/b["n"]) if b["n"] else None
            lines.append(f"| {i} | {(i+0.5)/10:.2f} | {b['n']} | {'-' if acc is None else f'{acc:.2f}'} |")
        return "\n".join(lines)

    btn_add.click(
        add_to_eval,
        [st_eval_rows, st_meta_json, last_type, last_is_d, last_conf, last_sup, out_json],
        [st_eval_rows, eval_tbl]
    )
    btn_compute.click(compute_eval, [st_eval_rows], [md_eval])
    btn_clear.click(clear_eval, [st_eval_rows], [st_eval_rows, eval_tbl])

    btn_send.click(
        on_chat_send,
        [chat_history, chat_input, img, st_meta_json, last_type],
        [chat_history, chatbox, img_html, img_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[OVERLAY_DIR])
