from __future__ import annotations
import os, json, csv, yaml
from src.dataio.wds_loader import iter_wds
from src.georag.retriever import DisasterKB, compose_retrieval_query_gaia
from src.georag.prompting import build_user_prompt, enforce_json
# from src.vlm.rs_llava import RSLLaVA
from src.vlm.qwen2vl import Qwen2VL


GAIA2DISASTER = {
    "Wildfire":"wildfire","Wildfires":"wildfire","Floods":"flood","Flood":"flood",
    "Severe Storms":"storm_wind","Storm":"storm_wind","Cyclone":"storm_wind","Hurricane":"storm_wind",
    "Volcano":"volcano","Eruption":"volcano","Earthquake":"earthquake","Landslide":"landslide",
    "Drought":"drought","Industrial":"industrial"
}

def main(shards_pattern: str, out_csv="results_test_wds.csv", topk:int=3, limit:int|None=None):
    cfg_prompt = yaml.safe_load(open("configs/prompts.yaml"))
    system = cfg_prompt["system"]; user_tpl = cfg_prompt["user_template"]
    kb = DisasterKB("geokb/.index")
    # model = RSLLaVA()
    model = Qwen2VL()

    rows=[]
    for rec in iter_wds(shards_pattern, limit=limit):
        # Better Retrieval v2: compose rich query from GAIA fields
        retrieval_query = compose_retrieval_query_gaia(rec.text, rec.meta_dict, GAIA2DISASTER)
        snips = kb.query(retrieval_query, k=topk)
        prompt = build_user_prompt(user_tpl, rec.eo, snips)
        full_prompt = system + "\n\n" + prompt
        raw = model.infer_json(rec.image_pil, full_prompt)
        js = enforce_json(raw) or {"is_disaster":"uncertain","type":None,"confidence":0.0,"rationale":raw,"evidence_ids":[]}
        rows.append({
            "image": "<wds>", "sensor": rec.eo.sensor, "satellite": rec.eo.satellite,
            "modality": rec.eo.modality, "resolution": rec.eo.resolution,
            "location": rec.eo.location, "lat": rec.eo.lat, "lon": rec.eo.lon,
            "raw": raw, "json": json.dumps(js, ensure_ascii=False),
            "retrieval_query": retrieval_query[:400]
        })
        print(f"OK: type={js.get('type')} conf={js.get('confidence')}  loc={rec.eo.location}  sat={rec.eo.satellite}")

    if rows:
        with open(out_csv, "w", newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {out_csv} with {len(rows)} rows.")
    else:
        print("No samples read. Check shards path/pattern.")

if __name__ == "__main__":
    shards = os.environ.get("TE_TRACK3_TEST_WDS", "")
    assert shards, "Set TE_TRACK3_TEST_WDS to /path/to/test or /path/to/test/{00000..00015}.tar"
    main(shards, out_csv="results_test_wds.csv", topk=3, limit=None)
