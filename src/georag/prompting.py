def format_retrieved(snips):
    lines = []
    for s in snips:
        lines.append(f"- [{s['id']}] ({s['type']}) {s['cue']}")
    return "\n".join(lines)

def build_user_prompt(tpl: str, meta, retrieved):
    return tpl.format(
        sensor=meta.sensor, date=meta.date, cloud=meta.cloud,
        retrieved=format_retrieved(retrieved)
    )

def enforce_json(s: str):
    # attempt to extract first JSON object
    import re, json
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

def build_user_prompt(tpl: str, meta, retrieved):
    return tpl.format(
        sensor=meta.sensor or "unknown",
        satellite=meta.satellite or "unknown",
        modality=meta.modality or "unknown",
        resolution=meta.resolution or "unknown",
        location=meta.location or "unknown",
        lat=meta.lat if meta.lat is not None else "unknown",
        lon=meta.lon if meta.lon is not None else "unknown",
        date=meta.date or "unknown",
        cloud=meta.cloud if meta.cloud is not None else 0,
        retrieved=format_retrieved(retrieved)
    )
