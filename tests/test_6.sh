python - <<'PY'
from PIL import Image
from src.vlm.rs_llava import RSLLaVA
m = RSLLaVA()
im = Image.new("RGB",(256,256),(128,128,128))
out = m.infer_json(im, "Say only: {\"is_disaster\": false, \"type\": null, \"confidence\": 0.0, \"rationale\":\"test\",\"evidence_ids\":[]}")
print("Model returned:", out[:120], "...")
PY
# OK if it prints a short string (we’ll parse JSON downstream).
