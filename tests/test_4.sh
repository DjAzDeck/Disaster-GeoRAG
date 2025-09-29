python - <<'PY'
from src.georag.retriever import DisasterKB
KB = DisasterKB("geokb/.index")
print(KB.query("flooded roads and brown water", k=3))
PY