bash scripts/build_kb.sh
python - <<'PY'
import json, faiss
from pathlib import Path
p = Path("geokb/.index")
print("Files:", list(p.iterdir()))
idx = faiss.read_index(str(p/"kb.index")); print("Index ntotal:", idx.ntotal)
print("OK if ntotal > 0")
PY
