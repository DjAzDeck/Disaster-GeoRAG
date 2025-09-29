bash scripts/install_rs-llava.sh
python - <<'PY'
import importlib, sys; print("PYTHONPATH OK:", any("RS-LLaVA" in p for p in sys.path))
print("llava import:", importlib.import_module("llava") is not None)
PY
