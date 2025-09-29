# Disaster-GeoRAG (DGRAG)

This is a self-contained repository for the submission on [Thinking Earth Hackathon - BiDS 2025](https://allhackathons.com/hackathon/harnessing-copernicus-foundation-models-to-decode-earth-from-space/). 

Some extra [resources](https://github.com/Orion-AI-Lab/ThinkingEarth_Hackathon_BiDS25).

## Solution direction

A retrieval-augmented VLM pipeline for Earth Observation disaster triage on the [GAIA test split](https://huggingface.co/datasets/azavras/GAIA/viewer/default/test?views%5B%5D=test). We stream samples directly from WebDataset .tar shards, run a grounded visual analysis with Qwen2-VL-7B-Instruct, and produce a compact JSON verdict: is_disaster, type, confidence, rationale, evidence_ids. A Gradio app lets you browse shards, preview images, run triage, and compute quick weak-label metrics (binary F1 and type macro-F1) using GAIA tags.

# Features

- Shard browser → catalog → pick sample from .tar files; stream without unpacking. 

- Grounded triage: Qwen2-VL prompts + retrieval cues (GeoRAG) -> JSON output. 

- Confidence calibration: blends model score with retrieval support and GAIA tag priors (simple, post-hoc). 

- In-app evaluation on the selected samples only (weak labels from GAIA tags).


# Repository layout

```
src/
  runners/gradio_app.py        # full UI: shard browser, preview, triage, eval
  dataio/wds_loader.py     # WebDataset loader + catalog + sample-by-index
  georag/                  # KB, retrieval, confidence calibration
  vlm/qwen2vl.py           # VLM wrapper (deterministic decoding)
  metrics/disaster_metrics.py
configs/prompts.yaml

```

## Output schema

```json
{
  "is_disaster": true|false|"uncertain",
  "type": "wildfire|flood|storm_wind|earthquake|landslide|volcano|drought|industrial|null",
  "confidence": 0.0-1.0,
  "rationale": "short explanation",
  "evidence_ids": ["kb:..."]
}

```

## Quickstart

Tested on Python 3.12.3

```
# If you are in a cluster: 
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

python -m venv venv
source venv/bin/activate
pip install -U pip setuptool wheel
pip install -r requirements.txt
### VLM related
pip install -U transformers
pip install qwen-vl-utils

# For use with llava (experimental - no use) ---
pip3 install torch torchvision
pip install transformers==4.35
pip install accelerate==0.31.0
pip install -U accelerate
```

## Running steps: 

1) Run `bash scripts/build_kb.sh`

2) Then just `bash run_app.sh`

**IMPORTANT:** The workflow requires at least 18GB of VRAM!

## Limitations: 

- Evaluation uses weak labels derived from GAIA tags/text; it’s indicative, not a benchmark score. 

- No object-level localization; triage-first design.

- Deterministic decoding favors stability over generative diversity.

## Contributors

Athansios Trantas & Udit Asopa