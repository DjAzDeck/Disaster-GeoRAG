#!/usr/bin/env bash
set -e
# Clone RS-LLaVA locally for imports
git clone https://github.com/BigData-KSU/RS-LLaVA.git || true
cd RS-LLaVA
pip install -r requirements.txt || true  # safe to ignore if missing
pip install transformers==4.35 accelerate peft einops sentencepiece gradio
cd ..
# Let Python find llava/*
export PYTHONPATH="$(pwd)/RS-LLaVA:${PYTHONPATH}"
echo "export PYTHONPATH=\"$(pwd)/RS-LLaVA:\$PYTHONPATH\"" >> ~/.bashrc || true
echo "RS-LLaVA installed. PYTHONPATH updated."
