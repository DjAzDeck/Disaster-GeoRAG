#!/usr/bin/env bash
set -e
python geokb/build_index.py --kb_tsv geokb/disaster_cues.tsv --out_dir geokb/.index
