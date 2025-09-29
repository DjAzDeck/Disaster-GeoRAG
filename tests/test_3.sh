export TE_TRACK3_TEST_WDS="/home/<user>/github/ThinkingEarth_Hackathon_BiDS25/track3/test/00015.tar"

python -m src.dataio.wds_loader

# Directory with many shards
export TE_TRACK3_TEST_WDS="/home/<user>/github/ThinkingEarth_Hackathon_BiDS25/track3/test/"
python -m src.dataio.wds_loader
# Expect: prints the first sample's text, a few meta fields, and EO dataclass.
