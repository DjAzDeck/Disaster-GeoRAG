from pathlib import Path
from .metadata import infer_metadata

def list_images(root: str):
    root = Path(root or "assets/samples/test")
    exts = {".png",".jpg",".jpeg",".tif",".tiff"}
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return [infer_metadata(str(p)) for p in sorted(imgs)]

if __name__ == "__main__":
    import os
    root = os.environ.get("TE_TRACK3_TEST_DIR")  # set this to your demo test split
    items = list_images(root)
    print(f"Found {len(items)} images under {root or 'assets/samples/test'}")
    for it in items[:3]:
        print(it)
