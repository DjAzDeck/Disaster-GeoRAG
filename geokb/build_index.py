import argparse, json, os, faiss, numpy as np, csv
from sentence_transformers import SentenceTransformer

def load_kb(tsv_path):
    items=[]
    with open(tsv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['id'].startswith('#') or not row['id']: continue
            items.append({"id":row['id'], "type":row['type'], "cue":row['cue'], "source":row.get('source','')})
    return items

def build_index(items, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    enc = SentenceTransformer(model_name)
    texts = [f"{it['type']}: {it['cue']}" for it in items]
    X = enc.encode(texts, normalize_embeddings=True)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X.astype('float32'))
    return index, np.array(X, dtype='float32'), items

def save(index, X, items, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "kb.index"))
    np.save(os.path.join(out_dir, "kb.npy"), X)
    with open(os.path.join(out_dir, "kb_items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_tsv", default="geokb/disaster_cues.tsv")
    ap.add_argument("--out_dir", default="geokb/.index")
    args = ap.parse_args()
    items = load_kb(args.kb_tsv)
    idx, X, items = build_index(items)
    save(idx, X, items, args.out_dir)
    print(f"Indexed {len(items)} cues → {args.out_dir}")
