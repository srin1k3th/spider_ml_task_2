import os, json, re
import pandas as pd
import config

def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def load_medquad():
    csv_path = os.path.join(config.RAW_DATA_DIR, "medquad.csv")
    if not os.path.exists(csv_path):
        print(f"  Warning: medquad.csv not found at {csv_path}. Skipping.")
        return []
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))
        if not a or a == "nan" or len(a.strip()) < 20:
            continue
        text = f"Q: {q}\nA: {a}" if q and q != "nan" else a
        source = str(row.get("source", "MedQuAD"))
        focus = str(row.get("focus_area", "general"))
        for chunk in chunk_text(text):
            docs.append({
                "text": chunk,
                "source_name": source if source != "nan" else "MedQuAD",
                "source_url": "https://www.kaggle.com/datasets/gpreda/medquad",
                "focus_area": focus if focus != "nan" else "general",
                "doc_type": "medquad"
            })
    return docs

def parse_frontmatter(content):
    meta = {"source": "Unknown", "url": "", "topic": "general"}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if match:
        for line in match.group(1).split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        content = content[match.end():]
    return meta, content

def load_guidelines():
    docs = []
    guidelines_dir = config.GUIDELINES_DIR
    if not os.path.exists(guidelines_dir):
        return docs
    for fname in os.listdir(guidelines_dir):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(guidelines_dir, fname), "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = parse_frontmatter(raw)
        for chunk in chunk_text(body):
            docs.append({
                "text": chunk,
                "source_name": meta["source"],
                "source_url": meta["url"],
                "focus_area": meta.get("topic", "general"),
                "doc_type": "guideline"
            })
    return docs

def run():
    print("loading medquad...")
    medquad = load_medquad()
    print(f"  got {len(medquad)} chunks from medquad")

    print("loading guidelines...")
    guidelines = load_guidelines()
    print(f"  got {len(guidelines)} chunks from guidelines")

    all_chunks = medquad + guidelines
    print(f"total: {len(all_chunks)} chunks")

    os.makedirs(os.path.dirname(config.CHUNKS_PATH), exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)
    print(f"saved to {config.CHUNKS_PATH}")

if __name__ == "__main__":
    run()
