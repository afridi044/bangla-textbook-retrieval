import json
import re
import pandas as pd
from pathlib import Path

INPUT_GLOB = "*.json"
OUT_CSV = "nodes.csv"
OUT_JSONL = "nodes.jsonl"

def clean_text(s: str) -> str:
    if not s:
        return ""
    # remove separator lines like '---'
    s = re.sub(r"\n\s*---\s*\n", "\n", s)
    # normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def flatten(node, path, rows, next_id, chapter_id, source_file):
    heading = clean_text(node.get("heading", ""))
    content = clean_text(node.get("content", ""))
    level = node.get("level", None)

    # Build heading path (skip empty headings)
    new_path = path + ([heading] if heading else [])
    heading_path = " → ".join([p for p in new_path if p])

    # keep node if it has meaningful content OR at least a heading
    if content or heading:
        rows.append({
            "node_id": next_id,
            "chapter_id": chapter_id,
            "source_file": source_file,
            "level": level,
            "heading": heading,
            "heading_path": heading_path,
            "content": content
        })
        next_id += 1

    for child in node.get("children", []) or []:
        next_id = flatten(child, new_path, rows, next_id, chapter_id, source_file)

    return next_id


def load_json_roots(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]

def main():
    base_dir = Path(__file__).resolve().parent
    # Simple deterministic order: BrTr first, then all others alphabetically.
    chapter_paths = sorted(
        base_dir.glob(INPUT_GLOB),
        key=lambda p: (0 if p.stem.strip().lower() == "brtr" else 1, p.name.lower()),
    )

    rows = []
    next_id = 0

    for chapter_path in chapter_paths:
        chapter_id = chapter_path.stem
        source_file = chapter_path.name
        roots = load_json_roots(chapter_path)

        for root in roots:
            next_id = flatten(root, [], rows, next_id, chapter_id, source_file)

    if not rows:
        raise ValueError(f"No nodes extracted. Check files matching: {INPUT_GLOB}")

    df = pd.DataFrame(rows)

    # optional: drop ultra-short junk
    df["content_len"] = df["content"].fillna("").str.len()
    df = df.drop(columns=["content_len"])

    out_csv_path = base_dir / OUT_CSV
    out_jsonl_path = base_dir / OUT_JSONL

    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    # jsonl for easy loading later
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_csv_path} ({len(df)} nodes), {out_jsonl_path}")

if __name__ == "__main__":
    main()
