import re, json, random, warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")

# ─── Paths (local defaults) ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODES_CSV    = str(PROJECT_ROOT / "DatasetPrep" / "nodes.csv")
QA_GOLD_CSV  = str(PROJECT_ROOT / "DatasetPrep" / "qa_Gold.csv")
OUTPUT_DIR   = str(PROJECT_ROOT / "results")
CHUNKING_JSON_PATH = ""  # if empty, defaults to <project_root>/results/chunkingAb.json

BM25_ONLY    = False          # set True to skip dense and save time
MODEL_NAME   = "intfloat/multilingual-e5-base"

SEEDS        = (42, 43, 44)
TRAIN_RATIO  = 0.75
VAL_RATIO    = 0.10
RECALL_AT_K  = (1, 3, 5, 10, 20)
BOOTSTRAP_N  = 1000

# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─── BM25 tokenizer (Bangla-aware, mirrors train_retriever.py) ────────────────
_BN_RE = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)

def bm25_tokenize(text):
    text = re.sub(r"\s+", " ", str(text).lower()).strip()
    return _BN_RE.findall(text)

# ─── Node-disjoint split (identical to train_retriever.py) ───────────────────
def group_split_by_node_id(df, train_ratio, val_ratio, seed):
    groups = df["positive_node_id"].astype(int).values
    uniq   = np.unique(groups)
    rng    = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n       = len(uniq)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio   * n)
    train_g = set(uniq[:n_train])
    val_g   = set(uniq[n_train:n_train + n_val])
    test_g  = set(uniq[n_train + n_val:])
    return (df[df["positive_node_id"].isin(train_g)].copy(),
            df[df["positive_node_id"].isin(val_g)].copy(),
            df[df["positive_node_id"].isin(test_g)].copy())

# ─── Metrics (identical to train_retriever.py) ───────────────────────────────
def metrics_from_ranks(ranks):
    ranks = list(map(int, ranks))
    rr    = [1.0 / r for r in ranks]
    out   = {
        "mrr":         float(np.mean(rr)),
        "mean_rank":   float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "ndcg@10":     float(np.mean([(1.0/np.log2(r+1)) if r<=10 else 0.0 for r in ranks])),
    }
    for k in RECALL_AT_K:
        out[f"recall@{k}"] = float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))
    return out

def bootstrap_ci(ranks, n_iter=BOOTSTRAP_N, seed=123):
    rng = np.random.RandomState(seed)
    n   = len(ranks)
    boot_mrr, boot_r1, boot_r5 = [], [], []
    for _ in range(n_iter):
        s = rng.choice(ranks, size=n, replace=True).tolist()
        m = metrics_from_ranks(s)
        boot_mrr.append(m["mrr"])
        boot_r1.append(m["recall@1"])
        boot_r5.append(m["recall@5"])
    ci = lambda a: {"lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}
    return {"mrr": ci(boot_mrr), "recall@1": ci(boot_r1), "recall@5": ci(boot_r5)}

# ─── Chunking strategies ─────────────────────────────────────────────────────
@dataclass
class ChunkConfig:
    name:     str
    strategy: str    # "word" | "char" | "sentence"
    size:     int = 256
    stride:   int = 64

def chunk_text(text, cfg):
    text = text.strip()
    if not text:
        return [text]

    if cfg.strategy == "sentence":
        parts = re.split(r"(?<=[।.!?])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts or [text]

    tokens = text.split() if cfg.strategy == "word" else list(text)
    if not tokens:
        return [text]

    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + cfg.size, len(tokens))
        chunks.append(("" if cfg.strategy == "char" else " ").join(tokens[start:end]))
        if end == len(tokens):
            break
        start += cfg.size - cfg.stride
    return chunks

CHUNK_ABLATIONS = [
    ChunkConfig("C0_word128",  "word",     128,  32),
    ChunkConfig("C1_word256",  "word",     256,  64),
    ChunkConfig("C2_word512",  "word",     512, 128),
    ChunkConfig("C3_char500",  "char",     500, 125),
    ChunkConfig("C4_char1000", "char",    1000, 250),
    ChunkConfig("C5_sentence", "sentence",  0,   0),
]

# ─── Build chunk corpus from nodes.csv ───────────────────────────────────────
def build_chunk_corpus(nodes_df, chunk_cfg):
    """
    For each node: join heading_path + content → chunk it.
    Each chunk inherits its parent node_id (used for gold matching).
    """
    chunk_texts, chunk_node_ids = [], []
    for _, row in nodes_df.iterrows():
        nid     = int(row["node_id"])
        heading = str(row.get("heading_path", "")).strip()
        content = str(row.get("content", "")).strip()
        text    = (heading + "\n" + content).strip() if heading else content
        for c in (chunk_text(text, chunk_cfg) or [text]):
            chunk_texts.append(c)
            chunk_node_ids.append(nid)
    return chunk_texts, chunk_node_ids

# ─── BM25 ranking (best-chunk-of-gold-node wins) ─────────────────────────────
def bm25_ranks_chunk(bm25, queries, gold_node_ids, chunk_node_ids, desc="BM25"):
    node2ci = {}
    for ci, nid in enumerate(chunk_node_ids):
        node2ci.setdefault(nid, []).append(ci)

    ranks = []
    for q, gnid in tqdm(zip(queries, gold_node_ids), total=len(queries), desc=desc):
        scores = bm25.get_scores(bm25_tokenize(q))
        order  = np.argsort(-scores, kind="mergesort")
        gold_set = set(node2ci.get(int(gnid), []))
        rank = next((pos+1 for pos, ci in enumerate(order) if ci in gold_set),
                    len(order)+1)
        ranks.append(rank)
    return ranks

# ─── Dense zero-shot ranking ─────────────────────────────────────────────────
@torch.no_grad()
def dense_ranks_chunk(model, tokenizer, queries, gold_node_ids,
                      chunk_texts, chunk_node_ids, device,
                      q_max=64, d_max=384, bs=64, desc="Dense"):
    model.eval()

    def encode(texts, max_len):
        embs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc   = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt")
            enc   = {k: v.to(device) for k, v in enc.items()}
            out   = model(**enc)
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            pool  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(1e-9)
            embs.append(F.normalize(pool, p=2, dim=1).cpu())
        return torch.cat(embs, 0)

    doc_embs = encode([f"passage: {t}" for t in chunk_texts], d_max)

    node2ci = {}
    for ci, nid in enumerate(chunk_node_ids):
        node2ci.setdefault(nid, []).append(ci)

    q_embs = encode([f"query: {q}" for q in queries], q_max)
    ranks  = []
    for qi in tqdm(range(len(queries)), desc=desc):
        scores   = (q_embs[qi].unsqueeze(0) @ doc_embs.t()).squeeze(0)
        order    = torch.argsort(scores, descending=True).tolist()
        gold_set = set(node2ci.get(int(gold_node_ids[qi]), []))
        rank     = next((pos+1 for pos, ci in enumerate(order) if ci in gold_set),
                        len(order)+1)
        ranks.append(rank)
    return ranks

# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    nodes_df = pd.read_csv(NODES_CSV)
    qa_df    = pd.read_csv(QA_GOLD_CSV)

    # Sanity check
    node_ids = set(nodes_df["node_id"].astype(int))
    missing  = set(qa_df["positive_node_id"].astype(int)) - node_ids
    assert not missing, f"Missing node_ids in nodes.csv: {sorted(missing)[:10]}"

    print(f"nodes={len(nodes_df)}  qa_pairs={len(qa_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model, tokenizer = None, None
    if not BM25_ONLY:
        print(f"Loading {MODEL_NAME} …")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        print("Model ready.\n")

    all_results  = []
    split_cache  = {}

    for chunk_cfg in CHUNK_ABLATIONS:
        print(f"\n{'─'*55}")
        print(f"{chunk_cfg.name}  strategy={chunk_cfg.strategy}  "
              f"size={chunk_cfg.size}  stride={chunk_cfg.stride}")

        chunk_texts, chunk_node_ids = build_chunk_corpus(nodes_df, chunk_cfg)
        avg_w = float(np.mean([len(t.split()) for t in chunk_texts]))
        print(f"  {len(chunk_texts)} chunks  (avg {avg_w:.1f} words)")

        bm25 = BM25Okapi([bm25_tokenize(t) for t in chunk_texts])

        for seed in SEEDS:
            set_seed(seed)

            if seed not in split_cache:
                _, _, test_df = group_split_by_node_id(qa_df, TRAIN_RATIO, VAL_RATIO, seed)
                split_cache[seed] = test_df
            test_df = split_cache[seed]

            queries   = test_df["question"].tolist()
            gold_nids = test_df["positive_node_id"].astype(int).tolist()

            # BM25
            bm25_r = bm25_ranks_chunk(bm25, queries, gold_nids, chunk_node_ids,
                                       desc=f"  BM25 {chunk_cfg.name} s{seed}")
            bm25_m  = metrics_from_ranks(bm25_r)
            bm25_ci = bootstrap_ci(bm25_r, seed=seed)

            all_results.append({
                "chunk_config": chunk_cfg.name, "retriever": "bm25",
                "seed": seed, "n_chunks": len(chunk_texts), "avg_words": round(avg_w, 1),
                **{f"bm25_{k}": v for k, v in bm25_m.items()},
                "bm25_ci": bm25_ci,
            })

            log = (f"  seed{seed}  BM25  MRR={bm25_m['mrr']:.4f}  "
                   f"R@1={bm25_m['recall@1']:.4f}  R@5={bm25_m['recall@5']:.4f}")

            # Dense
            if model is not None:
                dense_r = dense_ranks_chunk(model, tokenizer, queries, gold_nids,
                                            chunk_texts, chunk_node_ids, device,
                                            desc=f"  Dense {chunk_cfg.name} s{seed}")
                dense_m  = metrics_from_ranks(dense_r)
                dense_ci = bootstrap_ci(dense_r, seed=seed+1000)

                all_results.append({
                    "chunk_config": chunk_cfg.name, "retriever": "dense_zeroshot",
                    "seed": seed, "n_chunks": len(chunk_texts), "avg_words": round(avg_w, 1),
                    **{f"dense_{k}": v for k, v in dense_m.items()},
                    "dense_ci": dense_ci,
                })
                log += (f"  │  Dense  MRR={dense_m['mrr']:.4f}  "
                        f"R@1={dense_m['recall@1']:.4f}  R@5={dense_m['recall@5']:.4f}")
            print(log)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("CHUNK BASELINE  —  mean over seeds 42 / 43 / 44")
    print(f"{'='*72}")
    hdr = f"{'Config':<18} {'Retriever':<16} {'Chunks':>7} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'NDCG@10':>9}"
    print(hdr)
    print("─" * len(hdr))

    rows_by_key = {}
    for r in all_results:
        rows_by_key.setdefault((r["chunk_config"], r["retriever"]), []).append(r)

    summary_lines = []
    retrievers = ["bm25"] + (["dense_zeroshot"] if model else [])
    for cfg in CHUNK_ABLATIONS:
        for ret in retrievers:
            rows = rows_by_key.get((cfg.name, ret), [])
            if not rows:
                continue
            p  = "bm25_" if ret == "bm25" else "dense_"
            vals = lambda k: [r[f"{p}{k}"] for r in rows]
            mn = lambda k: np.mean(vals(k))
            sd = lambda k: np.std(vals(k))
            line = (f"{cfg.name:<18} {ret:<16} {rows[0]['n_chunks']:>7} "
                    f"{mn('mrr'):.4f}±{sd('mrr'):.4f} "
                    f"{mn('recall@1'):.4f}±{sd('recall@1'):.4f} "
                    f"{mn('recall@5'):.4f}±{sd('recall@5'):.4f} "
                    f"{mn('recall@10'):.4f}±{sd('recall@10'):.4f} "
                    f"{mn('ndcg@10'):.4f}±{sd('ndcg@10'):.4f}")
            print(line)
            summary_lines.append(line)

    # ── Save ──────────────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def serial(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: serial(v) for k, v in obj.items()}
        if isinstance(obj, list): return [serial(v) for v in obj]
        return obj

    results_path = f"{OUTPUT_DIR}/chunk_baseline_results_{stamp}.json"
    summary_path = f"{OUTPUT_DIR}/chunk_baseline_summary_{stamp}.txt"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serial(all_results), f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CHUNK BASELINE  —  mean over seeds 42 / 43 / 44\n")
        f.write("=" * 72 + "\n" + hdr + "\n" + "─" * len(hdr) + "\n")
        f.write("\n".join(summary_lines) + "\n\n")
        f.write("Gold match: rank = position of best chunk from the gold node.\n")
        f.write("Dense = zero-shot multilingual-e5-base (no fine-tuning).\n")
        f.write("Compare MRR/R@K here against hierarchy A0 (BM25) and A1 (Dense)\n")
        f.write("from train_retriever.py to confirm hierarchy advantage.\n")

    if CHUNKING_JSON_PATH:
        chunking_json_path = Path(CHUNKING_JSON_PATH)
    else:
        chunking_json_path = Path(__file__).resolve().parents[1] / "results" / "chunkingAb.json"
    chunking_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunking_json_path, "w", encoding="utf-8") as f:
        json.dump(serial(all_results), f, indent=2, ensure_ascii=False)

    print(f"\nResults → {results_path}")
    print(f"Summary → {summary_path}")
    print(f"Chunking JSON → {chunking_json_path}")

run()
