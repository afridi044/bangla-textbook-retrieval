
import os, re, json, random, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Bangla-friendly tokenizer for BM25
# ----------------------------
_BN_TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+", flags=re.UNICODE)

def bm25_tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return _BN_TOKEN_RE.findall(text)

# ----------------------------
# Stable argsort (tie-safe)
# ----------------------------
def stable_argsort_desc(x: np.ndarray) -> np.ndarray:
    return np.argsort(-x, kind="mergesort")

def stable_torch_argsort_desc(x: torch.Tensor) -> torch.Tensor:
    # torch argsort is stable in recent versions, but keep explicit behavior on CPU
    return torch.argsort(x, descending=True)

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # model
    model_name: str = "intfloat/multilingual-e5-base"
    use_e5_prefix: bool = True

    query_max_length: int = 64
    doc_max_length: int = 384

    # training
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    temperature: float = 0.05

    # negatives
    use_in_batch_negatives: bool = True
    num_random_negatives: int = 1
    bm25_mining_topk: int = 50
    num_bm25_hard_negatives: int = 2
    num_dense_hard_negatives: int = 2
    refresh_dense_hard_every: int = 2

    # data
    nodes_csv: str = "DatasetPrep/nodes.csv"
    qa_gold_csv: str = "DatasetPrep/qa_Gold.csv"
    train_ratio: float = 0.75
    val_ratio: float = 0.10

    # eval
    recall_at_k: Tuple[int, ...] = (1, 3, 5, 10, 20)
    doc_embed_batch_size: int = 64
    query_embed_batch_size: int = 64

    # hybrid
    hybrid_bm25_topk: int = 200
    rrf_k: int = 60

    # checkpoint protocol
    early_stopping_patience: int = 5
    # fixed alpha for HYBRID checkpointing when enabled (clean protocol)
    hybrid_alpha_ckpt: float = 0.5

    # alpha tuning (after training, once)
    tune_alpha_after_training: bool = True
    hybrid_alpha_grid: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # mining intersection
    mining_bm25_topk: int = 200
    mining_dense_topk: int = 200
    mining_keep: int = 50

    # bootstrap
    bootstrap_iters: int = 1000
    bootstrap_seed: int = 123

    # system
    output_root: str = "results/trained_retriever"
    stat_json_path: str = ""  # if empty, defaults to <project_root>/results/stat.json
    experiment_prefix: str = "biencoder_ablation_suite"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    # seeds
    seeds: Tuple[int, ...] = (42, 43, 44)

# ----------------------------
# Group-aware split by positive_node_id (node-disjoint)
# ----------------------------
def group_split_by_node_id(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    groups = df["positive_node_id"].astype(int).values
    uniq = np.unique(groups)

    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_groups = set(uniq[:n_train])
    val_groups   = set(uniq[n_train:n_train + n_val])
    test_groups  = set(uniq[n_train + n_val:])

    train_df = df[df["positive_node_id"].isin(train_groups)].copy()
    val_df   = df[df["positive_node_id"].isin(val_groups)].copy()
    test_df  = df[df["positive_node_id"].isin(test_groups)].copy()
    return train_df, val_df, test_df

# ----------------------------
# E5 formatting
# ----------------------------
def format_query(q: str, cfg: Config) -> str:
    q = str(q).strip()
    return f"query: {q}" if cfg.use_e5_prefix else q

def format_doc(d: str, cfg: Config) -> str:
    d = str(d).strip()
    return f"passage: {d}" if cfg.use_e5_prefix else d

def strip_e5_prefix_query(q: str) -> str:
    return str(q).replace("query:", "").strip()

def strip_e5_prefix_doc(d: str) -> str:
    return str(d).replace("passage:", "").strip()

# ----------------------------
# Model
# ----------------------------
class BiEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pooling(out.last_hidden_state, attention_mask)
        return F.normalize(emb, p=2, dim=1)

# ----------------------------
# Safe in-batch InfoNCE
# ----------------------------
def compute_contrastive_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: Optional[torch.Tensor],
    num_negatives: int,
    temperature: float,
    use_in_batch: bool,
    positive_ids: Optional[torch.Tensor] = None,
):
    bsz = q_emb.size(0)

    pos_scores = (q_emb * pos_emb).sum(dim=1) / temperature
    parts = [pos_scores.unsqueeze(1)]

    # explicit negatives
    if neg_emb is not None and num_negatives > 0:
        neg_emb = neg_emb.view(bsz, num_negatives, -1)
        neg_scores = torch.bmm(q_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1) / temperature
        parts.append(neg_scores)

    # safe in-batch negatives
    if use_in_batch and bsz > 1:
        sim = torch.matmul(q_emb, pos_emb.t()) / temperature
        eye = torch.eye(bsz, device=q_emb.device, dtype=torch.bool)
        mask = ~eye
        if positive_ids is not None:
            same = positive_ids.unsqueeze(1).eq(positive_ids.unsqueeze(0))
            mask = mask & (~same)

        row_vals = []
        maxn = 0
        for i in range(bsz):
            v = sim[i][mask[i]]
            row_vals.append(v)
            maxn = max(maxn, v.numel())

        if maxn > 0:
            padded = []
            for v in row_vals:
                if v.numel() == 0:
                    padded.append(torch.full((maxn,), -1e9, device=q_emb.device))
                else:
                    pad = maxn - v.numel()
                    if pad > 0:
                        v = torch.cat([v, torch.full((pad,), -1e9, device=q_emb.device)], dim=0)
                    padded.append(v)
            parts.append(torch.stack(padded, dim=0))

    logits = torch.cat(parts, dim=1)
    labels = torch.zeros(bsz, dtype=torch.long, device=q_emb.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc

# ----------------------------
# Dataset (NO chunking; nodes are your manual nodes)
# ----------------------------
class RetrievalDataset(Dataset):
    def __init__(self, qa_df: pd.DataFrame, nodes_df: pd.DataFrame, cfg: Config, split: str):
        self.qa_df = qa_df.reset_index(drop=True)
        self.cfg = cfg
        self.split = split

        # node texts
        self.node_texts: Dict[int, str] = {}
        for _, row in nodes_df.iterrows():
            nid = int(row["node_id"])
            heading = str(row.get("heading_path", "")).strip()
            content = str(row.get("content", "")).strip()
            text = (heading + "\n" + content).strip()
            self.node_texts[nid] = format_doc(text, cfg)

        self.all_node_ids = list(self.node_texts.keys())

        # bm25 hard list (fixed)
        self.bm25_hard: Dict[int, List[int]] = {}
        # dense hard list (refreshable)
        self.dense_hard: Dict[int, List[int]] = {}

        if split == "train":
            self._build_bm25()
            self._mine_bm25_hard()

    def _build_bm25(self):
        corpus_tokens = [bm25_tokenize(strip_e5_prefix_doc(self.node_texts[nid])) for nid in self.all_node_ids]
        self.bm25 = BM25Okapi(corpus_tokens)

    def _mine_bm25_hard(self):
        topk = self.cfg.bm25_mining_topk
        for idx in tqdm(range(len(self.qa_df)), desc="BM25 hard neg mining"):
            row = self.qa_df.iloc[idx]
            q_raw = str(row["question"])
            pos_id = int(row["positive_node_id"])
            scores = self.bm25.get_scores(bm25_tokenize(q_raw))
            top_idx = np.argsort(scores)[::-1][:topk]
            cand = []
            for i in top_idx:
                nid = self.all_node_ids[i]
                if nid != pos_id:
                    cand.append(nid)
            self.bm25_hard[idx] = cand

    def set_dense_hard(self, dense_hard: Dict[int, List[int]]):
        self.dense_hard = dense_hard

    def __len__(self):
        return len(self.qa_df)

    def __getitem__(self, idx):
        row = self.qa_df.iloc[idx]
        q = format_query(row["question"], self.cfg)
        pos_id = int(row["positive_node_id"])
        pos_text = self.node_texts[pos_id]

        neg_texts = []

        # BM25 hard negatives
        if self.split == "train" and self.cfg.num_bm25_hard_negatives > 0:
            hard = self.bm25_hard.get(idx, [])
            k = min(self.cfg.num_bm25_hard_negatives, len(hard))
            if k > 0:
                sampled = random.sample(hard, k)
                neg_texts.extend([self.node_texts[nid] for nid in sampled])

        # Dense hard negatives
        if self.split == "train" and self.cfg.num_dense_hard_negatives > 0:
            hard = self.dense_hard.get(idx, [])
            k = min(self.cfg.num_dense_hard_negatives, len(hard))
            if k > 0:
                sampled = random.sample(hard, k)
                neg_texts.extend([self.node_texts[nid] for nid in sampled])

        # Random negatives
        if self.split == "train" and self.cfg.num_random_negatives > 0:
            cand = [nid for nid in self.all_node_ids if nid != pos_id]
            k = min(self.cfg.num_random_negatives, len(cand))
            if k > 0:
                sampled = random.sample(cand, k)
                neg_texts.extend([self.node_texts[nid] for nid in sampled])

        return {"question": q, "positive": pos_text, "positive_id": pos_id, "negatives": neg_texts}

# ----------------------------
# Collate
# ----------------------------
def collate_fn(batch, tokenizer, q_max_len, d_max_len):
    questions = [x["question"] for x in batch]
    positives = [x["positive"] for x in batch]
    pos_ids = [int(x["positive_id"]) for x in batch]

    all_negs = []
    for x in batch:
        all_negs.extend(x["negatives"])
    num_negs = len(batch[0]["negatives"]) if len(batch) > 0 else 0

    q_enc = tokenizer(questions, padding=True, truncation=True, max_length=q_max_len, return_tensors="pt")
    p_enc = tokenizer(positives, padding=True, truncation=True, max_length=d_max_len, return_tensors="pt")

    n_enc = None
    if len(all_negs) > 0:
        n_enc = tokenizer(all_negs, padding=True, truncation=True, max_length=d_max_len, return_tensors="pt")

    return {
        "question_input": q_enc,
        "positive_input": p_enc,
        "negative_input": n_enc,
        "num_negatives_per_query": num_negs,
        "positive_ids": torch.tensor(pos_ids, dtype=torch.long),
    }

def move_batch_to_device(batch, device):
    batch["question_input"] = {k: v.to(device) for k, v in batch["question_input"].items()}
    batch["positive_input"] = {k: v.to(device) for k, v in batch["positive_input"].items()}
    if batch["negative_input"] is not None:
        batch["negative_input"] = {k: v.to(device) for k, v in batch["negative_input"].items()}
    batch["positive_ids"] = batch["positive_ids"].to(device)
    return batch

# ----------------------------
# Encoding
# ----------------------------
@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device: str, max_length: int, batch_size: int):
    model.eval()
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        e = model(**enc).detach().cpu()
        embs.append(e)
    return torch.cat(embs, dim=0)

# ----------------------------
# Metrics from ranks (single-relevant)
# ----------------------------
def metrics_from_ranks(ranks: List[int], recall_at_k: Tuple[int, ...]) -> Dict[str, float]:
    ranks = list(map(int, ranks))
    rr = [1.0 / r for r in ranks] if ranks else []
    out = {
        "mrr": float(np.mean(rr)) if rr else 0.0,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "median_rank": float(np.median(ranks)) if ranks else 0.0,
        "ndcg@10_single": float(np.mean([(1.0 / np.log2(r + 1)) if r <= 10 else 0.0 for r in ranks])) if ranks else 0.0,
    }
    for k in recall_at_k:
        out[f"recall@{k}"] = float(np.mean([1.0 if r <= k else 0.0 for r in ranks])) if ranks else 0.0
    return out

# ----------------------------
# Tie-safe rank computation: BM25 / Dense / Hybrid
# ----------------------------
def build_bm25_index_from_raw_docs(all_node_texts_raw: List[str]):
    tokenized = [bm25_tokenize(t) for t in all_node_texts_raw]
    return BM25Okapi(tokenized)

def bm25_ranks(
    bm25: BM25Okapi,
    queries_raw: List[str],
    gold_ids: List[int],
    all_node_ids: List[int],
    desc: str,
) -> List[int]:
    node_id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    ranks = []
    for q, gold in tqdm(list(zip(queries_raw, gold_ids)), total=len(queries_raw), desc=desc):
        scores = bm25.get_scores(bm25_tokenize(q))
        gold_idx = node_id_to_idx[int(gold)]
        order = stable_argsort_desc(np.asarray(scores, dtype=np.float32))
        rank = int(np.where(order == gold_idx)[0][0]) + 1
        ranks.append(rank)
    return ranks

@torch.no_grad()
def dense_ranks(
    model,
    tokenizer,
    queries_fmt: List[str],
    gold_ids: List[int],
    all_node_ids: List[int],
    all_node_texts_fmt: List[str],
    cfg: Config,
    desc: str,
) -> List[int]:
    doc_embs = encode_texts(model, tokenizer, all_node_texts_fmt, cfg.device, cfg.doc_max_length, cfg.doc_embed_batch_size)
    node_id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    ranks = []
    bs = cfg.query_embed_batch_size
    for i in tqdm(range(0, len(queries_fmt), bs), desc=desc):
        q_batch = queries_fmt[i:i+bs]
        g_batch = gold_ids[i:i+bs]
        q_emb = encode_texts(model, tokenizer, q_batch, cfg.device, cfg.query_max_length, len(q_batch))
        scores = torch.matmul(q_emb, doc_embs.t())  # [B,N]

        for j in range(len(q_batch)):
            gold_idx = node_id_to_idx[int(g_batch[j])]
            order = stable_torch_argsort_desc(scores[j])
            rank = int((order == gold_idx).nonzero(as_tuple=False).item()) + 1
            ranks.append(rank)
    return ranks

def _minmax_norm(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

@torch.no_grad()
def hybrid_ranks(
    model,
    tokenizer,
    bm25: BM25Okapi,
    fusion: str,  # "rrf" or "minmax"
    queries_fmt: List[str],
    queries_raw: List[str],
    gold_ids: List[int],
    all_node_ids: List[int],
    all_node_texts_fmt: List[str],
    all_node_texts_raw: List[str],
    cfg: Config,
    alpha: float,
    desc: str,
) -> List[int]:
    """
    Candidate gen: BM25 topK
    Dense rerank: dense scores for candidates only
    Fusion:
      - rrf: alpha*RRF_dense + (1-alpha)*RRF_bm25
      - minmax: alpha*dense + (1-alpha)*bm25_norm (candidate-local)
    """
    node_id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    doc_embs = encode_texts(model, tokenizer, all_node_texts_fmt, cfg.device, cfg.doc_max_length, cfg.doc_embed_batch_size)

    ranks = []
    bs = cfg.query_embed_batch_size
    for i in tqdm(range(0, len(queries_fmt), bs), desc=desc):
        qfmt_batch = queries_fmt[i:i+bs]
        qraw_batch = queries_raw[i:i+bs]
        gold_batch = gold_ids[i:i+bs]
        q_emb = encode_texts(model, tokenizer, qfmt_batch, cfg.device, cfg.query_max_length, len(qfmt_batch))

        for j in range(len(qfmt_batch)):
            gold_id = int(gold_batch[j])
            gold_idx = node_id_to_idx[gold_id]

            bm25_scores_full = np.asarray(bm25.get_scores(bm25_tokenize(qraw_batch[j])), dtype=np.float32)
            cand_idxs = np.argsort(bm25_scores_full)[::-1][:min(cfg.hybrid_bm25_topk, len(all_node_ids))].tolist()

            # if gold isn't in candidate set, rank is >K (consistent across hybrid variants)
            if gold_idx not in cand_idxs:
                ranks.append(len(cand_idxs) + 1)
                continue

            dense_scores = torch.matmul(q_emb[j:j+1], doc_embs[cand_idxs].t()).squeeze(0).numpy().astype(np.float32)

            if fusion == "rrf":
                # ranks within candidates
                dense_order = stable_argsort_desc(dense_scores)
                rank_dense = {cand_idxs[pos]: r for r, pos in enumerate(dense_order, start=1)}

                bm25_scores_cand = np.array([bm25_scores_full[di] for di in cand_idxs], dtype=np.float32)
                bm25_order = stable_argsort_desc(bm25_scores_cand)
                rank_bm25 = {cand_idxs[pos]: r for r, pos in enumerate(bm25_order, start=1)}

                fused = []
                for di in cand_idxs:
                    rd, rb = rank_dense[di], rank_bm25[di]
                    s = alpha * (1.0 / (cfg.rrf_k + rd)) + (1.0 - alpha) * (1.0 / (cfg.rrf_k + rb))
                    fused.append(s)
                fused = np.asarray(fused, dtype=np.float32)
                order = stable_argsort_desc(fused)
                ranked = [cand_idxs[pos] for pos in order]
                rank = ranked.index(gold_idx) + 1
                ranks.append(rank)

            elif fusion == "minmax":
                bm25_scores_cand = np.array([bm25_scores_full[di] for di in cand_idxs], dtype=np.float32)
                b_norm = _minmax_norm(bm25_scores_cand)
                fused = alpha * dense_scores + (1.0 - alpha) * b_norm
                order = stable_argsort_desc(fused)
                ranked = [cand_idxs[pos] for pos in order]
                rank = ranked.index(gold_idx) + 1
                ranks.append(rank)

            else:
                raise ValueError(f"Unknown fusion: {fusion}")

    return ranks

# ----------------------------
# Mining: BM25 ∩ Dense (refreshable)
# ----------------------------
@torch.no_grad()
def mine_dense_hard_negatives_intersection(
    train_df: pd.DataFrame,
    all_node_ids: List[int],
    all_node_texts_fmt: List[str],
    all_node_texts_raw: List[str],
    model,
    tokenizer,
    cfg: Config,
):
    print("Mining BM25∩Dense hard negatives with current model...")
    model.eval()

    bm25 = build_bm25_index_from_raw_docs(all_node_texts_raw)
    doc_embs = encode_texts(model, tokenizer, all_node_texts_fmt, cfg.device, cfg.doc_max_length, cfg.doc_embed_batch_size)

    node_id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    idx_to_node_id = {i: nid for i, nid in enumerate(all_node_ids)}

    queries_fmt = [format_query(q, cfg) for q in train_df["question"].tolist()]
    queries_raw = [str(q) for q in train_df["question"].tolist()]
    gold_ids = train_df["positive_node_id"].astype(int).tolist()

    dense_hard: Dict[int, List[int]] = {}

    bs = cfg.query_embed_batch_size
    for i in tqdm(range(0, len(queries_fmt), bs), desc="Mining (BM25∩Dense)"):
        qfmt_batch = queries_fmt[i:i+bs]
        qraw_batch = queries_raw[i:i+bs]
        gold_batch = gold_ids[i:i+bs]

        q_emb = encode_texts(model, tokenizer, qfmt_batch, cfg.device, cfg.query_max_length, len(qfmt_batch))
        scores = torch.matmul(q_emb, doc_embs.t())  # [B,N]

        for j in range(len(qfmt_batch)):
            q_idx = i + j
            gold_id = int(gold_batch[j])
            gold_doc_idx = node_id_to_idx[gold_id]

            d_top = torch.topk(scores[j], k=min(cfg.mining_dense_topk, scores.size(1)), largest=True).indices.tolist()
            d_top = [di for di in d_top if di != gold_doc_idx]

            bm25_scores = np.asarray(bm25.get_scores(bm25_tokenize(qraw_batch[j])), dtype=np.float32)
            b_top = np.argsort(bm25_scores)[::-1][:min(cfg.mining_bm25_topk, len(all_node_ids))].tolist()
            b_top = [bi for bi in b_top if bi != gold_doc_idx]
            b_set = set(b_top)

            inter = [di for di in d_top if di in b_set]  # keep dense order
            filled = inter[:]
            if len(filled) < cfg.mining_keep:
                for di in d_top:
                    if di not in filled:
                        filled.append(di)
                    if len(filled) >= cfg.mining_keep:
                        break

            dense_hard[q_idx] = [idx_to_node_id[di] for di in filled[:cfg.mining_keep]]

    return dense_hard

# ----------------------------
# Alpha tuning (ONCE on VAL after training)
# ----------------------------
def tune_alpha_once_on_val(
    model,
    tokenizer,
    bm25: BM25Okapi,
    fusion: str,
    val_queries_fmt,
    val_queries_raw,
    val_gold,
    all_node_ids,
    all_node_texts_fmt,
    all_node_texts_raw,
    cfg: Config,
):
    best_a, best_mrr = None, -1e9
    best_metrics = None
    for a in cfg.hybrid_alpha_grid:
        ranks = hybrid_ranks(
            model, tokenizer, bm25, fusion,
            val_queries_fmt, val_queries_raw, val_gold,
            all_node_ids, all_node_texts_fmt, all_node_texts_raw,
            cfg,
            alpha=float(a),
            desc=f"VAL hybrid ranks tune alpha={a} ({fusion})",
        )
        m = metrics_from_ranks(ranks, cfg.recall_at_k)
        if m["mrr"] > best_mrr:
            best_mrr = m["mrr"]
            best_a = float(a)
            best_metrics = m
    return best_a, best_metrics

# ----------------------------
# Bootstrap CIs
# ----------------------------
def bootstrap_ci_from_ranks(
    ranks: List[int],
    recall_at_k: Tuple[int, ...],
    iters: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    n = len(ranks)
    ranks = np.asarray(ranks, dtype=np.int32)

    keys = ["mrr", "ndcg@10_single"] + [f"recall@{x}" for x in recall_at_k]
    samples = {k: [] for k in keys}

    for _ in range(iters):
        idx = rng.randint(0, n, size=n)
        samp = ranks[idx].tolist()
        m = metrics_from_ranks(samp, recall_at_k)
        for k in keys:
            samples[k].append(m[k])

    out = {}
    for k, vals in samples.items():
        vals = np.asarray(vals, dtype=np.float32)
        out[k] = {
            "mean": float(np.mean(vals)),
            "ci95_low": float(np.percentile(vals, 2.5)),
            "ci95_high": float(np.percentile(vals, 97.5)),
        }
    return out

def bootstrap_delta_ci_mrr(ranks_a: List[int], ranks_b: List[int], iters: int, seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    ra = np.asarray(ranks_a, dtype=np.float32)
    rb = np.asarray(ranks_b, dtype=np.float32)
    n = len(ra)
    deltas = []
    for _ in range(iters):
        idx = rng.randint(0, n, size=n)
        d = float(np.mean(1.0 / ra[idx] - 1.0 / rb[idx]))
        deltas.append(d)
    deltas = np.asarray(deltas, dtype=np.float32)
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
    }

# ----------------------------
# Checkpoint I/O
# ----------------------------
def save_best_checkpoint(out_dir: Path, epoch: int, model: nn.Module, tokenizer, meta: Dict[str, Any], cfg: Config):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model_state_dict.pt")
    tokenizer.save_pretrained(out_dir)
    payload = {"epoch": int(epoch), "meta": meta, "config": asdict(cfg)}
    with open(out_dir / "best_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def load_best_checkpoint(out_dir: Path, model: nn.Module, device: str):
    state = torch.load(out_dir / "best_model_state_dict.pt", map_location=device)
    model.load_state_dict(state)
    meta_path = out_dir / "best_model_meta.json"
    if meta_path.exists():
        return json.load(open(meta_path, "r", encoding="utf-8"))
    return None

# ============================================================
# Ablation variant spec
# ============================================================
@dataclass
class Ablation:
    name: str
    train_dense: bool
    hybrid_eval: bool
    fusion: str                 # "rrf" or "minmax"
    mining_refresh: bool
    ckpt_metric: str            # "dense" or "hybrid"
    ckpt_alpha_fixed: float     # used only if ckpt_metric=="hybrid"
    tune_alpha_after: bool

def ablation_suite(cfg: Config) -> List[Ablation]:
    return [
        Ablation("A0_BM25", train_dense=False, hybrid_eval=False, fusion="rrf",
                 mining_refresh=False, ckpt_metric="dense", ckpt_alpha_fixed=cfg.hybrid_alpha_ckpt,
                 tune_alpha_after=False),

        Ablation("A1_DENSE", train_dense=True, hybrid_eval=False, fusion="rrf",
                 mining_refresh=False, ckpt_metric="dense", ckpt_alpha_fixed=cfg.hybrid_alpha_ckpt,
                 tune_alpha_after=False),

        Ablation("A2_DENSE+HYBRID_RRF_eval", train_dense=True, hybrid_eval=True, fusion="rrf",
                 mining_refresh=False, ckpt_metric="dense", ckpt_alpha_fixed=cfg.hybrid_alpha_ckpt,
                 tune_alpha_after=True),

        Ablation("A3_A2+MINING_refresh", train_dense=True, hybrid_eval=True, fusion="rrf",
                 mining_refresh=True, ckpt_metric="dense", ckpt_alpha_fixed=cfg.hybrid_alpha_ckpt,
                 tune_alpha_after=True),

        Ablation("A4_A3+HYBRID_ckpt", train_dense=True, hybrid_eval=True, fusion="rrf",
                 mining_refresh=True, ckpt_metric="hybrid", ckpt_alpha_fixed=cfg.hybrid_alpha_ckpt,
                 tune_alpha_after=True),

        Ablation("A5_A3+FUSION_minmax", train_dense=True, hybrid_eval=True, fusion="minmax",
                 mining_refresh=True, ckpt_metric="dense", ckpt_alpha_fixed=cfg.hybrid_alpha_ckpt,
                 tune_alpha_after=True),
    ]

# ============================================================
# Run one seed for one ablation
# ============================================================
def run_one_seed_one_ablation(
    cfg: Config,
    seed: int,
    ab: Ablation,
    nodes_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    split_cache: Dict[int, Dict[str, pd.DataFrame]],
) -> Dict[str, Any]:
    set_seed(seed)

    # Reuse the exact same split across all ablations for this seed.
    if seed not in split_cache:
        train_df, val_df, test_df = group_split_by_node_id(qa_df, cfg.train_ratio, cfg.val_ratio, seed)
        split_cache[seed] = {"train": train_df, "val": val_df, "test": test_df}
    else:
        train_df = split_cache[seed]["train"]
        val_df = split_cache[seed]["val"]
        test_df = split_cache[seed]["test"]

    # output dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = f"{cfg.experiment_prefix}/{ab.name}_seed{seed}_{stamp}"
    out_dir = Path(cfg.output_root) / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save split CSVs for reproducibility and auditability.
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)

    # node texts
    tmp_ds = RetrievalDataset(train_df, nodes_df, cfg, split="train")
    all_node_ids = list(tmp_ds.node_texts.keys())
    all_node_texts_fmt = [tmp_ds.node_texts[nid] for nid in all_node_ids]
    all_node_texts_raw = [strip_e5_prefix_doc(t) for t in all_node_texts_fmt]

    # queries
    val_queries_fmt = [format_query(q, cfg) for q in val_df["question"].tolist()]
    val_queries_raw = [strip_e5_prefix_query(q) for q in val_queries_fmt]
    val_gold = val_df["positive_node_id"].astype(int).tolist()

    test_queries_fmt = [format_query(q, cfg) for q in test_df["question"].tolist()]
    test_queries_raw = [strip_e5_prefix_query(q) for q in test_queries_fmt]
    test_gold = test_df["positive_node_id"].astype(int).tolist()

    # BM25 index (raw)
    bm25 = build_bm25_index_from_raw_docs(all_node_texts_raw)

    # Always compute BM25 baseline ranks+metrics on TEST for every ablation (same reference)
    bm25_test_ranks = bm25_ranks(bm25, test_queries_raw, test_gold, all_node_ids, desc=f"{ab.name} | BM25 ranks TEST")
    bm25_test_metrics = metrics_from_ranks(bm25_test_ranks, cfg.recall_at_k)
    bm25_test_metrics = {f"bm25_{k}": v for k, v in bm25_test_metrics.items()}

    # If this ablation does not train dense, finish here (A0)
    if not ab.train_dense:
        ci = {
            "bm25": bootstrap_ci_from_ranks(bm25_test_ranks, cfg.recall_at_k, cfg.bootstrap_iters, cfg.bootstrap_seed + seed),
        }
        payload = {
            "ablation": asdict(ab),
            "seed": seed,
            "metrics": bm25_test_metrics,
            "bootstrap_ci": ci,
        }
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        flat = {"seed": seed, "ablation": ab.name, **bm25_test_metrics}
        return {"flat": flat, "stat_row": payload}

    # ----------------------------
    # Train dense model
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = BiEncoder(cfg.model_name).to(cfg.device)

    train_ds = RetrievalDataset(train_df, nodes_df, cfg, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg.query_max_length, cfg.doc_max_length),
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = (len(train_loader) * cfg.num_epochs) // cfg.gradient_accumulation_steps
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda", enabled=(cfg.device == "cuda"))

    writer = SummaryWriter(str(out_dir / "logs"))
    train_ds.set_dense_hard({})

    best_val = -1e9
    patience = 0

    # fixed alpha used ONLY when checkpointing on hybrid
    ckpt_alpha = float(ab.ckpt_alpha_fixed)

    for epoch in range(cfg.num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        ep_loss, ep_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"{ab.name} | seed {seed} | Epoch {epoch+1}/{cfg.num_epochs}")

        for bi, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, cfg.device)

            with autocast("cuda", enabled=(cfg.device == "cuda")):
                q_emb = model(**batch["question_input"])
                p_emb = model(**batch["positive_input"])
                n_emb = model(**batch["negative_input"]) if batch["negative_input"] is not None else None

                loss, acc = compute_contrastive_loss(
                    q_emb, p_emb, n_emb,
                    num_negatives=batch["num_negatives_per_query"],
                    temperature=cfg.temperature,
                    use_in_batch=cfg.use_in_batch_negatives,
                    positive_ids=batch["positive_ids"],
                )
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (bi + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            ep_loss += float(loss.item()) * cfg.gradient_accumulation_steps
            ep_acc += float(acc.item())
            pbar.set_postfix(loss=f"{(loss.item()*cfg.gradient_accumulation_steps):.4f}", acc=f"{acc.item():.3f}")

        writer.add_scalar("Loss/train", ep_loss / max(len(train_loader), 1), epoch)
        writer.add_scalar("Acc/train", ep_acc / max(len(train_loader), 1), epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # optional mining refresh (ablated)
        if ab.mining_refresh:
            if (epoch + 1) >= 2 and cfg.num_dense_hard_negatives > 0 and ((epoch + 1) % cfg.refresh_dense_hard_every == 0):
                dense_hard = mine_dense_hard_negatives_intersection(
                    train_df, all_node_ids, all_node_texts_fmt, all_node_texts_raw,
                    model, tokenizer, cfg
                )
                train_ds.set_dense_hard(dense_hard)

        # ---------- VAL evaluation for checkpoint ----------
        # dense ranks always computed for logging + some ablations
        val_dense_r = dense_ranks(
            model, tokenizer, val_queries_fmt, val_gold, all_node_ids, all_node_texts_fmt, cfg,
            desc=f"{ab.name} | VAL dense ranks"
        )
        val_dense_m = metrics_from_ranks(val_dense_r, cfg.recall_at_k)
        for k, v in val_dense_m.items():
            writer.add_scalar(f"ValDense/{k}", v, epoch)

        if ab.ckpt_metric == "dense":
            current = float(val_dense_m["mrr"])
            ckpt_payload = {"val_dense": val_dense_m}
        elif ab.ckpt_metric == "hybrid":
            # hybrid checkpoint metric uses FIXED alpha (NO tuning during training)
            val_hyb_r = hybrid_ranks(
                model, tokenizer, bm25, ab.fusion,
                val_queries_fmt, val_queries_raw, val_gold,
                all_node_ids, all_node_texts_fmt, all_node_texts_raw, cfg,
                alpha=ckpt_alpha,
                desc=f"{ab.name} | VAL hybrid ranks (ckpt, alpha={ckpt_alpha}, {ab.fusion})"
            )
            val_hyb_m = metrics_from_ranks(val_hyb_r, cfg.recall_at_k)
            current = float(val_hyb_m["mrr"])
            ckpt_payload = {"val_dense": val_dense_m, "val_hybrid_ckpt": {**val_hyb_m, "alpha": ckpt_alpha, "fusion": ab.fusion}}
            for k, v in val_hyb_m.items():
                writer.add_scalar(f"ValHybridCkpt/{k}", v, epoch)
        else:
            raise ValueError("ckpt_metric must be 'dense' or 'hybrid'")

        print(f"\n{ab.name} seed{seed} epoch{epoch+1}: ckpt_metric={ab.ckpt_metric} value={current:.4f}")

        if current > best_val:
            best_val = current
            patience = 0
            save_best_checkpoint(out_dir, epoch, model, tokenizer, {"ablation": asdict(ab), "seed": seed, **ckpt_payload}, cfg)
            print("✅ saved best")
        else:
            patience += 1
            print(f"no improvement | patience {patience}/{cfg.early_stopping_patience}")
            if patience >= cfg.early_stopping_patience:
                print("🛑 early stopping")
                break

    # load best
    load_best_checkpoint(out_dir, model, cfg.device)

    # ----------------------------
    # TEST evaluation
    # ----------------------------
    test_dense_r = dense_ranks(
        model, tokenizer, test_queries_fmt, test_gold, all_node_ids, all_node_texts_fmt, cfg,
        desc=f"{ab.name} | TEST dense ranks"
    )
    test_dense_m = metrics_from_ranks(test_dense_r, cfg.recall_at_k)

    hyb_test_m = {}
    hyb_test_r = None
    tuned_alpha = None

    if ab.hybrid_eval:
        # tune alpha ONCE on VAL (clean protocol), then use on TEST
        alpha_for_test = ckpt_alpha
        if ab.tune_alpha_after and cfg.tune_alpha_after_training:
            tuned_alpha, tuned_val = tune_alpha_once_on_val(
                model, tokenizer, bm25, ab.fusion,
                val_queries_fmt, val_queries_raw, val_gold,
                all_node_ids, all_node_texts_fmt, all_node_texts_raw, cfg
            )
            alpha_for_test = float(tuned_alpha)
            print(f"{ab.name} seed{seed}: tuned_alpha={alpha_for_test} (VAL mrr={tuned_val['mrr']:.4f})")

        hyb_test_r = hybrid_ranks(
            model, tokenizer, bm25, ab.fusion,
            test_queries_fmt, test_queries_raw, test_gold,
            all_node_ids, all_node_texts_fmt, all_node_texts_raw, cfg,
            alpha=float(alpha_for_test),
            desc=f"{ab.name} | TEST hybrid ranks (alpha={alpha_for_test}, {ab.fusion})"
        )
        hyb_test_m = metrics_from_ranks(hyb_test_r, cfg.recall_at_k)
        hyb_test_m = {f"hybrid_{k}": v for k, v in hyb_test_m.items()}
        hyb_test_m["hybrid_alpha"] = float(alpha_for_test)
        hyb_test_m["hybrid_fusion"] = 0.0 if ab.fusion == "rrf" else 1.0  # numeric for aggregation

    # bootstrap CIs (per seed)
    ci = {
        "bm25": bootstrap_ci_from_ranks(bm25_test_ranks, cfg.recall_at_k, cfg.bootstrap_iters, cfg.bootstrap_seed + seed),
        "dense": bootstrap_ci_from_ranks(test_dense_r, cfg.recall_at_k, cfg.bootstrap_iters, cfg.bootstrap_seed + seed + 1000),
    }
    if hyb_test_r is not None:
        ci["hybrid"] = bootstrap_ci_from_ranks(hyb_test_r, cfg.recall_at_k, cfg.bootstrap_iters, cfg.bootstrap_seed + seed + 2000)
        ci["delta_hybrid_minus_bm25_mrr"] = bootstrap_delta_ci_mrr(hyb_test_r, bm25_test_ranks, cfg.bootstrap_iters, cfg.bootstrap_seed + seed + 3000)

    # save outputs
    payload = {
        "ablation": asdict(ab),
        "seed": seed,
        "metrics": {
            **bm25_test_metrics,
            **test_dense_m,
            **hyb_test_m,
        },
        "bootstrap_ci": ci,
        "tuned_alpha": tuned_alpha,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    writer.close()

    # return for multi-seed aggregation
    flat = {"seed": seed, "ablation": ab.name}
    flat.update(bm25_test_metrics)
    flat.update(test_dense_m)
    flat.update(hyb_test_m)
    return {"flat": flat, "stat_row": payload}

# ============================================================
# Aggregation
# ============================================================
def aggregate_metrics(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    # numeric keys only
    keys = sorted({k for r in runs for k in r.keys() if isinstance(r.get(k), (int, float))})
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in runs if k in r and isinstance(r[k], (int, float))]
        if len(vals) == 0:
            continue
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out

def print_compact_table(all_runs: List[Dict[str, float]]):
    # concentrated: only headline metrics
    cols = [
        "bm25_mrr", "bm25_recall@1", "bm25_recall@5",
        "mrr", "recall@1", "recall@5",                      # dense
        "hybrid_mrr", "hybrid_recall@1", "hybrid_recall@5",  # hybrid if present
    ]
    ablations = sorted(list(set([r["ablation"] for r in all_runs])))
    print("\n=== Ablation Summary (mean ± std over seeds) ===")
    for ab in ablations:
        runs = [r for r in all_runs if r["ablation"] == ab]
        agg = aggregate_metrics(runs)
        print(f"\n[{ab}]")
        for k in cols:
            if k in agg:
                print(f"  {k:16s}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

# ============================================================
# Main: run full ablation suite
# ============================================================
def run():
    cfg = Config()
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)

    # Default stat.json location for results/table.py
    if cfg.stat_json_path:
        stat_json_path = Path(cfg.stat_json_path)
    else:
        stat_json_path = Path(__file__).resolve().parents[1] / "results" / "stat.json"
    stat_json_path.parent.mkdir(parents=True, exist_ok=True)

    nodes_df = pd.read_csv(cfg.nodes_csv)
    qa_df = pd.read_csv(cfg.qa_gold_csv)

    node_ids = set(nodes_df["node_id"].astype(int).tolist())
    qa_pos_ids = qa_df["positive_node_id"].astype(int)
    missing_ids = sorted(set(qa_pos_ids.tolist()) - node_ids)
    if missing_ids:
        preview = missing_ids[:20]
        raise ValueError(
            "qa_gold.csv contains positive_node_id values not found in nodes.csv. "
            f"Missing count={len(missing_ids)}, sample={preview}"
        )

    suite = ablation_suite(cfg)

    all_runs: List[Dict[str, float]] = []
    stat_rows: List[Dict[str, Any]] = []
    split_cache: Dict[int, Dict[str, pd.DataFrame]] = {}

    for ab in suite:
        for seed in cfg.seeds:
            print(f"\n\n============================")
            print(f"Running {ab.name} | seed={seed}")
            print(f"train_dense={ab.train_dense}, hybrid_eval={ab.hybrid_eval}, fusion={ab.fusion}, mining_refresh={ab.mining_refresh}, ckpt={ab.ckpt_metric}")
            print(f"============================\n")

            run_out = run_one_seed_one_ablation(cfg, seed, ab, nodes_df, qa_df, split_cache)
            all_runs.append(run_out["flat"])
            stat_rows.append(run_out["stat_row"])

    # Save global summary
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(cfg.output_root) / f"SUMMARY_{cfg.experiment_prefix}_{stamp}.json"

    # per-ablation aggregation
    per_ab = {}
    for ab in sorted(list(set([r["ablation"] for r in all_runs]))):
        runs = [r for r in all_runs if r["ablation"] == ab]
        per_ab[ab] = {"aggregate": aggregate_metrics(runs), "runs": runs}

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "per_ablation": per_ab}, f, indent=2, ensure_ascii=False)

    with open(stat_json_path, "w", encoding="utf-8") as f:
        json.dump(stat_rows, f, indent=2, ensure_ascii=False)

    print_compact_table(all_runs)
    print(f"\nSaved global summary to: {summary_path}")
    print(f"Saved stat rows to: {stat_json_path}")

# Run it
run()
