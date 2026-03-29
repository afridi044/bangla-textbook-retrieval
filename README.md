# A Controlled Study of Bangla Biology Textbook Retrieval
## Diagnosing Unit Design, Dense Supervision, and Hybrid Fusion

This repository contains the code, preprocessing pipeline, and evaluation scripts for a controlled study of **structure-aware retrieval for Bangla biology textbooks**. The project studies retrieval as a core component for Bangla educational QA and teacher-support systems, focusing on how **retrieval-unit design**, **dense supervision**, and **sparse-dense fusion** affect evidence retrieval over structured textbook content.

Unlike open-domain retrieval, textbook evidence follows chapter hierarchy, pedagogical boundaries, and curriculum-aligned organization. This repository provides a reproducible workflow for building hierarchy-derived retrieval nodes, training dense retrievers, running controlled ablations, comparing chunking strategies, and generating publication-ready result summaries.

---

## Overview

The benchmark is built from **3 Bangla biology chapters**, **166 hierarchy-derived retrieval nodes**, and **1,320 question-node pairs**. The task is **single-positive retrieval**: for each question, rank all nodes so that the gold supporting node appears as high as possible.

The main experimental components in this repository are:

- **Hierarchy-derived node construction** from textbook chapter structure
- **BM25 lexical retrieval** over Bangla text
- **Dense retrieval** with `intfloat/multilingual-e5-base`
- **Hard-negative mining and refresh** during dense training
- **Hybrid fusion** using **RRF** and **MinMax**
- **Chunking baselines** with word, character, and sentence units
- **Multi-seed node-disjoint evaluation** and result aggregation

---

## Benchmark at a Glance

- **Domain:** Bangla biology textbooks
- **Chapters:** 3
- **Retrieval nodes:** 166
- **Question-node pairs:** 1,320
- **Split protocol:** node-disjoint train/validation/test
- **Seeds:** 42, 43, 44
- **Evaluation focus:** MRR, Recall@K, NDCG@10, Mean Rank

### Included textbook chapters

- **Algae and Fungi**
- **Bryophyta and Pteridophyta**
- **Gymnosperms and Angiosperms**

---

## Main Experimental Design

The repository supports the controlled ablation ladder described in the paper:

- **A0:** BM25
- **A1:** Dense retrieval
- **A2:** Dense + RRF fusion
- **A3:** A2 + hard-negative refresh
- **A4:** A3 + checkpoint selection by hybrid validation MRR
- **A5:** A3 + MinMax fusion

It also supports the **chunking study**, where hierarchy-preserving nodes are compared against flat chunk alternatives:

- **Word chunks:** 128, 256, 512
- **Character chunks:** 500, 1000
- **Sentence chunks**

---

## Repository Structure

```text
.
├── Processing/
│   ├── parse_md.py
│   ├── *.md
│   └── *.json
├── DatasetPrep/
│   ├── prepare_nodes.py
│   ├── qa_Gold.csv
│   ├── nodes.csv
│   └── nodes.jsonl
├── training/
│   ├── train_retriever.py
│   └── chunkingAb.py
└── results/
    ├── table.py
    ├── stat.json
    └── chunkingAb.json
```

### Directory details

#### `Processing/`
Utilities for converting textbook markdown chapters into hierarchical JSON trees.

- `parse_md.py`: parses markdown headings and content into nested JSON
- `*.md`: source chapter files
- `*.json`: parsed chapter trees

#### `DatasetPrep/`
Builds the hierarchy-derived retrieval corpus used by the benchmark.

- `prepare_nodes.py`: flattens chapter JSON into retrieval-ready node documents
- `qa_Gold.csv`: question to gold-node mapping
- `nodes.csv`: tabular retrieval corpus
- `nodes.jsonl`: JSONL version of the corpus

Expected columns in `nodes.csv`:

- `node_id`
- `chapter_id`
- `source_file`
- `level`
- `heading`
- `heading_path`
- `content`

#### `training/`
Training and evaluation scripts for the main ablation ladder and chunking baselines.

- `train_retriever.py`: multi-seed controlled ablation pipeline
- `chunkingAb.py`: flat-chunk comparison experiments

#### `results/`
Utilities for aggregating run outputs into paper-style summary tables.

- `table.py`: prints formatted comparison tables from summary JSON files
- `stat.json`: main ablation summary input
- `chunkingAb.json`: chunking-study summary input

---

## Environment Setup

Use **Python 3.10+** (3.11 recommended).

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies

```powershell
pip install -r DatasetPrep\requirements.txt
pip install -r training\requirements.txt
```

---

## Data Preparation

### Step 1. Parse textbook markdown into hierarchical JSON

Run once for each source chapter:

```powershell
python Processing\parse_md.py Processing\BrTr.md --json-out Processing\BrTr.json
python Processing\parse_md.py Processing\AlFng.md --json-out Processing\AlFng.json
python Processing\parse_md.py Processing\GymAng.md --json-out Processing\GymAng.json
```

If the parsed JSON files already exist, this step can be skipped.

### Step 2. Build hierarchy-derived retrieval nodes

Place the chapter JSON files in `DatasetPrep/` and run:

```powershell
python DatasetPrep\prepare_nodes.py
```

This generates:

- `DatasetPrep/nodes.csv`
- `DatasetPrep/nodes.jsonl`

The preparation pipeline preserves:

- node IDs
- hierarchy level
- heading path
- chapter ID
- node text/content

The resulting node representation keeps retrieval units aligned with textbook structure instead of applying fixed-length chunking by default.

---

## Running the Main Ablation Experiments

The main training script uses project-relative defaults inside its config:

- `nodes_csv = "DatasetPrep/nodes.csv"`
- `qa_gold_csv = "DatasetPrep/qa_Gold.csv"`
- `output_root = "results/trained_retriever"`
- `stat_json_path = ""` → auto-defaults to `results/stat.json`

Run from the repository root:

```powershell
python training\train_retriever.py
```

### What this script runs

- controlled ablation suite **A0–A5**
- node-disjoint train/validation/test splitting
- multi-seed evaluation with seeds **42, 43, 44**
- dense bi-encoder training and hybrid retrieval evaluation
- per-run result saving and global result summarization
- automatic export of `results/stat.json`

---

## Running the Chunking Study

The chunking baseline script uses local defaults such as:

- `NODES_CSV = "DatasetPrep/nodes.csv"`
- `QA_GOLD_CSV = "DatasetPrep/qa_Gold.csv"`
- `OUTPUT_DIR = "results"`
- `CHUNKING_JSON_PATH = ""` → auto-defaults to `results/chunkingAb.json`

Run:

```powershell
python training\chunkingAb.py
```

### Outputs

- `chunk_baseline_results_<timestamp>.json`
- `chunk_baseline_summary_<timestamp>.txt`
- `results/chunkingAb.json`

This script compares hierarchy-preserving nodes against flat chunk alternatives for:

- **BM25**
- **DenseZS** (zero-shot multilingual-E5)
- **DenseFT** (fine-tuned dense retrieval)

---

## Result Aggregation

To generate paper-style summary tables, run:

```powershell
python results\table.py
```

This script reads:

- `results/stat.json`
- `results/chunkingAb.json`

and prints formatted comparison tables for the main ablations and chunking experiments.

---

## Recommended End-to-End Order

1. Parse textbook markdown into chapter JSON
2. Build retrieval nodes with `DatasetPrep/prepare_nodes.py`
3. Verify that `qa_Gold.csv` matches valid `positive_node_id` values in `nodes.csv`
4. Run `training/train_retriever.py`
5. Run `training/chunkingAb.py`
6. Run `results/table.py`

---

## Reproducibility Notes

For reproducible runs, keep track of:

- exact script version / commit hash
- seed set
- final config values
- generated summary JSON files
- printed table outputs

Additional implementation notes:

- GPU is recommended for dense training
- CPU-only runs are possible but significantly slower
- `train_retriever.py` and `chunkingAb.py` execute immediately because they call `run()` at file end
- `train_retriever.py` validates that all `positive_node_id` entries in `qa_Gold.csv` exist in `nodes.csv`
- `prepare_nodes.py` uses deterministic chapter ordering

---

## Summary of Findings

The paper’s main takeaways, which this repository is designed to reproduce, are:

- **BM25 remains strong at early ranks**
- **Dense retrieval improves deeper recall**
- **Hybrid fusion performs best overall**
- **Hierarchy-aware units favor lexical retrieval**
- **Dense retrieval is more sensitive to chunk granularity and supervision**

The best reported configuration in the paper is **A5 with MinMax fusion**, which reaches **0.746 MRR** and **0.923 Recall@20** under the reported benchmark setting.

---

## Citation

If you use this repository, please cite the accompanying paper.

```bibtex
@misc{anonymous2026banglabiologyretrieval,
  title        = {A Controlled Study of Bangla Biology Textbook Retrieval: Diagnosing Unit Design, Dense Supervision, and Hybrid Fusion},
  author       = {Anonymous ACL Submission},
  year         = {2026},
  note         = {Update citation after review}
}
```

---

## A Note on Review Anonymity

If this repository is being used during double-blind review, avoid adding author names, institutional details, or identifying links until the review period is over.
