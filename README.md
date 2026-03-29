# BEA Submission: Retrieval Pipeline

This repository contains a full retrieval workflow for Bangla academic content:

1. Parse Markdown chapters into hierarchical JSON.
2. Flatten hierarchical nodes into retrieval-ready node documents.
3. Train and evaluate dense and hybrid retrievers via ablations.
4. Compare chunking strategies and print publication-ready result tables.

## Repository Structure

- `Processing/`
  - `parse_md.py`: Converts Markdown headings into nested JSON trees.
  - `*.md`, `*.json`: Chapter source and parsed outputs.
- `DatasetPrep/`
  - `prepare_nodes.py`: Flattens chapter JSON trees into `nodes.csv` and `nodes.jsonl`.
  - `qa_Gold.csv`: Question to positive node mapping (`question`, `positive_node_id`).
  - `nodes.csv`, `nodes.jsonl`: Prepared retrieval corpus.
- `training/`
  - `train_retriever.py`: Main multi-seed ablation training/evaluation pipeline.
  - `chunkingAb.py`: Chunking baseline experiments (BM25 and dense zero-shot).
- `results/`
  - `table.py`: Aggregates JSON results and prints comparison tables.
  - `stat.json`, `chunkingAb.json`: Inputs consumed by `table.py`.

## Environment Setup

Use Python 3.10+ (3.11 recommended).

### 1) Create and activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

Both requirements files are needed:

```powershell
pip install -r DatasetPrep\requirements.txt
pip install -r training\requirements.txt
```

## Data Preparation Workflow

### Step A: Convert Markdown to hierarchical JSON

Run once per chapter markdown file:

```powershell
python Processing\parse_md.py Processing\BrTr.md --json-out Processing\BrTr.json
python Processing\parse_md.py Processing\AlFng.md --json-out Processing\AlFng.json
python Processing\parse_md.py Processing\GymAng.md --json-out Processing\GymAng.json
```

If you already have JSONs in `DatasetPrep/`, you can skip this.

### Step B: Build retrieval nodes

Ensure chapter JSON files are present in `DatasetPrep/` (for example `BrTr.json`, `AlFng.json`, `GymAng.json`), then run:

```powershell
python DatasetPrep\prepare_nodes.py
```

Outputs:

- `DatasetPrep/nodes.csv`
- `DatasetPrep/nodes.jsonl`

Expected `nodes.csv` columns:

- `node_id`
- `chapter_id`
- `source_file`
- `level`
- `heading`
- `heading_path`
- `content`

## Training and Ablation Experiments

### Important path configuration

`training/train_retriever.py` now uses local project-relative defaults inside `Config`:

- `nodes_csv = "DatasetPrep/nodes.csv"`
- `qa_gold_csv = "DatasetPrep/qa_Gold.csv"`
- `output_root = "results/trained_retriever"`
- `stat_json_path = ""` (auto-defaults to `results/stat.json` when empty)

No path edits are required if you run commands from the repository root.

Then run:

```powershell
python training\train_retriever.py
```

What this script does:

- Runs ablation suite `A0` to `A5`.
- Uses seeds `(42, 43, 44)`.
- Uses node-disjoint train/val/test splitting.
- Trains dense bi-encoder variants and evaluates BM25, dense, and hybrid retrieval.
- Writes per-run `results.json` plus a global summary JSON under `output_root`.
- Automatically writes `results/stat.json` for `results/table.py`.

## Chunking Baseline Experiments

`training/chunkingAb.py` uses local defaults near the top:

- `NODES_CSV`
- `QA_GOLD_CSV`
- `OUTPUT_DIR`
- `CHUNKING_JSON_PATH` (auto-defaults to `results/chunkingAb.json` when empty)

Current defaults:

- `NODES_CSV = "DatasetPrep/nodes.csv"`
- `QA_GOLD_CSV = "DatasetPrep/qa_Gold.csv"`
- `OUTPUT_DIR = "results"`

No path edits are required if you run from the repository root.

Run:

```powershell
python training\chunkingAb.py
```

Outputs (saved to `OUTPUT_DIR`):

- `chunk_baseline_results_<timestamp>.json`
- `chunk_baseline_summary_<timestamp>.txt`

Also auto-generated for table aggregation:

- `results/chunkingAb.json`

## Result Table Generation

`results/table.py` reads:

- `results/stat.json`
- `results/chunkingAb.json`

and prints formatted tables to stdout.

Run from the `results/` directory or keep input files there:

```powershell
python results\table.py
```

## Typical End-to-End Run Order

1. Parse markdown chapters (optional if JSON already exists).
2. Build nodes via `DatasetPrep/prepare_nodes.py`.
3. Verify `qa_Gold.csv` maps to valid `positive_node_id` values in `nodes.csv`.
4. Run `training/train_retriever.py`.
5. Run `training/chunkingAb.py`.
6. Run `results/table.py`.

## Notes and Troubleshooting

- GPU is recommended for dense training (`torch` with CUDA).
- If running CPU-only, training will be much slower.
- `train_retriever.py` and `chunkingAb.py` execute immediately because they call `run()` at file end.
- `train_retriever.py` checks whether all `positive_node_id` values in `qa_Gold.csv` exist in `nodes.csv` and raises an error otherwise.
- `prepare_nodes.py` enforces deterministic chapter order with `BrTr` first, then alphabetical.

## Citation/Reporting Tip

For reproducible reporting, keep:

- the exact script version,
- seed set,
- final config values,
- generated summary JSON files,
- and printed table outputs.
