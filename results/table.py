import json
import math
from pathlib import Path
from statistics import mean, stdev
import pandas as pd

STAT_JSON = 'stat.json'
CHUNKING_JSON = 'chunkingAb.json'

ABLATION_ORDER = [
    'A0_BM25',
    'A1_DENSE',
    'A2_DENSE+HYBRID_RRF_eval',
    'A3_A2+MINING_refresh',
    'A4_A3+HYBRID_ckpt',
    'A5_A3+FUSION_minmax',
]
ABLATION_LABELS = {
    'A0_BM25': 'A0 BM25',
    'A1_DENSE': 'A1 Dense',
    'A2_DENSE+HYBRID_RRF_eval': 'A2 Hybrid (RRF)',
    'A3_A2+MINING_refresh': 'A3 +Refresh',
    'A4_A3+HYBRID_ckpt': 'A4 +Ckpt',
    'A5_A3+FUSION_minmax': 'A5 MinMax',
}
CHUNK_ORDER = {
    'C0_word128': 0,
    'C1_word256': 1,
    'C2_word512': 2,
    'C3_char500': 3,
    'C4_char1000': 4,
    'C5_sentence': 5,
    'Hierarchy_nodes': 6,
}
FULL_METRICS = ['MRR', 'R@1', 'R@3', 'R@5', 'R@10', 'R@20', 'NDCG@10', 'MeanRank', 'MedianRank']


def load_concat_json(path):
    text = Path(path).read_text(encoding='utf-8')
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        pos = 0
        items = []
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            obj, end = decoder.raw_decode(text, pos)
            items.append(obj)
            pos = end
        return items


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_stdev(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) <= 1:
        return 0.0
    return stdev(vals)


def fmt_pm(m, s, digits=4):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return '--'
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return f'{m:.{digits}f}'
    return f'{m:.{digits}f} ± {s:.{digits}f}'


def fmt_ci(lo, hi, digits=4):
    if lo is None or hi is None:
        return '--'
    return f'[{lo:.{digits}f}, {hi:.{digits}f}]'


def render_table(df, title, numeric_cols=None, lower_better_cols=None):
    numeric_cols = numeric_cols or []
    lower_better_cols = set(lower_better_cols or [])
    higher_better_cols = [c for c in numeric_cols if c not in lower_better_cols]

    marks = {c: set() for c in numeric_cols}
    for c in higher_better_cols:
        vals = pd.to_numeric(df[c], errors='coerce')
        if vals.notna().any():
            best = vals.max()
            marks[c] = set(df.index[vals == best].tolist())
    for c in lower_better_cols:
        vals = pd.to_numeric(df[c], errors='coerce')
        if vals.notna().any():
            best = vals.min()
            marks[c] = set(df.index[vals == best].tolist())

    out = df.copy().astype(object)
    for c in numeric_cols:
        for i in out.index:
            val = out.loc[i, c]
            if pd.isna(val):
                txt = '--'
            else:
                if c in {'#Chunks'}:
                    txt = f'{int(round(val))}'
                elif c in {'AvgWords', 'MeanRank', 'MedianRank', 'PosDeltaFrac', 'Alpha'}:
                    txt = f'{val:.2f}'
                else:
                    txt = f'{val:.4f}'
                if i in marks[c]:
                    txt = f'**{txt}**'
            out.loc[i, c] = txt

    print('\n' + '=' * len(title))
    print(title)
    print('=' * len(title))
    try:
        print(out.to_markdown(index=False))
    except Exception:
        print(out.to_string(index=False))


def family_for_ablation(rec):
    name = rec['ablation'].get('name', '')
    if name == 'A0_BM25':
        return 'bm25'
    # Detect ablations that correspond to a trained dense model.
    # This handles explicit flags or names containing 'trained'.
    if 'TRAINED' in name.upper() or rec['ablation'].get('trained_dense', False) or rec.get('trained_dense', False):
        return 'trained_dense'
    return 'hybrid' if rec['ablation'].get('hybrid_eval', False) else 'dense'


def metric_key(family, metric):
    if family == 'bm25':
        return f'bm25_{metric}'
    if family == 'hybrid':
        return f'hybrid_{metric}'
    if family == 'trained_dense':
        return f'trained_dense_{metric}'
    return metric


def build_main_tables(stat_rows):
    main_plain_rows = []
    main_pm_rows = []
    ci_rows = []
    delta_rows = []

    spec = [
        ('MRR', 'mrr'), ('R@1', 'recall@1'), ('R@3', 'recall@3'), ('R@5', 'recall@5'),
        ('R@10', 'recall@10'), ('R@20', 'recall@20'), ('NDCG@10', 'ndcg@10_single'),
        ('MeanRank', 'mean_rank'), ('MedianRank', 'median_rank')
    ]

    for ab in ABLATION_ORDER:
        rows = sorted([r for r in stat_rows if r['ablation']['name'] == ab], key=lambda r: r['seed'])
        if not rows:
            continue

        fam = family_for_ablation(rows[0])

        plain = {'ID': ab.split('_')[0], 'Method': ABLATION_LABELS.get(ab, ab)}
        pmrow = {'ID': ab.split('_')[0], 'Method': ABLATION_LABELS.get(ab, ab)}
        cirow = {'ID': ab.split('_')[0], 'Method': ABLATION_LABELS.get(ab, ab)}

        for public, internal in spec:
            vals = []
            for r in rows:
                key = metric_key(fam, internal)
                vals.append(r['metrics'].get(key))
            vals = [v for v in vals if v is not None]

            if vals:
                mu, sd = mean(vals), safe_stdev(vals)
                plain[public] = mu
                pmrow[public] = fmt_pm(mu, sd, digits=2 if 'Rank' in public else 4)
            else:
                plain[public] = None
                pmrow[public] = '--'

            if 'Rank' in public:
                cirow[public] = '--'
            else:
                cis = []
                for r in rows:
                    fam_ci = r.get('bootstrap_ci', {}).get(fam, {})
                    if internal in fam_ci:
                        cis.append(fam_ci[internal])

                if cis:
                    cirow[public] = fmt_ci(
                        mean(c['ci95_low'] for c in cis),
                        mean(c['ci95_high'] for c in cis)
                    )
                else:
                    cirow[public] = '--'

        if fam == 'hybrid':
            alphas = [r.get('tuned_alpha') for r in rows if r.get('tuned_alpha') is not None]
            cirow['Alpha'] = fmt_pm(mean(alphas), safe_stdev(alphas), 2) if alphas else '--'

            deltas = []
            for r in rows:
                d = r.get('bootstrap_ci', {}).get('delta_hybrid_minus_bm25_mrr')
                if d:
                    deltas.append(d)

            if deltas:
                pos = sum(1 for d in deltas if d['ci95_low'] > 0)
                delta_rows.append({
                    'ID': ab.split('_')[0],
                    'Method': ABLATION_LABELS.get(ab, ab),
                    'DeltaMRR': mean(d['delta_mean'] for d in deltas),
                    'CI_low': mean(d['ci95_low'] for d in deltas),
                    'CI_high': mean(d['ci95_high'] for d in deltas),
                    'PosDeltaFrac': pos / len(deltas),
                    'Alpha': mean(alphas) if alphas else None,
                    'SigSeeds': f'{pos}/{len(deltas)}',
                })
        else:
            cirow['Alpha'] = '--'

        main_plain_rows.append(plain)
        main_pm_rows.append(pmrow)
        ci_rows.append(cirow)

    return (
        pd.DataFrame(main_plain_rows),
        pd.DataFrame(main_pm_rows),
        pd.DataFrame(ci_rows),
        pd.DataFrame(delta_rows)
    )


def build_best_summary(stat_rows):
    picks = [
        ('BM25 (A0)', 'A0_BM25', 'bm25'),
        ('Best dense (A3)', 'A3_A2+MINING_refresh', 'dense'),
        ('Best hybrid (A5)', 'A5_A3+FUSION_minmax', 'hybrid'),
    ]
    rows = []
    for label, ab, fam in picks:
        sub = [r for r in stat_rows if r['ablation']['name'] == ab]
        if not sub:
            continue

        row = {'System': label}
        for public, internal in [
            ('MRR', 'mrr'),
            ('R@1', 'recall@1'),
            ('R@3', 'recall@3'),
            ('R@5', 'recall@5'),
            ('R@10', 'recall@10'),
            ('R@20', 'recall@20'),
            ('NDCG@10', 'ndcg@10_single')
        ]:
            vals = []
            for r in sub:
                vals.append(r['metrics'].get(metric_key(fam, internal)))
            vals = [v for v in vals if v is not None]
            row[public] = fmt_pm(mean(vals), safe_stdev(vals)) if vals else '--'
        rows.append(row)

    return pd.DataFrame(rows)


def build_chunking_tables(chunk_rows, stat_rows):
    rows = []
    for x in chunk_rows:
        retr = x['retriever']
        # Map retriever string to metric family and human label.
        if retr == 'bm25':
            fam = 'bm25'
            label = 'BM25'
            ci = x.get('bm25_ci', {})
        elif retr in ('dense_trained', 'trained_dense'):
            fam = 'trained_dense'
            label = 'Dense trained'
            ci = x.get('trained_dense_ci', {})
        else:
            # default/zero-shot dense
            fam = 'dense'
            label = 'Dense zero-shot'
            ci = x.get('dense_ci', {})

        rows.append({
            'Retriever': label,
            'Chunking': x['chunk_config'],
            '#Chunks': x['n_chunks'],
            'AvgWords': x['avg_words'],
            'MRR': x.get(f'{fam}_mrr'),
            'R@1': x.get(f'{fam}_recall@1'),
            'R@3': x.get(f'{fam}_recall@3'),
            'R@5': x.get(f'{fam}_recall@5'),
            'R@10': x.get(f'{fam}_recall@10'),
            'R@20': x.get(f'{fam}_recall@20'),
            'NDCG@10': x.get(f'{fam}_ndcg@10'),
            'MeanRank': x.get(f'{fam}_mean_rank'),
            'MedianRank': x.get(f'{fam}_median_rank'),
            'MRR_CI_lo': ci.get('mrr', {}).get('lo'),
            'MRR_CI_hi': ci.get('mrr', {}).get('hi'),
            'R1_CI_lo': ci.get('recall@1', {}).get('lo'),
            'R1_CI_hi': ci.get('recall@1', {}).get('hi'),
            'R5_CI_lo': ci.get('recall@5', {}).get('lo'),
            'R5_CI_hi': ci.get('recall@5', {}).get('hi'),
        })

    hierarchy_specs = [
        ('BM25', 'bm25', 'A0_BM25', 166),
        ('Dense trained', 'trained_dense', 'A3_A2+MINING_refresh', 166),
    ]
    for label, fam, ab, chunks in hierarchy_specs:
        for r in [x for x in stat_rows if x['ablation']['name'] == ab]:
            ci = r.get('bootstrap_ci', {}).get(fam, {})
            rows.append({
                'Retriever': label,
                'Chunking': 'Hierarchy_nodes',
                '#Chunks': chunks,
                'AvgWords': None,
                'MRR': r['metrics'].get(metric_key(fam, 'mrr')),
                'R@1': r['metrics'].get(metric_key(fam, 'recall@1')),
                'R@3': r['metrics'].get(metric_key(fam, 'recall@3')),
                'R@5': r['metrics'].get(metric_key(fam, 'recall@5')),
                'R@10': r['metrics'].get(metric_key(fam, 'recall@10')),
                'R@20': r['metrics'].get(metric_key(fam, 'recall@20')),
                'NDCG@10': r['metrics'].get(metric_key(fam, 'ndcg@10_single')),
                'MeanRank': r['metrics'].get(metric_key(fam, 'mean_rank')),
                'MedianRank': r['metrics'].get(metric_key(fam, 'median_rank')),
                'MRR_CI_lo': ci.get('mrr', {}).get('ci95_low'),
                'MRR_CI_hi': ci.get('mrr', {}).get('ci95_high'),
                'R1_CI_lo': ci.get('recall@1', {}).get('ci95_low'),
                'R1_CI_hi': ci.get('recall@1', {}).get('ci95_high'),
                'R5_CI_lo': ci.get('recall@5', {}).get('ci95_low'),
                'R5_CI_hi': ci.get('recall@5', {}).get('ci95_high'),
            })

    df = pd.DataFrame(rows)

    grouped = df.groupby(['Retriever', 'Chunking'], as_index=False).agg({
        '#Chunks': 'mean',
        'AvgWords': 'mean',
        'MRR': ['mean', 'std'],
        'R@1': ['mean', 'std'],
        'R@3': ['mean', 'std'],
        'R@5': ['mean', 'std'],
        'R@10': ['mean', 'std'],
        'R@20': ['mean', 'std'],
        'NDCG@10': ['mean', 'std'],
        'MeanRank': ['mean', 'std'],
        'MedianRank': ['mean', 'std'],
        'MRR_CI_lo': 'mean',
        'MRR_CI_hi': 'mean',
        'R1_CI_lo': 'mean',
        'R1_CI_hi': 'mean',
        'R5_CI_lo': 'mean',
        'R5_CI_hi': 'mean'
    })

    grouped.columns = [
        'Retriever', 'Chunking', '#Chunks', 'AvgWords',
        'MRR', 'MRR_std', 'R@1', 'R@1_std', 'R@3', 'R@3_std', 'R@5', 'R@5_std',
        'R@10', 'R@10_std', 'R@20', 'R@20_std', 'NDCG@10', 'NDCG@10_std',
        'MeanRank', 'MeanRank_std', 'MedianRank', 'MedianRank_std',
        'MRR_CI_lo', 'MRR_CI_hi', 'R1_CI_lo', 'R1_CI_hi', 'R5_CI_lo', 'R5_CI_hi'
    ]

    retr_order = {'BM25': 0, 'Dense zero-shot': 1, 'Dense trained': 2}
    grouped['retr_order'] = grouped['Retriever'].map(retr_order).fillna(9)
    grouped['chunk_order'] = grouped['Chunking'].map(CHUNK_ORDER).fillna(99)
    grouped = grouped.sort_values(['retr_order', 'chunk_order']).drop(columns=['retr_order', 'chunk_order']).reset_index(drop=True)

    compact = grouped[['Retriever', 'Chunking', '#Chunks', 'MRR', 'R@1', 'R@3', 'R@5', 'R@10', 'R@20', 'NDCG@10', 'MeanRank', 'MedianRank']].copy()

    full = grouped[['Retriever', 'Chunking', '#Chunks', 'AvgWords']].copy()
    for m in FULL_METRICS:
        full[m] = grouped.apply(
            lambda r: fmt_pm(r[m], r[f'{m}_std'], digits=2 if 'Rank' in m else 4),
            axis=1
        )
    full['MRR_CI95'] = grouped.apply(lambda r: fmt_ci(r['MRR_CI_lo'], r['MRR_CI_hi']), axis=1)
    full['R@1_CI95'] = grouped.apply(lambda r: fmt_ci(r['R1_CI_lo'], r['R1_CI_hi']), axis=1)
    full['R@5_CI95'] = grouped.apply(lambda r: fmt_ci(r['R5_CI_lo'], r['R5_CI_hi']), axis=1)

    return compact, full


def main():
    stat_rows = load_concat_json(STAT_JSON)
    chunk_rows = load_json(CHUNKING_JSON)

    main_plain, main_pm, main_ci, delta_df = build_main_tables(stat_rows)
    best_df = build_best_summary(stat_rows)
    chunk_compact, chunk_full = build_chunking_tables(chunk_rows, stat_rows)

    render_table(
        main_plain,
        'MAIN ABLATION TABLE (plain scores)',
        FULL_METRICS,
        ['MeanRank', 'MedianRank']
    )

    print('\n==================================')
    print('MAIN ABLATION TABLE (mean ± std)')
    print('==================================')
    print(main_pm.to_markdown(index=False))

    print('\n=====================================================')
    print('BOOTSTRAP 95% CI TABLE (avg seed-wise CI; alpha shown)')
    print('=====================================================')
    print(main_ci.to_markdown(index=False))

    if not delta_df.empty:
        render_table(
            delta_df[['ID', 'Method', 'DeltaMRR', 'CI_low', 'CI_high', 'PosDeltaFrac', 'Alpha']],
            'PAIRED HYBRID - BM25 MRR DELTA',
            ['DeltaMRR', 'CI_low', 'CI_high', 'PosDeltaFrac', 'Alpha'],
            []
        )
        print('\nPaired delta significance support:')
        print(delta_df[['ID', 'Method', 'SigSeeds']].to_markdown(index=False))

    print('\n=================================')
    print('BEST FAMILY SUMMARY (mean ± std)')
    print('=================================')
    print(best_df.to_markdown(index=False))

    render_table(
        chunk_compact,
        'CHUNKING TABLE (plain scores, includes hierarchy rows)',
        ['#Chunks', 'MRR', 'R@1', 'R@3', 'R@5', 'R@10', 'R@20', 'NDCG@10', 'MeanRank', 'MedianRank'],
        ['MeanRank', 'MedianRank']
    )

    print('\n===========================================================')
    print('CHUNKING TABLE (mean ± std; CI columns only where available)')
    print('===========================================================')
    print(chunk_full.to_markdown(index=False))


if __name__ == '__main__':
    main()