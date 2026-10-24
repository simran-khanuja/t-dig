"""One exact average-linkage tree, multiple cuts, and blinded human audit pairs."""

from collections import Counter
import html
from pathlib import Path
import random

from .cluster import select_members
from .dataset import SCHEMA
from .storage import digest


def cut_tree(children, heights, n, threshold):
    """Return cluster labels at `threshold`, merging only below it, as sklearn does (ties too)."""
    import numpy as np

    if not 0 < threshold <= 2:
        raise ValueError('Invalid cosine distance threshold')
    if children.shape != (max(n - 1, 0), 2) or heights.shape != (max(n - 1, 0),):
        raise ValueError('Invalid tree shape')
    if not np.isfinite(heights).all() or (np.diff(heights) < -1e-12).any():
        raise ValueError('Invalid/nonmonotonic average linkage')
    parent = np.arange(n)
    representative = np.arange(max(2 * n - 1, 0))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, pair in enumerate(children):
        if any(v < 0 or v >= n + i for v in pair):
            raise ValueError('Invalid tree child')
        if heights[i] >= threshold:
            break
        a, b = (find(representative[int(v)]) for v in pair)
        parent[b] = a
        representative[n + i] = a
    labels = [find(i) for i in range(n)]
    return np.unique(labels, return_inverse=True)[1]


def condensed_distances(vectors, block_rows=256, progress=None):
    """Return condensed cosine distances in scipy order, computed in row blocks."""
    import numpy as np

    n = len(vectors)
    distances = np.empty(n * (n - 1) // 2, dtype=np.float64)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        block = vectors[start:stop] @ vectors.T
        block *= -1
        block += 1
        np.clip(block, 0, 2, out=block)
        for local, i in enumerate(range(start, stop)):
            offset = n * i - i * (i + 1) // 2
            distances[offset : offset + n - i - 1] = block[local, i + 1 :]
        if progress is not None:
            progress('distance_blocks', rows_done=stop, rows_total=n)
    return distances


def full_tree(vectors, progress=None):
    import numpy as np
    from scipy.cluster.hierarchy import linkage

    if len(vectors) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    distances = condensed_distances(vectors, progress=progress)
    if progress is not None:
        progress('average_linkage', condensed_gib=distances.nbytes / 2**30)
    # sklearn's unstructured average linkage calls this scipy routine on the same
    # upper triangle.
    tree = linkage(distances, method='average')
    return tree[:, :2].astype(np.int64), tree[:, 2]


def pool_at(data, records, vectors, labels, threshold, provenance, seed=42):
    import numpy as np

    _, groups = select_members(
        vectors, labels, 1, 1, selection='random', seed=seed, ids=[r['id'] for r in records]
    )
    kept, clusters, removed_pairs = [], [], []
    for indices, picks, anchor in groups:
        ids = [records[i]['id'] for i in indices]
        cid = digest(sorted(ids))
        selected = picks[0]
        representative = records[anchor]['prompt']
        clusters.append(
            dict(
                cluster_id=cid,
                size=len(indices),
                representative=representative,
                member_ids=ids,
                selected_ids=[records[selected]['id']],
            )
        )
        kept.append({**records[selected], 'cluster_id': cid, 'cluster_rep': representative})
        for i in indices:
            if i != selected:
                removed_pairs.append((i, selected, cid, float(1 - vectors[i] @ vectors[selected])))
    sizes = np.array([len(g[0]) for g in groups])
    settings = dict(
        distance=threshold,
        max_per_cluster=1,
        min_cluster_size=1,
        model=provenance['model'],
        revision=provenance['revision'],
        method='agglomerative',
        metric='cosine',
        linkage='average',
        selection='random',
        seed=seed,
        shared_hierarchy=True,
    )
    metadata = dict(
        input_hash=digest(data),
        upstream_run_hash=data['run_hash'],
        upstream_complete=data['metadata']['full_pipeline_complete'],
        settings=settings,
        n_input=len(records),
        n_kept=len(kept),
        n_removed=len(records) - len(kept),
        n_clusters=len(clusters),
        sweep_provenance=provenance,
        threshold_validated=False,
        variation_axis_count_distribution=dict(
            Counter(str(len(r['variation_axes'])) for r in kept)
        ),
    )
    result = dict(
        schema_version=SCHEMA,
        kind='clustered_pool',
        metadata=metadata,
        kept=sorted(kept, key=lambda r: r['id']),
        cluster_summary=clusters,
    )
    stats = dict(
        distance=threshold,
        clusters=len(clusters),
        selected_prompts=len(kept),
        removed_prompts=len(removed_pairs),
        singletons=int((sizes == 1).sum()),
        largest_cluster=int(sizes.max()) if len(sizes) else 0,
        cluster_size_quantiles={
            str(q): float(np.quantile(sizes, q)) if len(sizes) else 0 for q in [0.5, 0.9, 0.99]
        },
    )
    rng = random.Random(f'{seed}:{threshold}:audit')
    random_pairs = rng.sample(removed_pairs, min(50, len(removed_pairs)))
    random_ids = {(p[0], p[1]) for p in random_pairs}
    difficult = [
        p
        for p in sorted(removed_pairs, key=lambda p: (-p[3], p[0]))
        if (p[0], p[1]) not in random_ids
    ][:20]
    audit = []
    for kind, pairs in [
        ('random_removed_prompt', random_pairs),
        ('largest_representative_distance', difficult),
    ]:
        for i, j, cid, distance in pairs:
            audit.append(
                dict(
                    audit_id=digest([threshold, records[i]['id'], records[j]['id']]),
                    threshold=threshold,
                    sampling=kind,
                    cluster_id=cid,
                    removed_id=records[i]['id'],
                    retained_id=records[j]['id'],
                    removed_prompt=records[i]['prompt'],
                    retained_prompt=records[j]['prompt'],
                    cosine_distance=distance,
                    label=None,
                )
            )
    return result, stats, audit


def write_review(out, stats, pairs):
    """Write `review.html`: offline labeling with distances hidden, labels exported locally."""
    import json

    rows = ''.join(
        '<tr>'
        + ''.join(
            f'<td>{html.escape(str(s[k]))}</td>'
            for k in ['distance', 'clusters', 'removed_prompts', 'singletons', 'largest_cluster']
        )
        + '</tr>'
        for s in stats
    )
    cards = []
    for n, p in enumerate(pairs):
        cards.append(
            f'<section><b>Pair {n+1}</b><p>Discarded: {html.escape(p["removed_prompt"])}</p>'
            f'<p>Retained: {html.escape(p["retained_prompt"])}</p><select data-index="{n}">'
            '<option value="">Unreviewed</option><option value="redundant">Acceptably redundant</option>'
            '<option value="distinct">Meaningfully distinct</option><option value="uncertain">Uncertain</option></select></section>'
        )
    payload = json.dumps(pairs, ensure_ascii=True).replace('<', '\\u003c')
    document = f'''<!doctype html><meta charset="utf-8"><title>HPDv2 threshold review</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:2em auto}}td,th{{padding:8px;border:1px solid #ddd}}
section{{padding:1em;border-bottom:1px solid #ccc}}select,button{{padding:8px}}</style>
<h1>HPDv2 clustering threshold sweep — provisional</h1>
<p>One seeded random prompt per cluster. No threshold is validated yet. Unresolved upstream prompts remain excluded from this sweep, not reclassified as rejects.</p>
<table><tr><th>Distance</th><th>Clusters / selected prompts</th><th>Removed</th><th>Singletons</th><th>Largest cluster</th></tr>{rows}</table>
<h2>Human audit</h2><p>Would keeping both prompts add meaningful subject or scene coverage? Judge redundancy for training, not just shared topic. Distances and sampling strata are hidden here to reduce anchoring. The JSON export includes them. Random and deliberately difficult examples must be analyzed separately.</p>
<p>Download labels before closing this page; they are stored in memory only.</p>
<button onclick="saveLabels()">Download labels</button>{''.join(cards)}
<script>const pairs={payload};document.querySelectorAll('select').forEach(e=>e.onchange=()=>pairs[+e.dataset.index].label=e.value||null);
function saveLabels(){{const u=URL.createObjectURL(new Blob([JSON.stringify(pairs,null,2)],{{type:'application/json'}}));const a=document.createElement('a');a.href=u;a.download='threshold-audit-labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}}</script>'''
    (Path(out) / 'review.html').write_text(document, encoding='utf-8')
