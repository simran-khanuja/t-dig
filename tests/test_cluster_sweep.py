import tempfile
from pathlib import Path
import unittest
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from hpdv2.cluster_sweep import cut_tree, full_tree, pool_at, write_review, condensed_distances
from hpdv2.cluster import cluster
from hpdv2.storage import atomic_json, prompt_id, read_json
from test_pipeline import decision


class SweepTests(unittest.TestCase):
    def test_blocked_cosine_matches_dense_upper_triangle(self):
        v = np.random.default_rng(18).normal(size=(73, 9))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        expected = np.clip(1 - v @ v.T, 0, 2)[np.triu_indices(len(v), 1)]
        for size in [1, 7, 32, 256]:
            np.testing.assert_allclose(condensed_distances(v, size), expected, rtol=0, atol=1e-14)

    def test_all_cuts_match_independent_sklearn_with_ties(self):
        v = np.random.default_rng(7).normal(size=(45, 6))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v[1] = v[0]
        children, heights = full_tree(v)
        distances = np.clip(1 - v @ v.T, 0, 2)
        np.fill_diagonal(distances, 0)
        for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.8, 1.0, 2.0, float(heights[10])]:
            labels = cut_tree(children, heights, len(v), t)
            other = AgglomerativeClustering(
                n_clusters=None, distance_threshold=t, metric='precomputed', linkage='average'
            ).fit_predict(distances)
            np.testing.assert_array_equal(
                labels[:, None] == labels[None, :], other[:, None] == other[None, :]
            )

    def test_pool_matches_original_cluster_and_preserves_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            texts = ['a chair', 'a seat', 'a car']
            rows = []
            for text in texts:
                r = {'id': prompt_id(text), 'prompt': text}
                rows.append({**r, **decision(r, 'pass2')})
            rows.sort(key=lambda r: r['id'])
            data = dict(
                schema_version=2,
                kind='filter_summary',
                stage='pass2',
                run_hash='test',
                metadata={'full_pipeline_complete': False},
                kept=rows,
            )
            atomic_json(out / 'input.json', data)
            mapping = dict(zip(texts, [[1.0, 0], [1.0, 0.01], [0.0, 1.0]]))
            v = np.array([mapping[r['prompt']] for r in rows])
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            children, heights = full_tree(v)
            for t in [0.15, 0.35, 0.45]:
                pool, stats, pairs = pool_at(
                    data,
                    rows,
                    v,
                    cut_tree(children, heights, 3, t),
                    t,
                    {'model': 'test', 'revision': 'test'},
                )
                target = out / f'{t}.json'
                cluster(
                    out / 'input.json',
                    target,
                    distance=t,
                    execute=True,
                    allow_partial=True,
                    embedder=lambda ts: [mapping[x] for x in ts],
                )
                original = read_json(target)
                self.assertEqual(pool['kept'], original['kept'])
                self.assertEqual(
                    sorted(pool['cluster_summary'], key=lambda r: r['cluster_id']),
                    sorted(original['cluster_summary'], key=lambda r: r['cluster_id']),
                )
                self.assertFalse(pool['metadata']['upstream_complete'])
                self.assertFalse(pool['metadata']['threshold_validated'])
                self.assertEqual(stats['clusters'], 2)
                self.assertTrue(
                    all(p['retained_id'] in {r['id'] for r in pool['kept']} for p in pairs)
                )
                write_review(out, [stats], pairs)
                self.assertIn('Download labels', (out / 'review.html').read_text())

    def test_empty_singleton_and_invalid_tree(self):
        for n in [0, 1]:
            children, heights = full_tree(np.ones((n, 2)))
            self.assertEqual(len(cut_tree(children, heights, n, 0.3)), n)
        with self.assertRaises(ValueError):
            cut_tree(np.array([[0, 5]]), np.array([0.1]), 2, 0.3)
        with self.assertRaises(ValueError):
            cut_tree(np.empty((0, 2)), np.array([0.1]), 2, 0.3)


if __name__ == '__main__':
    unittest.main()
