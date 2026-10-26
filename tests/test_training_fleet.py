from collections import Counter
import copy
import tempfile
from pathlib import Path
import unittest

from axis_grpo.training.fleet import allocation_hosts, gpu_roles, single_node_roles
from axis_grpo.training.split import split_rows, val_subset
from axis_grpo.training.run import stable_identity
from axis_grpo.training.order import PhaseBatchSampler


class FleetTests(unittest.TestCase):
    def test_all_sixteen_gpus_have_exactly_one_role(self):
        totals = dict(trainer=0, vqa=0, renderers=0)
        for node in (0, 1):
            visible = [f'GPU-node{node}-{i}' for i in range(8)]
            roles = gpu_roles(node, visible)
            used = roles['trainer'] + [roles['vqa']] + roles['renderers']
            self.assertEqual(used, visible)
            totals['trainer'] += len(roles['trainer'])
            totals['vqa'] += 1
            totals['renderers'] += len(roles['renderers'])
        self.assertEqual(totals, dict(trainer=2, vqa=2, renderers=12))
        with self.assertRaises(ValueError):
            gpu_roles(1, ['0'] * 8)
        with self.assertRaises(ValueError):
            gpu_roles(0, ['0', '1'])

    def test_single_node_roles_cover_every_gpu_count(self):
        g = [f'GPU-{i}' for i in range(10)]
        self.assertEqual(single_node_roles(g[:1]), dict(trainer=g[:1], vqa=g[0], renderers=g[:1]))
        self.assertEqual(single_node_roles(g[:2]), dict(trainer=g[:1], vqa=g[1], renderers=g[1:2]))
        self.assertEqual(single_node_roles(g[:3]), dict(trainer=g[:1], vqa=g[1], renderers=g[2:3]))
        self.assertEqual(single_node_roles(g[:8]), dict(trainer=g[:2], vqa=g[2], renderers=g[3:8]))
        self.assertEqual(
            single_node_roles(g[:8], 1), dict(trainer=g[:1], vqa=g[1], renderers=g[2:8])
        )
        self.assertEqual(single_node_roles(g, 4), dict(trainer=g[:4], vqa=g[4], renderers=g[5:]))
        for visible, train in (([], None), (g[:2], 2), (g[:3], 2), (g[:1] * 2, None)):
            with self.assertRaises(ValueError):
                single_node_roles(visible, train)

    def test_node_order_is_not_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'nodes'
            p.write_text('znode\nanode\nznode\n')
            self.assertEqual(allocation_hosts(p, 2), ['znode', 'anode'])
            p.write_text('znode\n')
            with self.assertRaises(ValueError):
                allocation_hosts(p, 2)

    def test_relocation_requires_same_service_identity_and_order(self):
        a = dict(
            config=dict(
                allow_endpoint_rebind=True, reward_endpoints=['http://one:1', 'http://two:1']
            ),
            reward_services={'http://one:1': 'hash-a', 'http://two:1': 'hash-b'},
            source={'x': 'hash'},
        )
        b = copy.deepcopy(a)
        b['config']['reward_endpoints'] = ['http://new:1', 'http://newer:1']
        b['reward_services'] = {'http://new:1': 'hash-a', 'http://newer:1': 'hash-b'}
        self.assertEqual(stable_identity(a), stable_identity(b))
        b['reward_services']['http://new:1'] = 'changed'
        self.assertNotEqual(stable_identity(a), stable_identity(b))
        self.assertEqual(a['config']['reward_endpoints'][0], 'http://one:1')


class SplitTests(unittest.TestCase):
    def rows(self):
        rows = []
        for phase in (1, 2, 3, 4):
            for t2i_model in ('a', 'b', 'c', 'd'):
                for i in range(5):
                    pid = f'{t2i_model}-{i}'
                    rows.append(
                        dict(
                            prompt=[dict(role='user', content=f'{pid}: axis {phase}')],
                            extra_info=dict(
                                index=len(rows),
                                row_id=f'{pid}-{phase}',
                                prompt_id=pid,
                                base_prompt=pid,
                                cluster_id=pid,
                                source_renderer=t2i_model,
                                split='train',
                                axis_slot=phase - 1,
                                axis_kind='axis',
                                pass_number=phase,
                            ),
                        )
                    )
        return rows

    def test_split_preserves_complete_prompt_groups_and_passes(self):
        rows = self.rows()
        original = copy.deepcopy(rows)
        train, val, ids = split_rows(rows, 1, 42)
        self.assertEqual(rows, original)
        self.assertEqual((len(train), len(val), len(ids)), (64, 16, 4))
        self.assertTrue(all(r['extra_info']['prompt_id'] in ids for r in val))
        self.assertFalse(any(r['extra_info']['prompt_id'] in ids for r in train))
        sampler = PhaseBatchSampler([r['extra_info'] for r in train], 7)
        self.assertEqual([i for b in sampler for i in b], list(range(64)))
        self.assertEqual(split_rows(rows, 1, 42)[2], ids)
        for r in train + val:
            self.assertEqual(
                r['prompt'], original[r['extra_info']['source_export_index']]['prompt']
            )


    def test_val_subset_keeps_lowest_slots_and_overall_rows(self):
        _, val, _ = split_rows(self.rows(), 1, 42)
        subset = val_subset(val, 1)
        kinds = Counter(r['extra_info']['axis_kind'] for r in subset)
        self.assertEqual(kinds['overall'], sum(r['extra_info']['axis_kind'] == 'overall' for r in val))
        self.assertEqual(kinds['axis'], 4)
        self.assertTrue(all(r['extra_info']['axis_slot'] == 0 for r in subset if r['extra_info']['axis_kind'] == 'axis'))
        self.assertEqual([r['extra_info']['index'] for r in subset], list(range(len(subset))))
        self.assertEqual(len(val_subset(val, 99)), len(val))
    def test_cluster_overlap_is_rejected(self):
        rows = self.rows()
        for r in rows:
            r['extra_info']['cluster_id'] = 'same-cluster'
        with self.assertRaisesRegex(ValueError, 'cluster_id'):
            split_rows(rows, 1, 42)


if __name__ == '__main__':
    unittest.main()
