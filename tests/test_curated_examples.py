import copy
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
from elicitation.curated_examples import build_bank, validate_bank
from elicitation.examples import pick_examples
from elicitation.runner import example_content, validate_config


class CuratedExamplesTests(unittest.TestCase):
    def test_verbatim_random_selection_preserves_specific_requests(self):
        inventory = dict(
            examples=[
                dict(
                    id='one',
                    prompt='A tree',
                    requests=[
                        'background',
                        'Different backgrounds',
                        'Make 1 look like Treebeard from LOTR!',
                        'A tree looking menacing or scary.',
                    ],
                )
            ]
        )
        plan = dict(
            version=2,
            seed=42,
            prompts=[dict(prompt='A tree', similarity_groups=[[0, 1], [2], [3]])],
        )
        bank = build_bank(inventory, plan)
        self.assertEqual(bank, build_bank(inventory, plan))
        row = bank['examples'][0]
        self.assertIn(row['requests'][0], inventory['examples'][0]['requests'][:2])
        self.assertEqual(row['requests'][1:], inventory['examples'][0]['requests'][2:])
        payload, _ = example_content(row, 'high')
        self.assertEqual(json.loads(payload[0]['text'])['axes'], row['requests'])
        bad = copy.deepcopy(plan)
        bad['prompts'][0]['similarity_groups'].append([0])
        with self.assertRaises(ValueError):
            build_bank(inventory, bad)

    def test_provenance_deduplication_payload_and_source_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'human.jsonl'
            p.write_text(
                json.dumps(dict(prompt='A clock', other_axes=['Clock style', 'style of clock']))
                + '\n'
            )
            inventory = dict(
                examples=[
                    dict(
                        id='source-one',
                        prompt='A clock',
                        requests=['Clock style', 'style of clock'],
                        source=str(p),
                        line=1,
                        source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    )
                ]
            )
            bank = build_bank(inventory, [[1, [['Clock design', [0, 1]]]]])
            validate_bank(bank)
            row = bank['examples'][0]
            self.assertEqual(row['requests'], ['Clock design'])
            self.assertEqual(len(row['axis_sources'][0]['sources']), 2)
            payload, _ = example_content(row, 'high')
            value = json.loads(payload[0]['text'])
            self.assertEqual(value['axes'], ['Clock design'])
            self.assertNotIn('human_requests', value)
            self.assertNotIn('source', value)
            broken = copy.deepcopy(bank)
            broken['examples'][0]['requests'] = ['Invented axis']
            with self.assertRaises(ValueError):
                validate_bank(broken)
            p.write_text('{}\n')
            with self.assertRaises(ValueError):
                validate_bank(bank)

    def test_all_examples_omit_target_and_config_accepts_explicit_all(self):
        bank = dict(
            kind='curated_merged_axis_examples',
            examples=[dict(id=str(i), prompt=f'Prompt {i}') for i in range(3)],
        )
        target = dict(id='target', prompt=' PROMPT 1 ')
        self.assertEqual([r['id'] for r in pick_examples(bank, target, 'all', 42)], ['0', '2'])
        self.assertEqual(len(pick_examples(bank, target, 1, 42)), 1)
        with self.assertRaises(ValueError):
            pick_examples(bank, target, 3, 42)
        cfg = dict(
            api_base='https://example.test/v1',
            model='test',
            timeout=30,
            max_tokens=1000,
            max_axes=4,
            example_count='all',
            seed=42,
            detail='high',
            reasoning_effort='medium',
        )
        validate_config(cfg)
        cfg['example_count'] = True
        with self.assertRaises(ValueError):
            validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
