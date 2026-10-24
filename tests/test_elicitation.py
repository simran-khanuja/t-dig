import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpdv2.storage import atomic_json, digest, prompt_id
from elicitation import policy
from elicitation.api import Reply
from elicitation.examples import load_bank, pick_examples
from elicitation.mixture import mix
from elicitation.runner import run, build_request
from elicitation.selection import assign, verified_images


def answer():
    return dict(
        status='ok',
        reason='',
        alignment_notes='',
        axes=[
            dict(
                request='I would like to see different times on the clock',
                tpips_factor='clock time',
                observation='The hands appear similarly positioned',
                why_free='The prompt specifies a clock, but no time',
                mapping_status='faithful',
                constraints=[],
            )
        ],
    )


class ElicitationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [
            dict(id=prompt_id(f'A clock {i}'), prompt=f'A clock {i}', cluster_id=str(i))
            for i in range(20)
        ]
        self.selected, _ = assign(self.rows, 2)
        self.assignment = dict(items=self.selected, k=4)
        self.ap = self.root / 'assignment.json'
        atomic_json(self.ap, self.assignment)
        atomic_json(self.root / 'assignment-hash.json', {'sha256': digest(self.assignment)})
        source = self.root / 'human.jsonl'
        source.write_text(json.dumps({'prompt': 'A boat', 'other_axes': ['Boat type']}) + '\n')
        self.bank = dict(
            kind='reviewed_axis_examples',
            reviewer='test fixture',
            examples=[
                dict(
                    id='e1',
                    source=str(source),
                    line=1,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_group='pilot',
                    participant='test',
                    prompt='A boat',
                    requests=['Boat type'],
                    approved=True,
                    review_note='Synthetic test fixture only',
                    provenance='human_free_text',
                    images=[],
                )
            ],
        )
        self.bp = self.root / 'bank.json'
        atomic_json(self.bp, self.bank)
        self.cfg = dict(
            api_base='https://example.test/v1',
            model='test-model',
            reasoning_effort='medium',
            timeout=30,
            max_tokens=1000,
            max_axes=4,
            example_count=1,
            seed=42,
            detail='high',
        )
        self.images = [
            dict(data=b'test image ' + bytes([i]), sha256=str(i), receipt={}, path=str(i))
            for i in range(4)
        ]

    def test_assignment_balanced_disjoint_order_independent(self):
        a, unused = assign(self.rows, 3, 17)
        b, _ = assign(list(reversed(self.rows)), 3, 17)
        self.assertEqual(a, b)
        self.assertEqual(len(set(r['id'] for r in a)), 12)
        self.assertFalse(set(unused) & {r['id'] for r in a})
        self.assertEqual(
            sorted(__import__('collections').Counter(r['renderer'] for r in a).values()), [3] * 4
        )
        self.assertNotEqual(a, assign(self.rows, 3, 18)[0])
        with self.assertRaises(ValueError):
            assign(self.rows + [self.rows[0]], 2)

    def test_schema_rejects_padding_and_duplicate_axes(self):
        self.assertEqual(policy.parse(json.dumps(answer()), 4), answer())
        v = answer()
        v['axes'] *= 2
        with self.assertRaises(ValueError):
            policy.parse(json.dumps(v), 4)
        v = answer()
        v['axes'][0]['tpips_factor'] = 'clock color and shape'
        with self.assertRaises(ValueError):
            policy.parse(json.dumps(v), 4)
        v = dict(
            status='no_variation', reason='Already sufficient variety', axes=[], alignment_notes=''
        )
        self.assertEqual(policy.parse(json.dumps(v), 4)['axes'], [])

    def test_examples_verbatim_and_no_same_prompt_leakage(self):
        load_bank(self.bp)
        self.assertEqual(len(pick_examples(self.bank, self.selected[0], 1, 42)), 1)
        with self.assertRaises(ValueError):
            pick_examples(self.bank, {'id': 'boat', 'prompt': ' A BOAT '}, 1, 42)
        self.bank['examples'][0]['requests'] = ['Invented request']
        atomic_json(self.bp, self.bank)
        with self.assertRaises(ValueError):
            load_bank(self.bp)

    def test_request_contains_four_images_no_t2i_model_or_participant(self):
        payload, audit = build_request(
            self.selected[0], self.images, self.bank['examples'], self.cfg
        )
        text = json.dumps(payload)
        self.assertNotIn(self.selected[0]['renderer'], text)
        self.assertNotIn('participant', text)
        self.assertEqual(text.count('data:image/png;base64,'), 4)
        changed = copy.deepcopy(self.images)
        changed[0]['data'] = b'changed'
        self.assertNotEqual(
            audit['payload_hash'],
            build_request(self.selected[0], changed, self.bank['examples'], self.cfg)[1][
                'payload_hash'
            ],
        )

    def test_budget_resume_and_config_change_guard(self):
        sent = []

        def factory(cfg, key):
            def client(payload):
                sent.append(payload)
                return Reply(
                    json.dumps(answer()), 'stop', {'total_tokens': 10}, 'actual-test-model'
                )

            return client

        args = (self.ap, self.bp, self.root / 'run', self.cfg)
        with patch('elicitation.runner.verified_images', return_value=self.images):
            dry = run(*args, limit=2, client_factory=factory)
            self.assertEqual(dry['ready'], 2)
            self.assertEqual(sent, [])
            first = run(
                *args, limit=2, execute=True, max_calls=1, api_key='fixture', client_factory=factory
            )
            self.assertEqual(first['completed'], 1)
            second = run(
                *args, limit=2, execute=True, max_calls=2, api_key='fixture', client_factory=factory
            )
            self.assertTrue(second['complete'])
            self.assertEqual(len(sent), 2)
            run(*args, limit=2, execute=True, max_calls=2, client_factory=factory)
            self.assertEqual(len(sent), 2)
            self.cfg['model'] = 'changed'
            with self.assertRaises(ValueError):
                run(*args, limit=2, execute=True, max_calls=3)

    def test_uncertain_call_is_not_automatically_repeated(self):
        def factory(cfg, key):
            def crash(payload):
                raise RuntimeError('simulated interrupt')

            return crash

        out = self.root / 'run'
        with patch('elicitation.runner.verified_images', return_value=self.images):
            with self.assertRaises(RuntimeError):
                run(
                    self.ap,
                    self.bp,
                    out,
                    self.cfg,
                    limit=1,
                    execute=True,
                    max_calls=1,
                    api_key='fixture',
                    client_factory=factory,
                )
            # Also covers interruption between global reservation and per-item save.
            (out / 'items' / self.selected[0]['id'] / 'attempt.json').unlink()
            result = run(self.ap, self.bp, out, self.cfg, limit=1, execute=True, max_calls=2)
            self.assertEqual(result['blocked'], 1)
            self.assertEqual(result['api_calls_reserved'], 1)

    def test_missing_images_and_invalid_reply_do_not_become_labels(self):
        with patch('elicitation.runner.verified_images', return_value=None):
            r = run(
                self.ap,
                self.bp,
                self.root / 'pending',
                self.cfg,
                limit=1,
                execute=True,
                max_calls=0,
            )
            self.assertEqual(r['pending_images'], 1)
            self.assertFalse(r['complete'])
        with patch('elicitation.runner.verified_images', return_value=self.images):
            r = run(
                self.ap,
                self.bp,
                self.root / 'bad',
                self.cfg,
                limit=1,
                execute=True,
                max_calls=1,
                api_key='fixture',
                client_factory=lambda *a: lambda p: Reply('oops', 'stop', {}, 'test'),
            )
            self.assertEqual(r['invalid'], 1)
            self.assertFalse(r['complete'])
            self.assertTrue(
                (self.root / 'bad/items' / self.selected[0]['id'] / 'reply.json').exists()
            )

    def test_receipts_detect_wrong_prompt_seed_and_corrupt_image(self):
        from PIL import Image
        from baselines.assets import seed_for

        row = self.selected[0]
        cfg = dict(seed=42, resolution=8)
        manifest = {'config': cfg}
        root = self.root / 'render' / row['renderer'] / f"shard-{row['shard']:02d}"
        atomic_json(root / 'manifest.json', manifest)
        assignment = dict(
            k=4,
            render_root=str(self.root / 'render'),
            manifests={row['renderer']: {str(row['shard']): manifest}},
        )
        self.assertIsNone(verified_images(assignment, row))
        for index in range(4):
            folder = root / 'images' / row['id'] / f'{index:04d}'
            folder.mkdir(parents=True)
            path = folder / 'image.png'
            Image.new('RGB', (8, 8), (index, 0, 0)).save(path)
            raw = path.read_bytes()
            atomic_json(
                folder / 'receipt.json',
                dict(
                    run_hash=digest(manifest),
                    prompt_id=row['id'],
                    index=index,
                    seed=seed_for(row['id'], 42, index),
                    render_prompt=row['prompt'],
                    sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw),
                ),
            )
        self.assertEqual(len(verified_images(assignment, row)), 4)
        path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            verified_images(assignment, row)

    def test_mixture_keeps_reward_wording_independent(self):
        rows = [
            dict(
                prompt_id=str(i),
                axis_id='a',
                cluster_id=str(i),
                split='train',
                renderer='model',
                request='Please vary the time displayed on the clock',
                tpips_factor='clock time',
                mapping_status='faithful',
                constraints=[],
                semantic_review='approved',
            )
            for i in range(8)
        ]
        mixed = mix(rows, 0.75, reward_wording='short')
        self.assertEqual(sum(r['input_style'] == 'tpips' for r in mixed), 6)
        self.assertTrue(all(r['reward_factor'] == 'clock time' for r in mixed))
        self.assertTrue(
            all(
                r['reward_factor'] == r['request']
                for r in mix(rows, 0.75, reward_wording='natural')
            )
        )
        rows[0]['constraints'] = ['Only times after noon']
        with self.assertRaises(ValueError):
            mix(rows, 0.75, reward_wording='short')


if __name__ == '__main__':
    unittest.main()
