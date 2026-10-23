import json
from pathlib import Path
import tempfile
import unittest

from tdig import instruction as I
from tdig.build_dataset import is_hard_contrastive, partition, rewrite
from tdig.iid_vqa import merge

BACKEND = Path(__file__).resolve().parents[1] / 'axis_framework/backend'


def row(i, prompt, axis, slot, phase, kind='axis', split='train'):
    spec = dict(base_prompt=f'base {prompt}', axis=axis, tpips_factor=axis, k=4)
    return dict(
        data_source='axis_diversity',
        prompt=[dict(role='user', content='old')],
        reward_model=dict(style='model', ground_truth=json.dumps(spec)),
        extra_info=dict(
            index=i,
            row_id=f'{prompt}:{axis}',
            prompt_id=prompt,
            base_prompt=f'base {prompt}',
            axis=axis,
            axis_kind=kind,
            axis_slot=slot,
            pass_number=phase,
            k=4,
            split=split,
        ),
    )


class InstructionTest(unittest.TestCase):
    def test_templates_equal_the_backend_constants(self):
        """The instruction templates are byte-identical to their source constants in axis_framework/backend."""
        # Compare against the source text. A backslash at a line end continues a string literal, so
        # the constant's value omits it.
        source = (BACKEND / 'instructions.py').read_text()
        source = source.replace('\\\n', '')
        for name in ('INSTRUCTION_V4', 'LENGTH_RULE', 'INSTRUCTION_OVERALL'):
            self.assertIn(getattr(I, name), source, name)
        self.assertIn(
            'LENGTH_BUDGETS = {"sd3": (60, 77), "flux": (150, 512), "qwen": (150, 512)}', source
        )

    def test_named_axis_instruction(self):
        """A named axis selects the V4 template with the T2I model's cap and a fixed rule 3."""
        text = I.instruction('A red chair', 'Show the chair from above', 4, 'sd35-large')
        self.assertIn('Base prompt: A red chair\nAxis: Show the chair from above\n', text)
        self.assertIn('3. Vary ONLY the given axis, and together', text)
        self.assertIn('UNDER 60 WORDS', text)
        self.assertIn('first 77 tokens', text)
        self.assertNotIn("Include the base prompt's wording", text)
        self.assertNotIn('{', text)
        self.assertIn(
            'UNDER 150 WORDS', I.instruction('A red chair', 'Color', 4, 'qwen-image-2512')
        )

    def test_overall_instruction(self):
        """The axis `overall` selects the overall template, with rule 4 naming no axis."""
        text = I.instruction('A red chair', 'overall', 4, 'flux2-klein-9b')
        self.assertIn('Axis: overall\n', text)
        self.assertIn('MUST NOT ALL DIFFER IN THE SAME ONE WAY', text)
        self.assertIn('Add what you are changing and stop.\n5. Keep every detail', text)
        self.assertNotIn('Add the axis change', text)
        self.assertNotIn('{', text)

    def test_image_context_and_errors(self):
        """Image variants carry four placeholders; invalid arguments raise."""
        for axis in ('Color', 'overall'):
            text = I.instruction('A red chair', axis, 4, 'flux2-klein-9b', images=True)
            self.assertEqual(text.count('<image>'), 4)
            self.assertLess(text.index('<image>'), text.index('Write exactly 4'))
        for args in (
            ('', 'Color', 4, 'sd35-medium'),
            ('A', ' ', 4, 'sd35-medium'),
            ('A', 'Color', 1, 'sd35-medium'),
            ('A', 'Color', 4, 'dall-e'),
        ):
            with self.assertRaises(ValueError):
                I.instruction(*args)


class FilterTest(unittest.TestCase):
    def test_hard_contrastive(self):
        """Exclusion requests are contrastive; paired and positional phrases are not."""
        for axis in (
            'Show full-body shots rather than portraits',
            'LED screens instead of billboards',
            'Sky background without a full moon',
            'Remove the title text',
        ):
            self.assertTrue(is_hard_contrastive(axis), axis)
        for axis in (
            'overall',
            'Camera angle',
            'Canvas presentation with or without an ornate frame',
            'Show portraits both with and without text banners.',
            'With/without rider',
            'Woman facing toward or away from the camera',
            'Non-humanoid creature designs',
            'Vary the mech proportions beyond a large head',
            'Include tennis players',
        ):
            self.assertFalse(is_hard_contrastive(axis), axis)

    def test_partition(self):
        """Filtering keeps order and passes and reindexes from 0."""
        train = [
            row(0, 'a', 'Color', 0, 1),
            row(1, 'b', 'X instead of Y', 0, 1),
            row(2, 'a', 'overall', 0, 1, 'overall'),
            row(3, 'a', 'Pose rather than none', 1, 2),
            row(4, 'b', 'Material', 1, 2),
        ]
        val = [
            row(0, 'v', 'Seated rather than standing', 0, 1, split='val'),
            row(1, 'v', 'Hat style', 1, 2, split='val'),
            row(2, 'v', 'overall', 0, 1, 'overall', 'val'),
            row(3, 'v', 'Shoe colour', 2, 3, split='val'),
        ]
        sets = partition(train, val)
        self.assertEqual(
            [r['extra_info']['axis'] for r in sets['train']], ['Color', 'overall', 'Material']
        )
        self.assertEqual([r['extra_info']['index'] for r in sets['train']], [0, 1, 2])
        self.assertEqual([r['extra_info']['tdig_source_index'] for r in sets['train']], [0, 2, 4])
        self.assertEqual(len(sets['dropped-contrastive-train']), 2)
        self.assertEqual(
            [r['extra_info']['axis'] for r in sets['probe-contrastive']],
            ['Seated rather than standing'],
        )
        self.assertEqual(train[1]['extra_info']['index'], 1, 'sources are not modified')

    def test_rotated_order(self):
        """Rotation keeps every row once, mixes slots in each pass, and keeps the pass body unique."""
        from collections import Counter
        from tdig.build_dataset import rotate_slots

        rows, i = [], 0
        for phase in range(1, 5):
            for n in range(40):
                if not (n == 7 and phase == 4):  # prompt 7 has three axes
                    rows.append(row(i, f'p{n}', f'axis {phase} of {n}', phase - 1, phase))
                    i += 1
                if (
                    n % 10 == 3 and n // 10 + 1 == phase
                ):  # one overall row each for p3, p13, p23, p33
                    rows.append(row(i, f'p{n}', 'overall', 0, phase, 'overall'))
                    i += 1
        rows.append(row(i, 'only-overall', 'overall', 0, 2, 'overall'))
        out = rotate_slots(rows)
        self.assertEqual(
            sorted(r['extra_info']['row_id'] for r in out),
            sorted(r['extra_info']['row_id'] for r in rows),
        )
        passes = [r['extra_info']['pass_number'] for r in out]
        self.assertEqual(passes, sorted(passes))
        first = [r['extra_info'] for r in out if r['extra_info']['pass_number'] == 1]
        self.assertEqual(len({e['axis_slot'] for e in first if e['axis_kind'] == 'axis'}), 4)
        body = first[:40]  # one row per prompt, then the tail
        self.assertEqual(len({e['prompt_id'] for e in body}), 40)
        self.assertEqual([e['axis'] for e in body if e['prompt_id'] == 'p3'], ['overall'])
        self.assertEqual([e['prompt_id'] for e in first[40:]], ['p3'])
        # Over the four passes each prompt shows each of its axes once.
        self.assertTrue(
            all(v == 1 for v in Counter(e['extra_info']['row_id'] for e in out).values())
        )
        self.assertEqual(rows[0]['extra_info']['pass_number'], 1, 'sources are not modified')

    def test_rewrite_keeps_reward_target(self):
        """Rewriting changes the prompt and instruction fields only."""
        source = row(0, 'a', 'Color', 0, 1)
        result = rewrite(source, 'sd35-medium', images=False)
        self.assertEqual(result['reward_model'], source['reward_model'])
        self.assertIn('Axis: Color', result['prompt'][0]['content'])
        self.assertEqual(result['extra_info']['instruction_version'], I.VERSION)
        self.assertEqual(source['prompt'][0]['content'], 'old')
        with self.assertRaises(ValueError):
            rewrite(source, 'sd35-medium', images=True)


class ThresholdTest(unittest.TestCase):
    def test_merge(self):
        """The threshold is the mean per T2I model and prompt; incomplete input raises."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            index = [
                dict(
                    renderer='sd35-medium',
                    prompt_id='p',
                    image_index=i,
                    path=f'/x/{i}.png',
                    base_prompt='P',
                )
                for i in range(4)
            ] + [
                dict(
                    renderer='sd35-large',
                    prompt_id='p',
                    image_index=1,
                    path='/y.png',
                    base_prompt='P',
                )
            ]
            (d / 'index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in index))
            scores = [0.2, 0.4, 0.6, 0.8, 0.5]
            for shard in range(2):
                part = [
                    dict(
                        renderer=r['renderer'], prompt_id='p', image_index=r['image_index'], score=s
                    )
                    for r, s in list(zip(index, scores))[shard::2]
                ]
                (d / f'scores-{shard:02d}.jsonl').write_text(
                    ''.join(json.dumps(r) + '\n' for r in part)
                )
            with self.assertRaises(ValueError):
                merge(d, 2)
            for shard in range(2):
                (d / f'done-{shard:02d}.json').write_text('{}')
            merge(d, 2)
            table = {
                r['renderer']: r
                for r in map(json.loads, (d / 'thresholds.jsonl').read_text().splitlines())
            }
            self.assertAlmostEqual(table['sd35-medium']['align_thresh'], 0.5)
            self.assertEqual(table['sd35-medium']['scores'], [0.2, 0.4, 0.6, 0.8])
            self.assertEqual(
                (table['sd35-large']['images'], table['sd35-large']['align_thresh']), (1, 0.5)
            )


class EditDatasetTest(unittest.TestCase):
    def test_edit_instruction(self):
        """k ordered image placeholders; named axes and `overall` get different rule 3 text."""
        text = I.edit_instruction('A red chair', 'Chair material', 4)
        self.assertEqual(
            [
                text.index(f'Image {i}: <image>') < text.index(f'Image {i+1}: <image>')
                for i in (1, 2, 3)
            ],
            [True] * 3,
        )
        self.assertEqual(text.count('<image>'), 4)
        self.assertIn('3. Vary ONLY the given axis between the 4 instructions', text)
        self.assertIn('Keep it under 30 words.', text)
        self.assertNotIn('{', text)
        overall = I.edit_instruction('A red chair', 'overall', 4)
        self.assertIn('MUST NOT ALL CHANGE THE SAME ONE THING', overall)
        self.assertNotIn('Vary ONLY', overall)
        with self.assertRaises(ValueError):
            I.edit_instruction('A red chair', 'Color', 1)

    def test_edit_row_keeps_the_reward_target_and_adds_sources(self):
        """The row keeps axis and threshold, gains sources for the editor and inputs for the policy."""
        from axis_grpo.training.protocol import parse_spec
        from tdig.build_edit_dataset import edit_row

        source = row(0, 'a', 'Color', 0, 1)
        spec = json.loads(source['reward_model']['ground_truth'])
        spec['align_thresh'] = 0.8
        source['reward_model']['ground_truth'] = json.dumps(spec)
        full = [f'/full/{i}.png' for i in range(4)]
        result = edit_row(source, full, [(f'/in/{i}.png', f'hash{i}') for i in range(4)])
        parsed = parse_spec(result['reward_model']['ground_truth'])
        self.assertEqual(
            (parsed['task'], parsed['source_images'], parsed['align_thresh'], parsed['axis']),
            ('edit', full, 0.8, 'Color'),
        )
        self.assertEqual(result['images'], [f'/in/{i}.png' for i in range(4)])
        self.assertEqual(result['prompt'][0]['content'].count('<image>'), len(result['images']))
        self.assertEqual(result['extra_info']['input_image_sha256'], [f'hash{i}' for i in range(4)])
        self.assertNotIn('task', json.loads(source['reward_model']['ground_truth']))
        with self.assertRaises(ValueError):
            edit_row(source, full[:3], [('/in/0.png', 'h')] * 3)


class BaselineScoreTest(unittest.TestCase):
    def test_one_set_is_scored_once_per_axis(self):
        """A held-out set yields one line per axis, sharing its VQAScore; a COCO row yields one line."""
        from tdig.score_baselines import merge as merge_baselines, _factors

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sets = [
                dict(
                    group='r|iid|heldout-axes',
                    prompt_id='h:1',
                    base_prompt='P',
                    paths=[],
                    align_thresh=0.8,
                    factors=[
                        dict(
                            row_id='a1',
                            factor='Camera distance',
                            axis_kind='axis',
                            contrastive=False,
                        ),
                        dict(
                            row_id='a2', factor='X instead of Y', axis_kind='axis', contrastive=True
                        ),
                        dict(row_id='a3', factor='overall', axis_kind='overall', contrastive=False),
                    ],
                ),
                dict(
                    group='r|iid|coco2017-val-5000',
                    prompt_id='coco2017:000000000001',
                    base_prompt='Q',
                    paths=[],
                ),
            ]
            self.assertEqual(_factors(sets[1])[0]['row_id'], 'coco2017:000000000001')
            (d / 'sets.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in sets))
            vqa = [
                dict(
                    renderer=s['group'],
                    prompt_id=s['prompt_id'],
                    align_thresh=0.7,
                    score_min=0.5,
                    scores=[0.5, 0.7, 0.8, 0.8],
                )
                for s in sets
            ]
            (d / 'thresholds.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in vqa))
            tp = [
                dict(
                    group=sets[0]['group'],
                    prompt_id='h:1',
                    row_id=r,
                    normalized_vendi=v,
                    pairwise=v / 2,
                )
                for r, v in (('a1', 0.2), ('a2', 0.4), ('a3', 0.6))
            ]
            tp.append(
                dict(
                    group=sets[1]['group'],
                    prompt_id='coco2017:000000000001',
                    normalized_vendi=0.3,
                    pairwise=0.1,
                )
            )  # no row_id
            (d / 'tpips-00.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in tp))
            with self.assertRaises(ValueError):
                merge_baselines(d, 1)
            (d / 'tpips-done-00.json').write_text('{}')
            summary = merge_baselines(d, 1)
            held = summary['r|iid|heldout-axes']
            self.assertEqual(held['lines'], 3)
            self.assertAlmostEqual(held['normalized_vendi'], 0.4)
            # penalty = mean(max(0, 0.8 - a)) over [0.5, 0.7, 0.8, 0.8] = 0.1
            self.assertAlmostEqual(held['fidelity_penalty'], 0.1)
            self.assertAlmostEqual(held['per_kind']['axis']['reward'], 0.2 - 0.1)
            self.assertEqual(sorted(held['per_kind']), ['axis', 'contrastive', 'overall'])
            self.assertEqual(summary['r|iid|coco2017-val-5000']['lines'], 1)
            self.assertNotIn('reward', summary['r|iid|coco2017-val-5000'])


class PaperMetricsTest(unittest.TestCase):
    def test_merge_joins_base_and_own_scores_per_axis(self):
        """One line per set and axis; own-prompt scores appear only for methods that rewrite."""
        from tdig.paper_metrics import _write_index, merge as merge_metrics, IMAGE_BASE, SET_LEVEL

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            factor = lambda rid, kind='axis', c=False: dict(
                row_id=rid, factor=rid, axis_kind=kind, contrastive=c
            )
            sets = [
                dict(
                    group='r|pag|heldout-axes',
                    set_id='h:1',
                    base_prompt='P',
                    own_prompts=['a', 'b'],
                    paths=['/1.png', '/2.png'],
                    align_thresh=0.8,
                    factors=[factor('x'), factor('y', c=True)],
                ),
                dict(
                    group='r|iid|coco2017-val-5000',
                    set_id='g:0',
                    base_prompt='Q',
                    own_prompts=None,
                    paths=['/3.png', '/4.png'],
                    align_thresh=None,
                    factors=[factor('g:0', 'overall')],
                ),
            ]
            groups = _write_index(d, sets)
            self.assertEqual(groups['r|pag|heldout-axes'], dict(sets=1, rewritten=1))
            index = [json.loads(l) for l in (d / 'index.jsonl').read_text().splitlines()]
            self.assertEqual(
                sorted({r['renderer'] for r in index}),
                [
                    'r|iid|coco2017-val-5000||base',
                    'r|pag|heldout-axes||base',
                    'r|pag|heldout-axes||own',
                ],
            )
            self.assertEqual(
                [r['base_prompt'] for r in index if r['renderer'].endswith('||own')], ['a', 'b']
            )
            vqa = [
                dict(
                    renderer='r|pag|heldout-axes||base',
                    prompt_id='h:1',
                    align_thresh=0.7,
                    score_min=0.6,
                    scores=[0.6, 0.8],
                ),
                dict(
                    renderer='r|pag|heldout-axes||own',
                    prompt_id='h:1',
                    align_thresh=0.9,
                    score_min=0.85,
                    scores=[0.85, 0.95],
                ),
                dict(
                    renderer='r|iid|coco2017-val-5000||base',
                    prompt_id='g:0',
                    align_thresh=0.5,
                    score_min=0.4,
                    scores=[0.4, 0.6],
                ),
            ]
            (d / 'thresholds.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in vqa))
            image = {k: [1.0, 3.0] for k in IMAGE_BASE}
            level = {k: 0.25 for k in SET_LEVEL}
            scored = [
                dict(
                    group='r|pag|heldout-axes',
                    set_id='h:1',
                    **image,
                    **level,
                    clipscore_own=[0.5, 0.7],
                    tpips={
                        'x': dict(tpips_vendi=0.4, tpips_pairwise=0.2),
                        'y': dict(tpips_vendi=0.6, tpips_pairwise=0.3),
                    },
                ),
                dict(
                    group='r|iid|coco2017-val-5000',
                    set_id='g:0',
                    **image,
                    **level,
                    tpips={'g:0': dict(tpips_vendi=0.1, tpips_pairwise=0.05)},
                ),
            ]
            (d / 'metrics-00.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in scored))
            (d / 'metrics-done-00.json').write_text('{}')
            summary = merge_metrics(d, 1)
            pag = summary['r|pag|heldout-axes']
            self.assertEqual(pag['lines'], 2)
            self.assertAlmostEqual(pag['tpips_vendi'], 0.5)
            self.assertAlmostEqual(pag['clipscore'], 2.0)
            self.assertNotIn(
                'clipscore_own', pag
            )  # 77-token cap truncates rewritten prompts; rows.jsonl only
            self.assertAlmostEqual(pag['vqascore_own_mean'], 0.9)
            self.assertAlmostEqual(
                pag['fidelity_penalty'], 0.1
            )  # mean(max(0, .8-.6), max(0, .8-.8))
            self.assertAlmostEqual(pag['per_kind']['axis']['reward'], 0.4 - 0.1)
            self.assertEqual(sorted(pag['per_kind']), ['axis', 'contrastive'])
            iid = summary['r|iid|coco2017-val-5000']
            self.assertNotIn('vqascore_own_mean', iid)
            self.assertNotIn('reward', iid)
            self.assertEqual(len((d / 'summary.csv').read_text().splitlines()), 3)


class FrontierTest(unittest.TestCase):
    def test_frontier_orders_from_aligned_to_diverse_and_drops_beaten_steps(self):
        from tdig.select_frontier import frontier

        points = {
            0: {'alignment': 0.86, 'diversity': 0.35},
            50: {'alignment': 0.84, 'diversity': 0.42},
            100: {'alignment': 0.80, 'diversity': 0.41},
            200: {'alignment': 0.81, 'diversity': 0.49},
        }
        self.assertEqual(frontier(points), [0, 50, 200])

    def test_margin_treats_small_differences_as_ties(self):
        from tdig.select_frontier import frontier

        points = {
            200: {'alignment': 0.829, 'diversity': 0.471},
            400: {'alignment': 0.850, 'diversity': 0.470},
        }
        self.assertEqual(frontier(points), [400, 200])
        self.assertEqual(frontier(points, margin=0.005), [400])

    def test_malformed_rows_count_as_zero_diversity_and_prompts_split_stably(self):
        from tdig.select_frontier import half, summarize

        rows = [
            {'metrics': {'malformed': 0.0, 'normalized_vendi': 0.6, 'fidelity_mean': 0.9}},
            {'metrics': {'malformed': 1.0}},
        ]
        out = summarize(rows)
        self.assertAlmostEqual(out['diversity'], 0.3)
        self.assertAlmostEqual(out['alignment'], 0.9)
        self.assertAlmostEqual(out['malformed'], 0.5)
        self.assertEqual(half('a cat', 0.5), half('a cat', 0.5))
        self.assertEqual({half(str(i), 1.0) for i in range(50)}, {'select'})


class TextRequestTest(unittest.TestCase):
    def test_flags_requested_caption_but_not_a_base_prompt_that_has_one(self):
        from tdig.select_frontier import asks_for_text

        hack = (
            "<prompt>A desert at dusk. Text at the bottom reads: 'a girl in a black dress'</prompt>"
        )
        self.assertTrue(asks_for_text(hack, 'a girl in a black dress'))
        self.assertFalse(
            asks_for_text(
                '<prompt>A girl in a black dress at dusk</prompt>', 'a girl in a black dress'
            )
        )
        self.assertFalse(
            asks_for_text(
                "<prompt>A sign that says 'open' at night</prompt>", "a sign that says 'open'"
            )
        )


if __name__ == '__main__':
    unittest.main()
