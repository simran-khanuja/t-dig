import copy
from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from axis_grpo.dataset import (
    audit,
    available_schedule,
    load_plan,
    make_plan,
    read_record,
    schedule,
    training_row,
    validate_loader_config,
)
from axis_grpo.prepare import main, new_directory
from baselines.assets import seed_for
from elicitation.selection import T2I_MODELS
from hpdv2.storage import atomic_json, digest, prompt_id


def assignment(n=40):
    items = [
        dict(
            id=prompt_id(f'A red chair beside a window {i}'),
            prompt=f'A red chair beside a window {i}',
            cluster_id=f'cluster-{i}',
            renderer=T2I_MODELS[i % 4],
            shard=0,
        )
        for i in range(n)
    ]
    manifests = {
        r: {'0': dict(config=dict(seed=42, resolution=1024, renderer=r))} for r in T2I_MODELS
    }
    return dict(items=items, k=4, render_root='/fixture/images', manifests=manifests)


def response():
    return dict(
        status='ok',
        reason='',
        alignment_notes='',
        axes=[
            dict(
                request=request,
                tpips_factor=f'factor {i}',
                observation='Same feature in the images',
                why_free='The base prompt leaves this open',
                mapping_status='faithful',
                constraints=[],
            )
            for i, request in enumerate(
                [
                    'Chair material',
                    'Can the window shapes look different?',
                    'Different viewing angles — perhaps from above',
                    'Amount of daylight',
                ]
            )
        ],
    )


def save_assignment(root, value):
    atomic_json(root / 'assignment.json', value)
    atomic_json(root / 'assignment-hash.json', {'sha256': digest(value)})


def save_item(folder, row, value):
    images = []
    manifest = value['manifests'][row['renderer']]['0']
    for i in range(4):
        sha = str(i) * 64
        receipt = dict(
            prompt_id=row['id'],
            render_prompt=row['prompt'],
            index=i,
            seed=seed_for(row['id'], 42, i),
            run_hash=digest(manifest),
            sha256=sha,
        )
        path = (
            Path(value['render_root'])
            / row['renderer']
            / 'shard-00/images'
            / row['id']
            / f'{i:04d}/image.png'
        )
        images.append(dict(path=str(path), sha256=sha, receipt=receipt))
    answer = response()
    atomic_json(folder / 'request.json', dict(row=row, payload_hash='test-hash', images=images))
    atomic_json(
        folder / 'reply.json',
        dict(text=json.dumps(answer), finish_reason='stop', refusal=None, model='fixture'),
    )
    atomic_json(
        folder / 'result.json',
        dict(
            row=row,
            payload_hash='test-hash',
            response=answer,
            actual_model='fixture',
            semantic_review='pending',
        ),
    )


def source_run(root, value):
    save_assignment(root, value)
    for row in value['items']:
        save_item(root / 'items' / row['id'], row, value)
    atomic_json(
        root / 'manifest.json',
        dict(
            config=dict(max_axes=4),
            assignment_hash=digest(value),
            source_hash='fixture-source',
            bank_hash='fixture-bank',
            ids=[r['id'] for r in value['items']],
        ),
    )


class TrainingPlanTests(unittest.TestCase):
    def test_full_epoch_counts_order_and_early_overall(self):
        a = assignment(20000)
        plan = make_plan(a)
        rows = list(schedule(plan))
        self.assertEqual(len(rows), 82000)
        self.assertEqual([e['index'] for e in rows], list(range(82000)))
        self.assertEqual(len({e['row_id'] for e in rows}), 82000)
        overall = [e for e in rows if e['axis_kind'] == 'overall']
        self.assertEqual(len({e['prompt_id'] for e in overall}), 2000)
        self.assertEqual([e['index'] for e in overall], list(range(40, 82000, 41)))
        by_id = {r['id']: r for r in a['items']}
        for phase in range(1, 5):
            part = [e for e in rows if e['pass_number'] == phase]
            axes = [e for e in part if e['axis_kind'] == 'axis']
            ov = [e for e in part if e['axis_kind'] == 'overall']
            self.assertEqual(len(part), 20500)
            self.assertEqual([e['prompt_id'] for e in axes], plan['prompt_order'])
            self.assertEqual({e['axis_slot'] for e in axes}, {phase - 1})
            self.assertEqual(
                Counter(by_id[e['prompt_id']]['renderer'] for e in axes),
                Counter(dict.fromkeys(T2I_MODELS, 5000)),
            )
            self.assertEqual(
                Counter(by_id[e['prompt_id']]['renderer'] for e in ov),
                Counter(dict.fromkeys(T2I_MODELS, 125)),
            )
        self.assertEqual(
            Counter(by_id[e['prompt_id']]['renderer'] for e in overall),
            Counter(dict.fromkeys(T2I_MODELS, 500)),
        )

    def test_reproducible_order_independent_of_source_list_order(self):
        a = assignment(80)
        p = make_plan(a, expected_prompts=80)
        b = copy.deepcopy(a)
        b['items'].reverse()
        q = make_plan(b, expected_prompts=80)
        self.assertEqual(p['prompt_order'], q['prompt_order'])
        self.assertEqual(p['overall_by_block_and_pass'], q['overall_by_block_and_pass'])
        self.assertNotEqual(
            p['prompt_order'], make_plan(a, seed=9, expected_prompts=80)['prompt_order']
        )
        self.assertEqual(list(schedule(p)), list(schedule(make_plan(a, expected_prompts=80, k=8))))

    def test_reject_wrong_count_duplicates_changed_text_and_unbalanced_t2i_models(self):
        a = assignment()
        for edit in (
            lambda x: x['items'].pop(),
            lambda x: x['items'].__setitem__(1, x['items'][0]),
            lambda x: x['items'][0].update(prompt='wrong'),
            lambda x: x['items'][0].update(renderer=T2I_MODELS[1]),
        ):
            b = copy.deepcopy(a)
            edit(b)
            with self.assertRaises(ValueError):
                make_plan(b, expected_prompts=40)
        with self.assertRaises(ValueError):
            make_plan(a)

    def test_exact_natural_wording_reward_choice_and_overall_instruction(self):
        a = assignment()
        p = make_plan(a, expected_prompts=40)
        by_id = {r['id']: r for r in a['items']}
        entries = list(schedule(p))
        record = dict(response=response(), payload_hash='test')
        entry = next(e for e in entries if e['axis_slot'] == 2)
        row = training_row(entry, by_id[entry['prompt_id']], p, record, 'tpips_factor')
        self.assertIn('Different viewing angles — perhaps from above', row['prompt'][0]['content'])
        self.assertNotIn('factor 2', row['prompt'][0]['content'])
        self.assertEqual(
            json.loads(row['reward_model']['ground_truth'])['tpips_factor'], 'factor 2'
        )
        raw = training_row(entry, by_id[entry['prompt_id']], p, record)
        self.assertEqual(
            json.loads(raw['reward_model']['ground_truth'])['tpips_factor'],
            response()['axes'][2]['request'],
        )
        self.assertEqual(raw['extra_info']['axis'], raw['extra_info']['tpips_factor'])
        overall = training_row(entries[40], by_id[entries[40]['prompt_id']], p, record)
        spec = json.loads(overall['reward_model']['ground_truth'])
        self.assertEqual((spec['axis'], spec['tpips_factor'], spec['k']), ('overall', 'overall', 4))
        self.assertNotIn('keep unrelated details', overall['prompt'][0]['content'])
        self.assertIn('multiple attributes', overall['prompt'][0]['content'])
        self.assertIn('Preserve every explicit requirement', overall['prompt'][0]['content'])

    def test_loader_guard_prevents_reordering_or_dropped_tail(self):
        c = dict(
            data=dict(
                shuffle=False,
                train_max_samples=-1,
                filter_overlong_prompts=False,
                truncation='error',
                train_batch_size=40,
            ),
            trainer=dict(total_epochs=1),
        )
        self.assertEqual(validate_loader_config(c, 82000)['steps'], 2050)
        for key, value in [
            ('shuffle', True),
            ('filter_overlong_prompts', True),
            ('train_max_samples', 20000),
            ('truncation', 'left'),
            ('train_batch_size', 128),
            ('gen_batch_size', 64),
            ('sampler', {'class_path': 'anything'}),
        ]:
            b = copy.deepcopy(c)
            b['data'][key] = value
            with self.assertRaises(ValueError):
                validate_loader_config(b, 82000)
        c['trainer']['total_epochs'] = 2
        with self.assertRaises(ValueError):
            validate_loader_config(c, 82000)


class TrainingImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = assignment()
        self.plan = make_plan(self.a, expected_prompts=40)
        self.run = self.root / 'source'
        source_run(self.run, self.a)
        self.row = self.a['items'][1]
        self.folder = self.run / 'items' / self.row['id']

    def change_answer(self, edit):
        answer = response()
        edit(answer)
        r = json.loads((self.folder / 'result.json').read_text())
        r['response'] = answer
        raw = json.loads((self.folder / 'reply.json').read_text())
        raw['text'] = json.dumps(answer)
        atomic_json(self.folder / 'result.json', r)
        atomic_json(self.folder / 'reply.json', raw)

    def test_complete_readonly_import_keeps_four_images(self):
        before = {str(p): p.read_bytes() for p in self.run.rglob('*.json')}
        records, report = audit(self.a, self.plan, self.run)
        self.assertTrue(report['complete'])
        self.assertEqual(report['by_status'], {'ready': 40})
        self.assertEqual(len(records[self.row['id']]['images']), 4)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.run.rglob('*.json')})

    def test_missing_fourth_axis_is_not_padded(self):
        self.change_answer(lambda r: r['axes'].pop())
        records, report = audit(self.a, self.plan, self.run)
        self.assertTrue(report['complete'])
        self.assertEqual(report['committed_results_by_axis_count']['3'], 1)
        self.assertEqual(report['training_rows'], 163)
        self.assertEqual(len(records[self.row['id']]['response']['axes']), 3)
        rows = list(available_schedule(self.plan, records))
        self.assertEqual([e['index'] for e in rows], list(range(163)))
        self.assertEqual(sum(e['axis_kind'] == 'overall' for e in rows), 4)
        self.assertFalse(
            any(e['prompt_id'] == self.row['id'] and e['axis_slot'] == 3 for e in rows)
        )

    def test_no_variation_omits_conditioned_rows_without_changing_overall_selection(self):
        self.change_answer(
            lambda r: r.update(status='no_variation', axes=[], reason='No useful axis')
        )
        records, report = audit(self.a, self.plan, self.run)
        self.assertTrue(report['complete'])
        self.assertEqual(report['training_rows'], 160)
        expected = [e['prompt_id'] for e in schedule(self.plan) if e['axis_kind'] == 'overall']
        actual = [
            e['prompt_id']
            for e in available_schedule(self.plan, records)
            if e['axis_kind'] == 'overall'
        ]
        self.assertEqual(actual, expected)

    def test_compact_result_envelope_validates_against_raw_reply(self):
        result = json.loads((self.folder / 'result.json').read_text())
        atomic_json(self.folder / 'result.json', {k: result[k] for k in ('row', 'response')})
        record, issue = read_record(self.folder, self.row, self.a)
        self.assertIsNone(issue)
        self.assertTrue(record['compact_result_envelope'])
        self.assertEqual(record['actual_model'], 'fixture')
        self.assertEqual(record['payload_hash'], 'test-hash')

    def test_review_flag_blocks_concise_reward_but_preserves_natural_request(self):
        self.change_answer(lambda r: r['axes'][0].update(mapping_status='needs_review'))
        self.assertFalse(audit(self.a, self.plan, self.run, 'tpips_factor')[1]['complete'])
        self.assertTrue(audit(self.a, self.plan, self.run)[1]['complete'])

    def test_missing_result_raw_reply_errors_and_refusals(self):
        (self.folder / 'result.json').unlink()
        self.assertEqual(
            read_record(self.folder, self.row, self.a, 'tpips_factor')[1], 'pending_result_commit'
        )
        record, issue = read_record(self.folder, self.row, self.a)
        self.assertIsNone(issue)
        self.assertTrue(record['recovered_from_raw_reply'])
        raw = json.loads((self.folder / 'reply.json').read_text())
        raw['text'] = '{}'
        atomic_json(self.folder / 'reply.json', raw)
        self.assertEqual(read_record(self.folder, self.row, self.a)[1], 'invalid_reply')
        raw['finish_reason'] = 'content_filter'
        atomic_json(self.folder / 'reply.json', raw)
        self.assertEqual(read_record(self.folder, self.row, self.a)[1], 'blocked_reply')
        (self.folder / 'reply.json').unlink()
        self.assertEqual(read_record(self.folder, self.row, self.a)[1], 'pending_reply')
        atomic_json(self.folder / 'error-000001.json', {'error': 'test'})
        self.assertEqual(read_record(self.folder, self.row, self.a)[1], 'api_error')

    def test_verbatim_reward_recovers_raw_reply_without_rewriting_or_weakening_request_checks(self):
        (self.folder / 'result.json').unlink()
        answer = response()
        answer['axes'][0].update(
            request='Chair color and material — including wood.',
            tpips_factor='chair color and material',
            mapping_status='needs_review',
        )
        raw = json.loads((self.folder / 'reply.json').read_text())
        raw['text'] = json.dumps(answer)
        atomic_json(self.folder / 'reply.json', raw)
        before = {p.name: p.read_bytes() for p in self.folder.glob('*.json')}
        record, issue = read_record(self.folder, self.row, self.a)
        self.assertIsNone(issue)
        self.assertEqual(record['response'], answer)
        self.assertNotIn('result.json', record['files'])
        self.assertTrue(record['recovered_from_raw_reply'])
        self.assertEqual(
            read_record(self.folder, self.row, self.a, 'tpips_factor')[1], 'invalid_reply'
        )
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.glob('*.json')})
        answer['axes'][1]['request'] = answer['axes'][0]['request']
        raw['text'] = json.dumps(answer)
        atomic_json(self.folder / 'reply.json', raw)
        self.assertEqual(read_record(self.folder, self.row, self.a)[1], 'invalid_reply')

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'pyarrow optional')
    def test_blocked_prompt_excluded_from_axes_overall_and_provenance_export(self):
        import pyarrow.parquet as pq

        plan = self.prepare_plan()
        pid = next(
            e['prompt_id']
            for e in schedule(self.plan)
            if e['axis_kind'] == 'overall' and e['prompt_id'] != self.a['items'][0]['id']
        )
        folder = self.run / 'items' / pid
        (folder / 'result.json').unlink()
        raw = json.loads((folder / 'reply.json').read_text())
        raw['finish_reason'] = 'content_filter'
        atomic_json(folder / 'reply.json', raw)
        before = {str(p): p.read_bytes() for p in self.run.rglob('*.json')}
        out = self.root / 'blocked-export'
        self.assertEqual(
            main(
                [
                    'build',
                    '--plan',
                    str(plan),
                    '--elicitation-run',
                    str(self.run),
                    '--out',
                    str(out),
                    '--execute',
                ]
            ),
            0,
        )
        rows = pq.read_table(out / 'train.parquet').to_pylist()
        self.assertEqual(len(rows), 159)
        self.assertEqual([r['extra_info']['index'] for r in rows], list(range(159)))
        expected = [e['row_id'] for e in schedule(self.plan) if e['prompt_id'] != pid]
        self.assertEqual([r['extra_info']['row_id'] for r in rows], expected)
        self.assertTrue(all(r['extra_info']['prompt_id'] != pid for r in rows))
        records = [json.loads(s) for s in (out / 'source-records.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 39)
        self.assertTrue(all(r['row']['id'] != pid for r in records))
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(
            (manifest['base_prompts'], manifest['overall_rows'], manifest['excluded_prompts']),
            (39, 3, 1),
        )
        exclusions = json.loads((out / 'exclusions.json').read_text())
        self.assertEqual(exclusions[0]['prompt_id'], pid)
        self.assertIn('reply.json', exclusions[0]['files'])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.run.rglob('*.json')})

    def test_api_block_excluded_but_missing_and_invalid_replies_still_require_attention(self):
        (self.folder / 'result.json').unlink()
        (self.folder / 'reply.json').unlink()
        error = self.folder / 'error-000001.json'
        atomic_json(error, {'error': 'ProviderError: blocked'})
        records, report = audit(self.a, self.plan, self.run)
        self.assertTrue(report['complete'])
        self.assertEqual(report['excluded_prompts'], 1)
        self.assertEqual(report['exclusions'][0]['reason'], 'api_error')
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            list(available_schedule(self.plan, records))
        error.unlink()
        self.assertFalse(audit(self.a, self.plan, self.run)[1]['complete'])
        atomic_json(
            self.folder / 'reply.json',
            dict(text='{}', finish_reason='stop', refusal=None, model='fixture'),
        )
        report = audit(self.a, self.plan, self.run)[1]
        self.assertFalse(report['complete'])
        self.assertEqual(report['excluded_prompts'], 0)
        self.assertEqual(report['issues'][0]['reason'], 'invalid_reply')

    def test_raw_response_and_image_identity_tampering_detected(self):
        result = json.loads((self.folder / 'result.json').read_text())
        result['response']['axes'][0]['request'] = 'Not the original request'
        atomic_json(self.folder / 'result.json', result)
        with self.assertRaises(ValueError):
            read_record(self.folder, self.row, self.a)
        save_item(self.folder, self.row, self.a)
        request = json.loads((self.folder / 'request.json').read_text())
        request['images'][0]['receipt']['prompt_id'] = 'wrong'
        atomic_json(self.folder / 'request.json', request)
        with self.assertRaises(ValueError):
            read_record(self.folder, self.row, self.a)

    def test_run_must_cover_every_assigned_prompt(self):
        manifest = json.loads((self.run / 'manifest.json').read_text())
        manifest['ids'] = manifest['ids'][1:]
        atomic_json(self.run / 'manifest.json', manifest)
        with self.assertRaises(ValueError):
            audit(self.a, self.plan, self.run)

    def prepare_plan(self):
        out = self.root / 'plan'
        args = [
            'plan',
            '--assignment',
            str(self.run / 'assignment.json'),
            '--out',
            str(out),
            '--expected-prompts',
            '40',
        ]
        self.assertEqual(main(args), 0)
        self.assertFalse(out.exists())
        self.assertEqual(main(args + ['--execute']), 0)
        self.assertEqual(load_plan(out), (self.a, self.plan))
        return out

    def test_incomplete_build_refused_and_no_training_file_written(self):
        plan = self.prepare_plan()
        (self.folder / 'result.json').unlink()
        (self.folder / 'reply.json').unlink()
        out = self.root / 'export'
        code = main(
            [
                'build',
                '--plan',
                str(plan),
                '--elicitation-run',
                str(self.run),
                '--out',
                str(out),
                '--execute',
            ]
        )
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'pyarrow optional')
    def test_parquet_and_jsonl_roundtrip_preserve_order_and_loader_guard(self):
        import pyarrow.parquet as pq

        plan = self.prepare_plan()
        out = self.root / 'export'
        args = [
            'build',
            '--plan',
            str(plan),
            '--elicitation-run',
            str(self.run),
            '--out',
            str(out),
            '--execute',
        ]
        self.assertEqual(main(args), 0)
        json_rows = [json.loads(s) for s in (out / 'train.jsonl').read_text().splitlines()]
        self.assertEqual(pq.read_table(out / 'train.parquet').to_pylist(), json_rows)
        self.assertEqual(
            [r['extra_info']['row_id'] for r in json_rows],
            [e['row_id'] for e in schedule(self.plan)],
        )
        self.assertEqual(len(json_rows), 164)
        for row in json_rows:
            spec = json.loads(row['reward_model']['ground_truth'])
            self.assertEqual(spec['axis'], spec['tpips_factor'])
            self.assertEqual(row['extra_info']['axis'], spec['axis'])
        settings = json.loads((out / 'loader-settings.json').read_text())
        settings['data']['train_batch_size'] = 4
        # Real settings live outside the immutable export.
        config = self.root / 'config.json'
        atomic_json(config, settings)
        self.assertEqual(main(['check-loader', '--dataset', str(out), '--config', str(config)]), 0)
        with self.assertRaises(ValueError):
            main(args)

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'pyarrow optional')
    def test_short_candidate_export_records_actual_row_count(self):
        import pyarrow.parquet as pq

        plan = self.prepare_plan()
        self.change_answer(lambda r: r['axes'].pop())
        out = self.root / 'short-export'
        self.assertEqual(
            main(
                [
                    'build',
                    '--plan',
                    str(plan),
                    '--elicitation-run',
                    str(self.run),
                    '--out',
                    str(out),
                    '--execute',
                ]
            ),
            0,
        )
        rows = pq.read_table(out / 'train.parquet').to_pylist()
        self.assertEqual(len(rows), 163)
        self.assertEqual([r['extra_info']['index'] for r in rows], list(range(163)))
        self.assertEqual(sum(r['extra_info']['axis_kind'] == 'overall' for r in rows), 4)
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['rows'], 163)
        ids = [r['extra_info']['row_id'] for r in rows]
        self.assertEqual(len(set(ids)), 163)
        self.assertNotIn(f"{self.row['id']}:axis-4", ids)

    def test_atomic_publication_and_input_protection(self):
        with self.assertRaises(ValueError):
            with new_directory(self.run / 'nested', (self.run,)):
                pass
        with self.assertRaises(RuntimeError):
            with new_directory(self.root / 'partial') as temp:
                (temp / 'data').write_text('partial')
                raise RuntimeError('simulated interruption')
        self.assertFalse((self.root / 'partial').exists())
        with self.assertRaises(ValueError):
            with new_directory(self.run):
                pass


if __name__ == '__main__':
    unittest.main()
