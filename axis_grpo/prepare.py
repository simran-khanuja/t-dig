"""CLI: python -m axis_grpo.prepare {plan,audit,build,check-loader}."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from elicitation.selection import load_assignment
from hpdv2.storage import atomic_json, digest, read_json, sync_directory
from .dataset import (
    audit,
    export_parquet,
    load_plan,
    make_plan,
    rows_for_export,
    schedule,
    sha256,
    training_row,
    validate_loader_config,
    write_jsonl,
)
from .policy import VERSION


@contextmanager
def new_directory(out, forbidden=()):
    """Yield a temporary directory and rename it to `out` once the block completes.

    Raises ValueError if `out` exists or overlaps a directory in `forbidden`. Files are
    fsynced before the rename; on failure the temporary directory is removed.
    """
    out = Path(out).resolve()
    for source in forbidden:
        source = Path(source).resolve()
        if out == source or source in out.parents or out in source.parents:
            raise ValueError("Output directory overlaps a read-only source")
    if out.exists():
        raise ValueError(f"Output exists; use a new directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # The lock file excludes concurrent builders of the same output name.
    lock = out.parent / f'.{out.name}.prepare.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temp = Path(tempfile.mkdtemp(prefix=f'.{out.name}.', dir=out.parent))
    try:
        yield temp
        for p in temp.rglob('*'):
            if p.is_file():
                with p.open('rb') as stream:
                    os.fsync(stream.fileno())
        if out.exists():
            raise ValueError("Output appeared during build")
        os.rename(temp, out)
        sync_directory(out.parent)
    finally:
        os.close(fd)
        lock.unlink()
        if temp.exists():
            shutil.rmtree(temp)


def concise(report):
    return {k: v for k, v in report.items() if k not in {'issues', 'exclusions'}}


def preview(assignment, plan, records, report):
    """Return inspection rows: the first 12 available, plus one axis and one overall row per pass.

    The preview is for reading only and is not a training dataset.
    """
    bad = {r['prompt_id'] for r in report['issues'] + report.get('exclusions', [])}
    by_id = {r['id']: r for r in assignment['items']}
    available = [
        e
        for e in schedule(plan)
        if e['prompt_id'] not in bad
        and (
            e['axis_slot'] is None
            or e['axis_slot'] < len(records[e['prompt_id']]['response']['axes'])
        )
    ]
    selected = available[:12]
    for phase in range(1, 5):
        for kind in ('axis', 'overall'):
            e = next(
                (e for e in available if e['pass_number'] == phase and e['axis_kind'] == kind), None
            )
            if e is not None and e not in selected:
                selected.append(e)
    selected.sort(key=lambda e: e['index'])
    return [
        training_row(e, by_id[e['prompt_id']], plan, records[e['prompt_id']], report['reward_text'])
        for e in selected
    ]


def preview_text(rows):
    lines = [
        'INSPECTION PREVIEW ONLY — not a complete training dataset.',
        'Indices below are zero-based positions in the maximum-size slot plan; gaps are intentional.',
        '',
    ]
    for r in rows:
        e = r['extra_info']
        lines += [
            f"Planned row {e['planned_index']} | pass {e['pass_number']} | {e['source_renderer']} | {e['axis_kind']}",
            f"Base prompt: {e['base_prompt']}",
            f"Axis: {e['axis']}",
            f"Reward text: {e['tpips_factor']}",
            '',
        ]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('plan', help='Freeze prompt order and 2,000 overall selections')
    q.add_argument('--assignment', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--seed', type=int, default=42)
    q.add_argument('--expected-prompts', type=int, default=20000)
    q.add_argument('--k', type=int, default=4, help='Expanded prompts per training example')
    q.add_argument('--execute', action='store_true')
    for name in ('audit', 'build'):
        q = sub.add_parser(name)
        q.add_argument('--plan', type=Path, required=True)
        q.add_argument('--elicitation-run', type=Path, required=True)
        q.add_argument('--out', type=Path, required=True)
        q.add_argument(
            '--reward-text',
            choices=('tpips_factor', 'request'),
            default='request',
            help='Default: pass the identical original axis request to both policy and TPIPS',
        )
        q.add_argument('--execute', action='store_true')
    q = sub.add_parser(
        'check-loader', help='Validate a resolved JSON trainer config against the export'
    )
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--config', type=Path, required=True)
    a = p.parse_args(argv)
    if getattr(a, 'execute', False) and a.out.exists():
        raise ValueError(f"Output exists; use a new directory: {a.out}")
    if a.command == 'check-loader':
        manifest = read_json(a.dataset / 'manifest.json')
        if not manifest['complete']:
            raise ValueError('Dataset is incomplete')
        for name, expected in manifest['files'].items():
            if sha256(a.dataset / name) != expected:
                raise ValueError(f'Dataset file changed: {name}')
        config = read_json(a.config)
        files = config['data']['train_files']
        if isinstance(files, str):
            files = [files]
        if [Path(f).resolve() for f in files] != [(a.dataset / 'train.parquet').resolve()]:
            raise ValueError('Require only this frozen train.parquet in train_files')
        print(json.dumps(validate_loader_config(config, manifest['rows']), indent=2))
        return 0
    if a.command == 'plan':
        assignment = load_assignment(a.assignment)
        plan = make_plan(assignment, a.seed, a.expected_prompts, a.k)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in plan.items()
                    if k not in ('prompt_order', 'overall_by_block_and_pass')
                },
                indent=2,
            )
        )
        if a.execute:
            with new_directory(a.out, (a.assignment.parent,)) as out:
                atomic_json(out / 'assignment.json', assignment)
                atomic_json(out / 'assignment-hash.json', {'sha256': digest(assignment)})
                atomic_json(out / 'plan.json', plan)
                atomic_json(out / 'plan-hash.json', {'sha256': digest(plan)})
                write_jsonl(out / 'schedule.jsonl', schedule(plan))
                (out / 'README.md').write_text(
                    '# Frozen generation-policy schedule\n\n'
                    f"{plan['base_prompts']:,} unique base prompts; four ordered axis passes; "
                    f"{plan['overall_rows']:,} distinct overall prompts; at most {plan['total_rows']:,} rows.\n\n"
                    'This is an ordering plan, not a complete training dataset. Each block of 40 '
                    'planned axis slots is followed by one overall row. Missing candidate slots '
                    'are omitted at export without new API calls. Candidate slots are zero-based. '
                    'See ../../axis_grpo/README.md for audit/build commands.\n',
                    encoding='utf-8',
                )
        return 0
    assignment, plan = load_plan(a.plan)
    records, report = audit(assignment, plan, a.elicitation_run, a.reward_text)
    report['created_utc'] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(concise(report), indent=2))
    if not a.execute:
        return 0 if a.command == 'audit' or report['complete'] else 2
    if a.command == 'build' and not report['complete']:
        print(
            'No dataset exported: resolve the audit issues first. Use audit --execute to save the issue list.'
        )
        return 2
    # The output may sit beside the plan, but not inside it or inside any elicitation input.
    with new_directory(a.out, (a.plan, a.elicitation_run, assignment['render_root'])) as out:
        atomic_json(out / 'readiness.json', report)
        atomic_json(out / 'exclusions.json', report['exclusions'])
        sample = preview(assignment, plan, records, report)
        write_jsonl(out / 'preview.jsonl', sample)
        (out / 'preview.txt').write_text(preview_text(sample), encoding='utf-8')
        if a.command == 'build':
            excluded_ids = {e['prompt_id'] for e in report['exclusions']}
            rows = lambda: rows_for_export(assignment, plan, records, a.reward_text, excluded_ids)
            export_parquet(out / 'train.parquet', rows())
            write_jsonl(out / 'train.jsonl', rows())
            write_jsonl(
                out / 'source-records.jsonl',
                (records[pid] for pid in plan['prompt_order'] if pid not in excluded_ids),
            )
            atomic_json(out / 'assignment.json', assignment)
            atomic_json(out / 'assignment-hash.json', {'sha256': digest(assignment)})
            atomic_json(out / 'plan.json', plan)
            atomic_json(out / 'plan-hash.json', {'sha256': digest(plan)})
            atomic_json(
                out / 'loader-settings.json',
                dict(
                    data=dict(
                        train_files=str(a.out.resolve() / 'train.parquet'),
                        train_batch_size=None,
                        shuffle=False,
                        filter_overlong_prompts=False,
                        train_max_samples=-1,
                        truncation='error',
                    ),
                    trainer=dict(total_epochs=1, total_training_steps=None),
                ),
            )
            source = out / 'builder-source'
            source.mkdir()
            for path in Path(__file__).parent.glob('*.py'):
                shutil.copyfile(path, source / path.name)
            from elicitation import policy as elicitation_policy

            shutil.copyfile(elicitation_policy.__file__, source / 'elicitation-policy.py')
            files = {
                str(path.relative_to(out)): sha256(path)
                for path in out.rglob('*')
                if path.is_file()
            }
            atomic_json(
                out / 'manifest.json',
                dict(
                    schema_version=2,
                    complete=True,
                    rows=report['training_rows'],
                    maximum_rows=plan['total_rows'],
                    missing_axis_policy=report['missing_axis_policy'],
                    base_prompts=report['base_prompts'],
                    planned_base_prompts=plan['base_prompts'],
                    excluded_prompts=report['excluded_prompts'],
                    blocked_prompt_policy=report['blocked_prompt_policy'],
                    overall_rows=report['overall_rows'],
                    planned_overall_rows=plan['overall_rows'],
                    plan_sha256=digest(plan),
                    instruction_version=VERSION,
                    reward_text=a.reward_text,
                    files=files,
                    semantic_quality='not established by this structural export',
                    training_launched=False,
                ),
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
