"""Frozen four-pass schedule and read-only import of saved elicitation results.

No API calls, image generation, axis paraphrasing, padding, or training occur here.
An incomplete source can be audited, but cannot become a training export.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from elicitation.policy import parse
from elicitation.selection import T2I_MODELS, load_assignment
from hpdv2.storage import canonical, digest, prompt_id, read_json
from .policy import VERSION, instruction


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_plan(assignment, seed=42, expected_prompts=20000, k=4):
    """Build the frozen four-pass plan from balanced 40-prompt blocks.

    Each block draws one overall prompt per T2I model; each of the four passes places one of
    those draws after the block, rotating T2I models so overall rows are balanced per pass and
    per epoch. Candidate slots keep the elicitor's order. `k` is the number of generated
    expansions, independent of the four candidate slots per base prompt.
    """
    rows = assignment["items"]
    if type(seed) is not int or type(k) is not int or k < 1:
        raise ValueError("Require integer seed and positive k")
    if (
        type(expected_prompts) is not int
        or expected_prompts < 40
        or expected_prompts % 40
        or len(rows) != expected_prompts
    ):
        raise ValueError("Require the expected prompt count, a positive multiple of 40")
    if assignment.get("k") != 4:
        raise ValueError("Expected four source images per prompt")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prompt IDs")
    if len({' '.join(r['prompt'].casefold().split()) for r in rows}) != len(rows):
        raise ValueError("Repeated normalized base prompt")
    if any(r["id"] != prompt_id(r["prompt"]) for r in rows):
        raise ValueError("Prompt ID/text mismatch")
    if Counter(r["renderer"] for r in rows) != Counter({r: len(rows) // 4 for r in T2I_MODELS}):
        raise ValueError("Require equal counts from the four assigned T2I models")
    groups = {}
    for t2i_model in T2I_MODELS:
        groups[t2i_model] = sorted((r["id"] for r in rows if r["renderer"] == t2i_model))
        random.Random(f"generation-order:{seed}:{t2i_model}").shuffle(groups[t2i_model])
    order = [groups[r][i] for i in range(len(rows) // 4) for r in T2I_MODELS]
    by_id = {r['id']: r for r in rows}
    overall = []
    for block in range(len(rows) // 40):
        ids = order[block * 40 : (block + 1) * 40]
        choices = {
            r: random.Random(f"generation-overall:{seed}:{block}:{r}").choice(
                [pid for pid in ids if by_id[pid]['renderer'] == r]
            )
            for r in T2I_MODELS
        }
        overall.append([choices[T2I_MODELS[(block + phase) % 4]] for phase in range(4)])
    return dict(
        schema_version=1,
        kind="generation_policy_four_passes",
        algorithm="balanced-block40-interleaved-overall-v1",
        seed=seed,
        assignment_sha256=digest(assignment),
        base_prompts=len(rows),
        candidate_axes_per_prompt=4,
        generations_per_example=k,
        conditioned_rows=4 * len(rows),
        overall_rows=len(rows) // 10,
        total_rows=4 * len(rows) + len(rows) // 10,
        prompt_order=order,
        overall_by_block_and_pass=overall,
    )


def schedule(plan):
    index = 0
    for phase in range(4):
        for block, overall_ids in enumerate(plan["overall_by_block_and_pass"]):
            ids = plan["prompt_order"][40 * block : 40 * (block + 1)]
            for pid, slot in [(pid, phase) for pid in ids] + [(overall_ids[phase], None)]:
                yield dict(
                    index=index,
                    pass_number=phase + 1,
                    prompt_id=pid,
                    axis_slot=slot,
                    axis_kind="overall" if slot is None else "axis",
                    row_id=f"{pid}:overall" if slot is None else f"{pid}:axis-{slot+1}",
                )
                index += 1


def available_schedule(plan, records, excluded_ids=()):
    """Yield the schedule without unavailable slots and without rows of excluded prompts.

    Missing slots are not filled by retries or duplicates. `index` is renumbered
    contiguously; `planned_index` keeps the position in the frozen slot plan.
    """
    excluded_ids = set(excluded_ids)
    planned_ids = set(plan['prompt_order'])
    if not excluded_ids <= planned_ids:
        raise ValueError('Exclusion contains an unknown prompt')
    if planned_ids - excluded_ids - set(records):
        raise ValueError('Missing unresolved records must not be silently excluded')
    index = 0
    for entry in schedule(plan):
        if entry['prompt_id'] in excluded_ids:
            continue
        slot = entry['axis_slot']
        if slot is not None and slot >= len(records[entry['prompt_id']]['response']['axes']):
            continue
        yield dict(entry, planned_index=entry['index'], index=index)
        index += 1


def load_plan(directory):
    directory = Path(directory)
    assignment = load_assignment(directory / "assignment.json")
    plan = read_json(directory / "plan.json")
    expected = make_plan(
        assignment, plan['seed'], plan['base_prompts'], plan['generations_per_example']
    )
    if plan != expected or read_json(directory / "plan-hash.json") != {"sha256": digest(plan)}:
        raise ValueError("Frozen plan changed")
    return assignment, plan


def locate_sources(run, assignment):
    """Verify the run identity and return the item directory of every assigned prompt."""
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest['assignment_hash'] != digest(assignment) or manifest['config']['max_axes'] != 4:
        raise ValueError("Wrong elicitation input or candidate limit")
    ids = [r['id'] for r in assignment['items']]
    if sorted(manifest['ids']) != sorted(ids):
        raise ValueError("Elicitation run does not cover every assigned prompt exactly once")
    return {pid: run / "items" / pid for pid in ids}, manifest


def read_record(folder, row, assignment, reward_text='request'):
    """Verify a committed result, or recover one from the saved raw reply for verbatim reward text.

    Returns `(record, None)`, or `(None, reason)` for a prompt that has no usable result.
    Image receipts are checked; image bytes are not reread, since the text-only policy does
    not consume the images.
    """
    folder = Path(folder)
    result_path = folder / "result.json"
    if reward_text not in {'request', 'tpips_factor'}:
        raise ValueError('Unknown reward text choice')
    recovered = not result_path.exists()
    if not result_path.exists():
        if (folder / "reply.json").exists():
            reply = read_json(folder / "reply.json")
            if reply.get('refusal') or reply.get('finish_reason') != 'stop':
                return None, "blocked_reply"
            try:
                parsed = parse(
                    reply['text'], 4, validate_short_factor=reward_text == 'tpips_factor'
                )
            except (ValueError, TypeError, KeyError):
                return None, "invalid_reply"
            if reward_text != 'request':
                return None, "pending_result_commit"
            if not (folder / 'request.json').is_file():
                return None, "missing_request"
            request = read_json(folder / 'request.json')
            # In-memory envelope only; no source file is written. The request and images are
            # verified below as for committed data.
            result = dict(row=request['row'], response=parsed, payload_hash=request['payload_hash'])
        elif any(folder.glob('error-*.json')):
            return None, "api_error"
        else:
            return None, "pending_reply"
    else:
        result, request, reply = [
            read_json(folder / name) for name in ('result.json', 'request.json', 'reply.json')
        ]
    # Results of the 32-call concurrency probe use a compact envelope without payload_hash
    # or actual_model; their request and raw reply carry the identities checked here.
    if (
        result['row'] != row
        or request['row'] != row
        or ('payload_hash' in result and result['payload_hash'] != request['payload_hash'])
    ):
        raise ValueError(f"Source identity mismatch for {row['id']}")
    if reply.get('refusal') or reply['finish_reason'] != 'stop':
        raise ValueError("Committed result has a blocked raw reply")
    response = parse(reply['text'], 4, validate_short_factor=reward_text == 'tpips_factor')
    if response != result['response'] or (
        'actual_model' in result and reply['model'] != result['actual_model']
    ):
        raise ValueError("Result differs from saved raw reply")
    manifest = assignment['manifests'][row['renderer']][str(row['shard'])]
    cfg = manifest['config']
    from baselines.assets import seed_for

    images = request['images']
    if len(images) != 4:
        raise ValueError("Expected four image audit records")
    for i, image in enumerate(images):
        receipt = image['receipt']
        wanted = dict(
            prompt_id=row['id'],
            render_prompt=row['prompt'],
            index=i,
            seed=seed_for(row['id'], cfg['seed'], i),
            run_hash=digest(manifest),
        )
        if any(receipt.get(k) != v for k, v in wanted.items()):
            raise ValueError("Image audit receipt identity mismatch")
        expected_path = (
            Path(assignment['render_root'])
            / row['renderer']
            / f"shard-{row['shard']:02d}"
            / "images"
            / row['id']
            / f"{i:04d}"
            / "image.png"
        )
        if Path(image['path']) != expected_path or image['sha256'] != receipt['sha256']:
            raise ValueError("Image audit path or checksum mismatch")
    return (
        dict(
            row=row,
            response=response,
            semantic_review=result.get('semantic_review', 'pending'),
            source_directory=str(folder.resolve()),
            actual_model=reply['model'],
            payload_hash=request['payload_hash'],
            images=images,
            compact_result_envelope='payload_hash' not in result,
            recovered_from_raw_reply=recovered,
            files={
                name: sha256(folder / name)
                for name in ('request.json', 'reply.json', 'result.json', 'validation.json')
                if (folder / name).is_file()
            },
        ),
        None,
    )


def audit(assignment, plan, run, reward_text="request"):
    if reward_text not in {"tpips_factor", "request"}:
        raise ValueError("Unknown reward text choice")
    sources, source_manifest = locate_sources(run, assignment)
    records, issues, exclusions, status_counts, slot_counts = {}, [], [], Counter(), Counter()
    for row in assignment['items']:
        record, issue = read_record(sources[row['id']], row, assignment, reward_text)
        if record is not None:
            records[row['id']] = record
            response = record['response']
            slot_counts[str(len(response['axes']))] += 1
            if response['status'] == 'unassessable':
                issue = response['status']
            elif any(
                a['request'].strip().casefold() == 'overall'
                or (
                    reward_text == 'tpips_factor'
                    and a['tpips_factor'].strip().casefold() == 'overall'
                )
                for a in response['axes']
            ):
                issue = 'reserved_overall_axis'
            elif reward_text == 'tpips_factor' and any(
                a['mapping_status'] != 'faithful' for a in response['axes']
            ):
                issue = 'reward_mapping_needs_review'
        status_counts[issue or 'ready'] += 1
        if issue:
            detail = dict(
                prompt_id=row['id'],
                renderer=row['renderer'],
                reason=issue,
                axes_available=len(record['response']['axes']) if record else None,
                source_directory=str(sources[row['id']]),
            )
            if issue in {'blocked_reply', 'api_error'}:
                # Blocked replies and API errors exclude every row of the prompt, including
                # its overall row. Source files are not retried or rewritten.
                folder = sources[row['id']]
                detail['files'] = {p.name: sha256(p) for p in sorted(folder.glob('*.json'))}
                exclusions.append(detail)
            else:
                issues.append(detail)
    excluded_ids = {e['prompt_id'] for e in exclusions}
    overall_rows = sum(
        e['axis_kind'] == 'overall' and e['prompt_id'] not in excluded_ids for e in schedule(plan)
    )
    known_axes = sum(len(r['response']['axes']) for r in records.values())
    report = dict(
        schema_version=3,
        complete=not issues,
        maximum_rows=plan['total_rows'],
        training_rows=known_axes + overall_rows if not issues else None,
        known_candidate_rows=known_axes,
        overall_rows=overall_rows,
        recovered_raw_replies=sum(r['recovered_from_raw_reply'] for r in records.values()),
        planned_overall_rows=plan['overall_rows'],
        excluded_prompts=len(exclusions),
        excluded_overall_rows=plan['overall_rows'] - overall_rows,
        blocked_prompt_policy='exclude all rows for blocked replies and API errors; no replacements or retries',
        missing_axis_policy='omit unavailable slots; no padding or follow-up calls',
        base_prompts=plan['base_prompts'] - len(exclusions),
        planned_base_prompts=plan['base_prompts'],
        by_status=dict(status_counts),
        exclusions=exclusions,
        committed_results_by_axis_count=dict(slot_counts),
        source_run=str(Path(run).resolve()),
        source_manifest_sha256=digest(source_manifest),
        plan_sha256=digest(plan),
        reward_text=reward_text,
        semantic_review='not certified by structural dataset validation',
        image_check='saved audit receipts only; no image bytes reread',
        issues=issues,
    )
    return records, report


def training_row(entry, row, plan, record, reward_text="request"):
    overall = entry['axis_slot'] is None
    candidate = None if overall else record['response']['axes'][entry['axis_slot']]
    axis = 'overall' if overall else candidate['request']
    factor = 'overall' if overall else candidate['tpips_factor']
    reward_condition = axis if reward_text == 'request' else factor
    k = plan['generations_per_example']
    spec = dict(base_prompt=row['prompt'], axis=axis, tpips_factor=reward_condition, k=k)
    return dict(
        data_source='axis_diversity',
        ability='axis_diversity',
        prompt=[dict(role='user', content=instruction(row['prompt'], axis, k))],
        reward_model=dict(style='model', ground_truth=json.dumps(spec, ensure_ascii=False)),
        extra_info=dict(
            index=entry['index'],
            row_id=entry['row_id'],
            split='train',
            planned_index=entry.get('planned_index', entry['index']),
            prompt_id=row['id'],
            base_prompt=row['prompt'],
            cluster_id=row.get('cluster_id'),
            axis=axis,
            axis_kind=entry['axis_kind'],
            axis_slot=entry['axis_slot'],
            pass_number=entry['pass_number'],
            k=k,
            tpips_factor=reward_condition,
            concise_factor=factor,
            reward_text=reward_text,
            mapping_status='not_applicable' if overall else candidate['mapping_status'],
            constraints=[] if overall else candidate['constraints'],
            source_renderer=row['renderer'],
            source_record_id=row['id'],
            source_payload_hash=record['payload_hash'] if record else None,
            instruction_version=VERSION,
        ),
    )


def rows_for_export(assignment, plan, records, reward_text, excluded_ids=()):
    by_id = {r['id']: r for r in assignment['items']}
    for entry in available_schedule(plan, records, excluded_ids):
        yield training_row(
            entry, by_id[entry['prompt_id']], plan, records[entry['prompt_id']], reward_text
        )


def validate_loader_config(config, total_rows):
    """Check a resolved config for the stock verl loader, which hard-codes drop_last=True."""
    data, trainer = config['data'], config['trainer']
    if data.get('shuffle') is not False or trainer.get('total_epochs') != 1:
        raise ValueError("Require data.shuffle=false and trainer.total_epochs=1")
    if data.get('filter_overlong_prompts') is not False or data.get('train_max_samples') != -1:
        raise ValueError("No filtering or sample limit may alter the frozen schedule")
    if data.get('truncation') != 'error':
        raise ValueError("Require truncation=error; do not truncate the training task")
    if trainer.get('total_training_steps') is not None:
        raise ValueError("Do not override the one-epoch step count")
    sampler = data.get('sampler') or {}
    if sampler.get('class_path') or sampler.get('class_name'):
        raise ValueError("Custom samplers need a separate order audit")
    batch = data.get('gen_batch_size', data.get('train_batch_size'))
    if type(batch) is not int or batch < 1 or total_rows % batch:
        raise ValueError(
            "Loader batch size must divide the row count: verl drops the final partial batch"
        )
    return dict(rows=total_rows, loader_batch_size=batch, steps=total_rows // batch, epochs=1)


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')


def export_parquet(path, rows):
    """Write rows to Parquet in 1,000-row chunks, with the schema inferred from the first chunk."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from itertools import islice

    iterator = iter(rows)
    first = list(islice(iterator, 1000))
    if not first:
        raise ValueError("Empty export")
    # `constraints` is set to list<string> explicitly: overall rows carry an empty list,
    # from which Arrow cannot infer the element type.
    table = pa.Table.from_pylist(first)
    extra = table.schema.field('extra_info').type
    fields = [
        pa.field(f.name, pa.list_(pa.string()) if f.name == 'constraints' else f.type)
        for f in extra
    ]
    schema = table.schema.set(
        table.schema.get_field_index('extra_info'), pa.field('extra_info', pa.struct(fields))
    )
    with pq.ParquetWriter(path, schema, compression='zstd') as writer:
        writer.write_table(pa.Table.from_pylist(first, schema=schema))
        while chunk := list(islice(iterator, 1000)):
            writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
