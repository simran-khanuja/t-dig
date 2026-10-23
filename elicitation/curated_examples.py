"""Build a merged prompt/axis bank from explicit, source-linked editorial choices."""

from collections import defaultdict
import hashlib
import random
from pathlib import Path
from hpdv2.storage import digest, prompt_id, read_events


def build_bank(inventory, plan):
    if isinstance(plan, dict) and plan.get('version') == 2:
        return build_verbatim_bank(inventory, plan)
    groups = defaultdict(list)
    for row in inventory['examples']:
        for axis in row['requests']:
            groups[row['prompt']].append(dict(record_id=row['id'], text=axis))
    prompts = sorted(groups, key=str.casefold)
    if [r[0] for r in plan] != list(range(1, len(prompts) + 1)):
        raise ValueError('Plan must cover each source prompt exactly once')
    examples, audit = [], []
    for number, choices in plan:
        prompt = prompts[number - 1]
        sources = groups[prompt]
        requests = []
        mappings = []
        used = set()
        for wording, indices in choices:
            if not wording.strip() or not indices or len(set(indices)) != len(indices):
                raise ValueError('Each curated axis needs wording and distinct source references')
            refs = []
            for i in indices:
                if type(i) is not int or not 0 <= i < len(sources):
                    raise ValueError('Invalid source-axis reference')
                refs.append(sources[i])
                used.add(i)
            requests.append(wording)
            mappings.append(dict(axis=wording, sources=refs))
        if len(set(w.casefold() for w in requests)) != len(requests):
            raise ValueError('Duplicate curated axis')
        # An explicit empty choice list excludes the prompt but retains its omission audit.
        if requests:
            examples.append(
                dict(
                    id=prompt_id(prompt),
                    prompt=prompt,
                    requests=requests,
                    images=[],
                    provenance='human_derived_edited',
                    axis_sources=mappings,
                )
            )
        audit.append(
            dict(prompt=prompt, omitted=[sources[i] for i in range(len(sources)) if i not in used])
        )
    return dict(
        kind='curated_merged_axis_examples',
        version=1,
        reviewer='assistant; consolidation requested by user',
        source_inventory=inventory,
        source_inventory_hash=digest(inventory),
        curation_plan=plan,
        examples=examples,
        omission_audit=audit,
        wording_note='Merged and edited human-derived examples, not verbatim individual responses. Study v1 requests followed model suggestions. Images are not supplied.',
    )


def build_verbatim_bank(inventory, plan):
    """Select one exact source string per reviewed similarity group; never rewrite."""
    groups = defaultdict(list)
    for row in inventory['examples']:
        for axis in row['requests']:
            groups[row['prompt']].append(dict(record_id=row['id'], text=axis))
    prompts = sorted(groups, key=str.casefold)
    if [r['prompt'] for r in plan['prompts']] != prompts or type(plan['seed']) is not int:
        raise ValueError('Plan must cover source prompts in order with an integer seed')
    examples, audit = [], []
    for entry in plan['prompts']:
        prompt = entry['prompt']
        source = groups[prompt]
        used = set()
        mappings = []
        requests = []
        for indices in entry['similarity_groups']:
            if (
                not indices
                or len(set(indices)) != len(indices)
                or used.intersection(indices)
                or any(type(i) is not int or not 0 <= i < len(source) for i in indices)
            ):
                raise ValueError('Invalid or overlapping similarity groups')
            used.update(indices)
            refs = [source[i] for i in indices]
            # Uniform over distinct original strings, not their annotation frequency.
            alternatives = sorted({r['text'] for r in refs})
            seed = int(digest([plan['seed'], prompt, alternatives]), 16)
            chosen = random.Random(seed).choice(alternatives)
            requests.append(chosen)
            mappings.append(
                dict(
                    axis=chosen,
                    selected_sources=[r for r in refs if r['text'] == chosen],
                    sources=refs,
                )
            )
        if len({s.casefold().strip() for s in requests}) != len(requests):
            raise ValueError('Repeated requests remain across similarity groups')
        if requests:
            examples.append(
                dict(
                    id=prompt_id(prompt),
                    prompt=prompt,
                    requests=requests,
                    images=[],
                    provenance='human_verbatim_pooled',
                    axis_sources=mappings,
                )
            )
        audit.append(
            dict(prompt=prompt, omitted=[source[i] for i in range(len(source)) if i not in used])
        )
    return dict(
        kind='curated_merged_axis_examples',
        version=2,
        reviewer='assistant similarity grouping; user requested verbatim seeded selection',
        source_inventory=inventory,
        source_inventory_hash=digest(inventory),
        curation_plan=plan,
        examples=examples,
        omission_audit=audit,
        wording_note='Exact original human wording, pooled by prompt. One seeded random representative per similarity group. Study v1 requests followed model suggestions. No example images supplied.',
    )


def validate_bank(bank):
    inventory = bank['source_inventory']
    if digest(inventory) != bank['source_inventory_hash']:
        raise ValueError('Curated source inventory changed')
    if build_bank(inventory, bank['curation_plan']) != bank:
        raise ValueError('Curated bank disagrees with its source-linked plan')
    files = {}
    for r in inventory['examples']:
        path = r['source']
        if path not in files:
            raw = Path(path).read_bytes()
            files[path] = (hashlib.sha256(raw).hexdigest(), read_events(path))
        sha, events = files[path]
        if sha != r['source_sha256']:
            raise ValueError('Human source file changed')
        event = events[r['line'] - 1]
        original = event.get('other_axes', [event.get('axis')])
        if any(a not in original for a in r['requests']):
            raise ValueError('Invented source annotation')
        if event.get('prompt') is not None and event['prompt'] != r['prompt']:
            raise ValueError('Source base prompt changed')
    return bank
