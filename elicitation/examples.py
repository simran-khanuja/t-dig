"""Reviewed example banks and in-context example sampling."""

import hashlib
from pathlib import Path

from hpdv2.storage import digest, read_events, read_json


def load_bank(path):
    bank = read_json(path)
    if bank.get('kind') == 'curated_merged_axis_examples':
        from .curated_examples import validate_bank

        return validate_bank(bank)
    if bank.get('kind') != 'reviewed_axis_examples' or not bank.get('reviewer'):
        raise ValueError('Use a reviewed example bank, not the raw inventory')
    examples = bank.get('examples', [])
    if not examples or len({r['id'] for r in examples}) != len(examples):
        raise ValueError('Example bank is empty or has duplicate IDs')
    for row in examples:
        if (
            row.get('approved') is not True
            or not row.get('review_note')
            or not row.get('prompt')
            or not row.get('source')
            or row.get('provenance') not in {'human_free_text', 'interactive_reviewed'}
        ):
            raise ValueError('Every example needs explicit review and provenance')
        original = read_events(row['source'])[row['line'] - 1]
        if hashlib.sha256(Path(row['source']).read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('Human example source changed')
        raw_requests = original.get('other_axes', [original.get('axis')])
        requests = row['requests']
        if (
            not isinstance(requests, list)
            or any(not isinstance(a, str) or not a.strip() for a in requests)
            or any(a not in raw_requests for a in requests)
        ):
            raise ValueError('Example must retain verbatim human requests')
        if original.get('prompt') is not None and original['prompt'] != row['prompt']:
            raise ValueError('Example base prompt was changed')
        if not requests and not row.get('none_reason'):
            raise ValueError('No-variation example needs its human explanation')
        if row.get('images') and not row.get('image_reviewed'):
            raise ValueError('Do not guess which images a human saw')
    return bank


def pick_examples(bank, target, count, seed):
    """Return `count` examples (or all) whose normalized prompt differs from the target's.

    A curated merged bank is sampled uniformly. A reviewed bank is sampled uniformly by source
    group, then participant, without repeating a source prompt.
    """
    import random

    if count != 'all' and (type(count) is not int or count < 1):
        raise ValueError('At least one reviewed example required')
    target_text = ' '.join(target['prompt'].casefold().split())
    pool = [r for r in bank['examples'] if ' '.join(r['prompt'].casefold().split()) != target_text]
    if count == 'all':
        if not pool:
            raise ValueError('No examples remain after target exclusion')
        return pool
    rng = random.Random(digest([seed, target['id'], 'examples-v1']))
    if bank.get('kind') == 'curated_merged_axis_examples':
        if len(pool) < count:
            raise ValueError('Not enough distinct curated example prompts')
        return rng.sample(pool, count)
    chosen = []
    while pool and len(chosen) < count:
        group = rng.choice(sorted({r['source_group'] for r in pool}))
        people = sorted({r['participant'] for r in pool if r['source_group'] == group})
        who = rng.choice(people)
        row = rng.choice(
            sorted(
                [r for r in pool if r['source_group'] == group and r['participant'] == who],
                key=lambda r: r['id'],
            )
        )
        chosen.append(row)
        # No source prompt appears twice in one request, even from different raters.
        text = ' '.join(row['prompt'].casefold().split())
        pool = [r for r in pool if ' '.join(r['prompt'].casefold().split()) != text]
    if len(chosen) != count:
        raise ValueError('Not enough distinct reviewed example prompts after leakage exclusion')
    return chosen
