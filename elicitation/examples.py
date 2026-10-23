"""Human-source inventory; importing a response never approves it for prompting."""

from collections import Counter
import hashlib
from pathlib import Path

from hpdv2.storage import digest, read_events, read_json

PILOT_B_ALLOWLIST = {'participant01', 'participant02', 'participant03'}
EXCLUDED_PILOT_SOURCES = {('responses_pilot_qwen', 'participant05')}
PREFERRED_PILOT_B_SOURCES = {'participant01'}
EXCLUDED_PILOT_RESPONSES = {
    ('responses_pilot_b', 'participant03', 'p012', 'A cautionary scene.'),
    ('responses_pilot_b', 'participant01', 'p019', 'a large blue clock is hanging up outside'),
    ('responses_pilot_b', 'participant03', 'p019', 'a large blue clock is hanging up outside'),
    (
        'responses_pilot_b',
        'participant03',
        'p016',
        'A red door underwater surrounded by bubbles and marine life.',
    ),
    (
        'responses_pilot_qwen',
        'participant04',
        'p026',
        'A river going through the middle of a city filled with boats.',
    ),
    (
        'responses_pilot_qwen',
        'participant06',
        'p026',
        'A river going through the middle of a city filled with boats.',
    ),
    (
        'responses_pilot_b',
        'participant02',
        'p026',
        'A river going through the middle of a city filled with boats.',
    ),
    (
        'responses_pilot_qwen',
        'participant06',
        'p032',
        'An empty street is lined with large umbrellas.',
    ),
}
EXCLUDED_PILOT_AXES = {
    (
        'responses_pilot_b',
        'participant03',
        'p025',
        'Two yellow rubber ducks in a bathtub engaged in an epic battle.',
    ): {
        'they all need to be in a bathtub not a wash basin',
        'there should be physical contact between them to emphasise the battle',
    }
}


def inventory(human_study, study_run):
    human_study, study_run = Path(human_study), Path(study_run)
    records, excluded = [], Counter()
    for folder in ('responses_pilot_qwen', 'responses_pilot_b'):
        for path in sorted((human_study / folder).glob('*.jsonl')):
            if folder == 'responses_pilot_b' and path.stem not in PILOT_B_ALLOWLIST:
                continue  # Do not even read unapproved participants' response files.
            for line, row in enumerate(read_events(path), 1):
                if not row.get('prompt_id') or not row.get('prompt'):
                    excluded['pilot_non_response_records'] += 1
                    continue
                if (folder, path.stem) in EXCLUDED_PILOT_SOURCES:
                    excluded['user_excluded_pilot_responses'] += 1
                    continue
                if (folder, path.stem, row['prompt_id'], row['prompt']) in EXCLUDED_PILOT_RESPONSES:
                    excluded['user_excluded_individual_responses'] += 1
                    continue
                requests = row.get('other_axes', [])
                if not isinstance(requests, list) or any(not isinstance(a, str) for a in requests):
                    raise ValueError('Unexpected human request format')
                blocked_axes = EXCLUDED_PILOT_AXES.get(
                    (folder, path.stem, row['prompt_id'], row['prompt']), set()
                )
                excluded['user_excluded_axis_requests'] += sum(a in blocked_axes for a in requests)
                requests = [a for a in requests if a not in blocked_axes]
                records.append(
                    dict(
                        id=digest([str(path.resolve()), line]),
                        source=str(path.resolve()),
                        line=line,
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        source_group=folder,
                        participant=digest([folder, path.stem])[:12],
                        prompt=row['prompt'],
                        original_prompt_id=row['prompt_id'],
                        requests=requests,
                        want_variation=row.get('want_variation'),
                        none_reason=row.get('none_reason', ''),
                        images=[],
                        image_provenance='unverified: identify the exact study image set before pairing',
                        provenance='human_free_text',
                        approved=False,
                        review_note='Not reviewed',
                        user_preferred_source=(
                            folder == 'responses_pilot_b' and path.stem in PREFERRED_PILOT_B_SOURCES
                        ),
                    )
                )
    events_path = study_run / 'events.jsonl'
    events = read_events(events_path)
    source_sha = hashlib.sha256(events_path.read_bytes()).hexdigest()
    sessions, iid, suggested = {}, {}, set()
    for line, row in enumerate(events, 1):
        sid = row.get('session_id')
        if row.get('event') == 'session_new':
            sessions[sid] = row.get('user_id') or (row.get('client') or {}).get('participant_id')
        elif row.get('event') == 'iid':
            iid[sid] = row
        elif row.get('event') == 'suggest':
            suggested.add(sid)
        elif row.get('event') == 'expand':
            who = sessions.get(sid)
            if who == 'sim':
                excluded['study_author_exploration_sim'] += 1
                continue
            if not who:
                excluded['study_unknown_participant'] += 1
                continue
            base = iid.get(sid, {})
            initial = (row['respin_of'] == 'iid') if row.get('respin_of') is not None else None
            # A typed request may echo a displayed suggestion; retain that evidence.
            records.append(
                dict(
                    id=digest([str(events_path.resolve()), line]),
                    source=str(events_path.resolve()),
                    line=line,
                    source_sha256=source_sha,
                    source_group='study_v1',
                    participant=digest(['study_v1', who])[:12],
                    prompt=base.get('base_prompt', ''),
                    original_prompt_id=sid,
                    requests=[row['axis']],
                    want_variation=True,
                    none_reason='',
                    images=row.get('source_images', []),
                    image_provenance='event source_images',
                    axis_source=row.get('axis_source'),
                    initial_iid=initial,
                    suggestions_preceded_request=sid in suggested,
                    provenance=(
                        'interactive_exposed_to_suggestions' if sid in suggested else 'interactive'
                    ),
                    approved=False,
                    review_note='Needs human authorship and source-image review',
                )
            )
    # Keep only records with a nonempty request; no-axis responses stay in the source files.
    with_axes = [r for r in records if any(a.strip() for a in r['requests'])]
    excluded['user_excluded_no_axis_responses'] += len(records) - len(with_axes)
    records = with_axes
    return dict(
        schema_version=1,
        kind='axis_example_candidates',
        examples=records,
        excluded=dict(excluded),
        counts=dict(Counter(r['source_group'] for r in records)),
    )


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
