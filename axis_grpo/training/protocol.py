"""Shared JSON wire contract and deterministic seeds; no GPU/framework imports."""

import hashlib
import json
import math
import re

from hpdv2.storage import digest

PROTOCOL = 1
METRICS = (
    'score',
    'reward',
    'diversity',
    'pairwise',
    'vendi',
    'normalized_vendi',
    'floor',
    'fidelity_mean',
    'fidelity_min',
    'fidelity_penalty',
    'fidelity_threshold',
    'length_penalty',
    'words_per_prompt',
    'drift_mean',
    'drift_penalty',
    'malformed',
    'n_parsed',
)
REWARD_KEYS = {'metric', 'mode', 'fidelity_weight', 'fidelity_threshold', 'floor'}
# Optional length term: reward -= length_weight * mean_i min(MAX_LENGTH_EXCESS, max(0, words_i - cap) / cap).
# Without these keys there is no length term, and nothing is added to the reward hash.
LENGTH_KEYS = {'length_weight', 'length_cap'}
MAX_LENGTH_EXCESS = 3.0
# Optional drift term of the editing task: reward -= drift_weight * mean_i min(1, max(0, d_i - drift_threshold)),
# where d_i is the perceptual distance between source image i and its edit.
DRIFT_KEYS = {'drift_weight', 'drift_threshold'}
# Optional quality term: reward -= quality_weight * mean_i min(MAX_QUALITY_DEFICIT,
# max(0, quality_thresh - quality_margin - q_i)), where q_i is the HPSv3 score of image i against
# the base prompt and quality_thresh is the row's own value, the mean HPSv3 of the T2I model's
# i.i.d. renders of the prompt. HPSv3 is unbounded, about 5 to 16 on these T2I models, so the
# weight is a cost per HPSv3 point. The margin is the deficit a set may have free of cost. A
# configuration with the term adds QUALITY_METRICS to every metric dictionary.
QUALITY_KEYS = {'quality_weight', 'quality_margin'}
QUALITY_METRICS = ('quality_mean', 'quality_threshold', 'quality_penalty')
QUALITY_SCORER = 'hpsv3'
MAX_QUALITY_DEFICIT = 6.0
# Optional ground-truth keys. `task` is 'edit' when the completion holds k edit instructions for
# the k images of `source_images`, in order; absent for the generation task.
# `length_cap` is a row's own word cap; it replaces the configuration's length_cap for that row
# (tdig/recap_dataset.py --relative-budget sets it to the base prompt's length plus a budget).
OPTIONAL_SPEC_KEYS = {'align_thresh', 'quality_thresh', 'length_cap', 'task', 'source_images'}
SPEC_KEYS = {'base_prompt', 'axis', 'tpips_factor', 'k'}
# Logged-only scores a reward service can add to METRICS (axis_grpo/training/tracked.py).
TRACKED = ('imagereward', 'hpsv3', 'hpsv2', 'clip_score', 'aesthetic')


def validate_tracked(names):
    if (
        not isinstance(names, (list, tuple))
        or len(set(names)) != len(names)
        or set(names) - set(TRACKED)
    ):
        raise ValueError(f'track_metrics must be distinct names from {TRACKED}')
    return tuple(names)


PROMPTS = re.compile(r'<prompt>(.*?)</prompt>', re.DOTALL | re.IGNORECASE)


def validate_reward(config):
    if (
        set(config) - LENGTH_KEYS - DRIFT_KEYS - QUALITY_KEYS != REWARD_KEYS
        or len(set(config) & LENGTH_KEYS) == 1
        or len(set(config) & DRIFT_KEYS) == 1
        or len(set(config) & QUALITY_KEYS) == 1
    ):
        raise ValueError(
            'Missing or unknown reward setting; length_*, drift_* and quality_* keys come in pairs'
        )
    for key in set(config) & QUALITY_KEYS:
        v = config[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f'{key} must be a finite non-negative number')
    if 'drift_weight' in config:
        for key in DRIFT_KEYS:
            v = config[key]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0
            ):
                raise ValueError(f'{key} must be a finite non-negative number')
    if 'length_weight' in config:
        w, cap = config['length_weight'], config['length_cap']
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
            raise ValueError('length_weight must be a finite non-negative number')
        if type(cap) is not int or cap < 1:
            raise ValueError('length_cap must be a positive integer number of words')
    if config['metric'] not in {'pairwise', 'vendi'} or config['mode'] not in {'hinge', 'linear'}:
        raise ValueError('Unsupported generation reward')
    if config['floor'] not in {'none', 'overall'}:
        raise ValueError('Unsupported diversity floor')
    for key in ('fidelity_weight', 'fidelity_threshold'):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f'{key} must be finite')
    if config['fidelity_weight'] < 0 or not 0 <= config['fidelity_threshold'] <= 1:
        raise ValueError('Invalid fidelity weight or threshold')
    return config


def malformed_score(config):
    validate_reward(config)
    # Diversity minus floor is at least -2 (both lie in [0, 2]) and each penalty is at most its
    # weight times its cap, so this is strictly below every valid reward.
    return (
        -3.0
        - config['fidelity_weight']
        - MAX_LENGTH_EXCESS * config.get('length_weight', 0.0)
        - config.get('drift_weight', 0.0)
        - MAX_QUALITY_DEFICIT * config.get('quality_weight', 0.0)
    )


def length_excess(prompts, cap):
    """Return the mean fraction by which each prompt's word count exceeds `cap`.

    Each fraction is capped at MAX_LENGTH_EXCESS. Words are whitespace-separated, the unit the
    policy's instruction uses.
    """
    return sum(min(MAX_LENGTH_EXCESS, max(0, len(p.split()) - cap) / cap) for p in prompts) / len(
        prompts
    )


def malformed_metrics(config, n_parsed=0, tracked=()):
    # Carry every key a scored completion returns: verl averages each key over the batch and
    # fails on a missing one.
    metrics = dict.fromkeys(
        METRICS
        + validate_tracked(tracked)
        + (QUALITY_METRICS if 'quality_weight' in config else ()),
        0.0,
    )
    metrics.update(
        score=malformed_score(config),
        reward=malformed_score(config),
        malformed=1.0,
        n_parsed=float(n_parsed),
    )
    return metrics


def parse_prompts(text, k):
    if not isinstance(text, str):
        raise ValueError('Completion must be text')
    matches = list(PROMPTS.finditer(text))
    if len(matches) != k or PROMPTS.sub('', text).strip():
        raise ValueError(f'Expected exactly {k} prompt tags and no surrounding text')
    result = [m.group(1).strip() for m in matches]
    if any(not p or re.search(r'</?prompt\b', p, re.I) for p in result):
        raise ValueError('Empty or nested prompt tags')
    return result


def parse_spec(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) - OPTIONAL_SPEC_KEYS != SPEC_KEYS:
        raise ValueError('Use the generation dataset ground_truth schema (axis_grpo.dataset)')
    if ('task' in value) != ('source_images' in value):
        raise ValueError('task and source_images come together')
    if 'task' in value:
        sources = value['source_images']
        if (
            value['task'] != 'edit'
            or not isinstance(sources, list)
            or any(not isinstance(p, str) or not p for p in sources)
        ):
            raise ValueError("task must be 'edit' with a list of source image paths")
        if type(value.get('k')) is int and len(sources) != value['k']:
            raise ValueError('An edit row needs one source image per instruction')
    if 'length_cap' in value and (type(value['length_cap']) is not int or value['length_cap'] < 1):
        raise ValueError('length_cap must be a positive integer number of words')
    if 'quality_thresh' in value:
        q = value['quality_thresh']
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q):
            raise ValueError('quality_thresh must be a finite HPSv3 score')
    # Optional per-row fidelity threshold (tdig/iid_vqa.py attach). When present it
    # replaces the reward configuration's fidelity_threshold for this row.
    if 'align_thresh' in value:
        t = value['align_thresh']
        if (
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            or not 0 <= t <= 1
        ):
            raise ValueError('align_thresh must be a VQAScore probability in [0, 1]')
    for key in ('base_prompt', 'axis', 'tpips_factor'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f'Empty {key}')
    if type(value['k']) is not int or value['k'] < 2:
        raise ValueError('Diversity training requires k >= 2')
    if (value['axis'] == 'overall') != (value['tpips_factor'] == 'overall'):
        raise ValueError('The overall request and reward condition must agree')
    return value


def seed_for(base_prompt, factor, seed=0, scope='base'):
    if type(seed) is not int or scope not in {'base', 'pair'}:
        raise ValueError('Invalid seed settings')
    key = base_prompt if scope == 'base' else f'{base_prompt}\x1f{factor}'
    return (int(hashlib.md5(key.encode()).hexdigest()[:8], 16) + seed) % (2**31 - 1)


def request_id(item):
    return digest(item)


def validate_item(item):
    required = SPEC_KEYS | {'prompts', 'seed', 'split', 'row_id'}
    if not isinstance(item, dict) or set(item) - OPTIONAL_SPEC_KEYS != required:
        raise ValueError('Unknown or missing reward item field')
    parse_spec({k: item[k] for k in SPEC_KEYS | OPTIONAL_SPEC_KEYS if k in item})
    if (
        not isinstance(item['prompts'], list)
        or len(item['prompts']) != item['k']
        or any(not isinstance(p, str) or not p.strip() for p in item['prompts'])
    ):
        raise ValueError('Wrong expanded-prompt list')
    if type(item['seed']) is not int or not 0 <= item['seed'] < 2**63:
        raise ValueError('Invalid latent seed')
    if item['split'] not in {'train', 'val'} or not isinstance(item['row_id'], str):
        raise ValueError('Invalid training provenance')
    return item


def validate_metrics(metrics, tracked=()):
    expected = set(METRICS + validate_tracked(tracked))
    if not isinstance(metrics, dict) or set(metrics) not in (
        expected,
        expected | set(QUALITY_METRICS),
    ):
        raise ValueError('Reward metric schema mismatch')
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in metrics.values()
    ):
        raise ValueError('Non-finite/non-numeric reward metric')
    if metrics['score'] != metrics['reward']:
        raise ValueError('Inconsistent score and reward')
    return {k: float(v) for k, v in metrics.items()}
