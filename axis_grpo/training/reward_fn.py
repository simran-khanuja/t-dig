"""verl async custom reward client. Service faults raise; bad policy syntax is penalized."""

import asyncio
import itertools
import json
import threading
import urllib.request

from hpdv2.storage import digest
from axis_grpo.training.protocol import (
    PROTOCOL,
    malformed_metrics,
    parse_prompts,
    parse_spec,
    request_id,
    seed_for,
    validate_item,
    validate_metrics,
    validate_reward,
)

_COUNTERS = {}
_LOCK = threading.Lock()


def endpoint_for(routes, t2i_model):
    if t2i_model not in routes or not routes[t2i_model]:
        raise ValueError(f'No configured reward endpoint for T2I model {t2i_model}')
    urls = routes[t2i_model]
    if not isinstance(urls, list) or any(
        not isinstance(u, str) or not u.startswith('http://') for u in urls
    ):
        raise ValueError('Use an explicit list of internal HTTP reward endpoints')
    with _LOCK:
        counter = _COUNTERS.setdefault(tuple(urls), itertools.count())
        return urls[next(counter) % len(urls)].rstrip('/')


def post(url, item, t2i_model, reward, timeout, service_hashes=None, tracked=()):
    identity = request_id(item)
    payload = dict(
        protocol=PROTOCOL,
        request_id=identity,
        renderer=t2i_model,
        reward_hash=digest(reward),
        item=item,
    )
    req = urllib.request.Request(
        url + '/score',
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={'Content-Type': 'application/json'},
    )
    # No automatic retries: the server journals completed requests for explicit
    # resumption, and a timeout is not evidence that rendering did not occur.
    with urllib.request.urlopen(req, timeout=timeout) as r:
        answer = json.loads(r.read())
    if (
        answer.get('protocol') != PROTOCOL
        or answer.get('request_id') != identity
        or answer.get('renderer') != t2i_model
        or answer.get('reward_hash') != digest(reward)
    ):
        raise RuntimeError('Reward response provenance mismatch')
    if service_hashes is not None and answer.get('service_hash') != service_hashes.get(url):
        raise RuntimeError('Reward service configuration changed during training')
    return validate_metrics(answer['metrics'], tracked)


async def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    routes=None,
    reward=None,
    renderer='sd35-medium',
    routing='fixed',
    seed=0,
    seed_scope='base',
    timeout=1800,
    service_hashes=None,
    tracked=(),
    **kwargs,
):
    if data_source != 'axis_diversity':
        raise ValueError('Unexpected training data source')
    if isinstance(routes, str):
        routes = json.loads(routes)
    if isinstance(reward, str):
        reward = json.loads(reward)
    if isinstance(service_hashes, str):
        service_hashes = json.loads(service_hashes)
    if isinstance(tracked, str):
        tracked = json.loads(tracked)
    validate_reward(reward)
    spec = parse_spec(ground_truth)
    if routing != 'fixed':
        raise ValueError('Each policy must use one fixed training T2I model')
    extra_info = extra_info or {}
    selected = renderer
    url = endpoint_for(routes, selected)
    try:
        prompts = parse_prompts(solution_str, spec['k'])
    except ValueError:
        return malformed_metrics(reward, tracked=tracked)
    item = validate_item(
        dict(
            **spec,
            prompts=prompts,
            seed=seed_for(spec['base_prompt'], spec['tpips_factor'], seed, seed_scope),
            split=extra_info.get('split', 'train'),
            row_id=extra_info.get('row_id', ''),
        )
    )
    return await asyncio.to_thread(
        post, url, item, selected, reward, timeout, service_hashes, tracked
    )
