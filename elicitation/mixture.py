"""Choose the policy-input wording per training row; the reward phrase stays fixed.

Input rows are semantically reviewed candidates, not raw API responses.
"""

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import random

from hpdv2.storage import digest


def mix(rows, short_fraction, seed=42, *, reward_wording):
    fraction = Decimal(str(short_fraction))
    if not fraction.is_finite() or not 0 <= fraction <= 1:
        raise ValueError('Short fraction must lie in [0,1]')
    if reward_wording not in {'short', 'natural'}:
        raise ValueError('Choose reward wording explicitly after the TPIPS wording probe')
    groups = defaultdict(list)
    ids = set()
    prompt_t2i_models = {}
    prompt_splits = {}
    for row in rows:
        if row.get('semantic_review') != 'approved':
            raise ValueError('Export requires reviewed semantic mappings')
        if row['mapping_status'] != 'faithful' or row.get('constraints'):
            raise ValueError('Ambiguous/constrained requests need a separate reward design')
        ident = (row['prompt_id'], row['axis_id'])
        if ident in ids:
            raise ValueError('Duplicate prompt-axis row')
        ids.add(ident)
        for mapping, key, value in (
            (prompt_t2i_models, row['prompt_id'], row['renderer']),
            (prompt_splits, row['cluster_id'], row['split']),
        ):
            if key in mapping and mapping[key] != value:
                raise ValueError('Prompt T2I model or cluster split is inconsistent')
            mapping[key] = value
        groups[(row['renderer'], row['split'])].append(row)
    result = []
    for group, candidates in sorted(groups.items()):
        ordered = sorted(candidates, key=lambda r: (r['prompt_id'], r['axis_id']))
        random.Random(digest([seed, group, 'wording-mix-v1'])).shuffle(ordered)
        n_short = int((fraction * len(ordered)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        for i, row in enumerate(ordered):
            style = 'tpips' if i < n_short else 'natural'
            result.append(
                {
                    **row,
                    'input_style': style,
                    'axis_input': row['tpips_factor'] if style == 'tpips' else row['request'],
                    'reward_factor': (
                        row['tpips_factor'] if reward_wording == 'short' else row['request']
                    ),
                }
            )
    return result
