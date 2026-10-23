"""Generation diversity and fidelity math, independently testable on CPU."""

import numpy as np
from .protocol import (
    validate_reward,
    validate_metrics,
    malformed_score,
    length_excess,
    MAX_QUALITY_DEFICIT,
)


def diversity(embeddings):
    e = np.asarray(embeddings, dtype=np.float64)
    if e.ndim != 2 or len(e) < 2 or not np.isfinite(e).all():
        raise ValueError('Require at least two finite embedding vectors')
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('Zero embedding vector')
    e = e / norms
    gram = e @ e.T
    pairwise = float(np.mean(1 - gram[np.triu_indices(len(e), k=1)]))
    eigenvalues = np.linalg.eigvalsh(gram / len(e))
    if eigenvalues.min() < -1e-10:
        raise ValueError('Invalid cosine Gram matrix')
    w = np.clip(eigenvalues, 0, None)
    w = w[w > 0]
    w /= w.sum()
    vendi = float(np.exp(-np.sum(w * np.log(w))))
    return dict(
        pairwise=float(np.clip(pairwise, 0, 2)),
        vendi=vendi,
        normalized_vendi=float(np.clip((vendi - 1) / (len(e) - 1), 0, 1)),
    )


def score(
    embeddings,
    alignment,
    factor,
    config,
    overall_embeddings=None,
    threshold=None,
    prompts=None,
    drift=None,
    quality=None,
    quality_threshold=None,
    length_cap=None,
):
    # `threshold` is the row's own align_thresh; None selects config['fidelity_threshold'].
    # `prompts` are the expanded prompts; required when the configuration has a length term.
    # `drift` holds, for the editing task, the distance of each edited image from its source.
    # `quality` holds the HPSv3 score of each image and `quality_threshold` the row's quality_thresh;
    # both are required when the configuration has a quality term.
    validate_reward(config)
    if threshold is None:
        threshold = config['fidelity_threshold']
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise ValueError('Fidelity threshold must be a probability in [0, 1]')
    d = diversity(embeddings)
    a = np.asarray(alignment, dtype=np.float64)
    if a.shape != (len(embeddings),) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Require one VQAScore probability per generated image')
    metric_key = 'pairwise' if config['metric'] == 'pairwise' else 'normalized_vendi'
    dist = d[metric_key]
    floor = 0.0
    # With floor='overall', named-axis rows subtract generic diversity. Overall rows never do:
    # generic diversity is their objective.
    if config['floor'] == 'overall' and factor != 'overall':
        if overall_embeddings is None:
            raise ValueError('Missing generic-diversity embeddings')
        if len(overall_embeddings) != len(embeddings):
            raise ValueError('Floor image count mismatch')
        floor = diversity(overall_embeddings)[metric_key]
    penalty = float(np.maximum(0, threshold - a).mean())
    reward = (
        dist
        - floor
        + config['fidelity_weight'] * (float(a.mean()) if config['mode'] == 'linear' else -penalty)
    )
    length = 0.0
    if config.get('length_weight'):
        if prompts is None or len(prompts) != len(a):
            raise ValueError('The length term requires the expanded prompts')
        # `length_cap` is the row's own cap; None selects the configuration's.
        length = config['length_weight'] * length_excess(
            prompts, length_cap or config['length_cap']
        )
        reward -= length
    words = float(np.mean([len(p.split()) for p in prompts])) if prompts is not None else 0.0
    drift_mean = drift_penalty = 0.0
    if drift is not None:
        if 'drift_weight' not in config:
            raise ValueError('An edited set needs drift_weight and drift_threshold in the reward')
        distances = np.asarray(drift, dtype=np.float64)
        if (
            distances.shape != (len(a),)
            or not np.isfinite(distances).all()
            or np.any(distances < 0)
        ):
            raise ValueError('Require one non-negative drift distance per edited image')
        drift_mean = float(distances.mean())
        excess = np.minimum(1.0, np.maximum(0.0, distances - config['drift_threshold']))
        drift_penalty = config['drift_weight'] * float(excess.mean())
        reward -= drift_penalty
    extra = {}
    if 'quality_weight' in config:
        if quality is None or quality_threshold is None:
            raise ValueError(
                'The quality term requires per-image quality scores and the row quality_thresh'
            )
        q = np.asarray(quality, dtype=np.float64)
        if q.shape != (len(a),) or not np.isfinite(q).all():
            raise ValueError('Require one finite quality score per image')
        deficit = np.minimum(
            MAX_QUALITY_DEFICIT, np.maximum(0.0, quality_threshold - config['quality_margin'] - q)
        )
        extra = dict(
            quality_mean=float(q.mean()),
            quality_threshold=float(quality_threshold),
            quality_penalty=config['quality_weight'] * float(deficit.mean()),
        )
        reward -= extra['quality_penalty']
    if reward <= malformed_score(config):
        raise ValueError('Valid reward crossed the malformed lower bound')
    return validate_metrics(
        dict(
            score=reward,
            reward=reward,
            diversity=dist,
            floor=floor,
            fidelity_mean=float(a.mean()),
            fidelity_min=float(a.min()),
            fidelity_penalty=penalty,
            fidelity_threshold=float(threshold),
            length_penalty=length,
            words_per_prompt=words,
            drift_mean=drift_mean,
            drift_penalty=drift_penalty,
            malformed=0.0,
            n_parsed=float(len(a)),
            **extra,
            **d
        )
    )
