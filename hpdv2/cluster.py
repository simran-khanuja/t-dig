"""Global average-linkage clustering, deterministic capped selection, provenance."""

from collections import Counter
from pathlib import Path
import random

from .dataset import SCHEMA, load_input
from .runner import environment
from .storage import atomic_json, digest


def select_members(vectors, labels, cap, min_size, *, selection="max-min", seed=42, ids=None):
    """Select up to `cap` members of each cluster of at least `min_size`, without replacement.

    `random` seeds each cluster's RNG from its sorted member IDs, so the choice does not depend
    on label order. `max-min` starts from the centroid-nearest member and adds greedy max-min picks.
    """
    if selection not in {"random", "max-min"}:
        raise ValueError("Unknown cluster selection method")
    if selection == "random" and (ids is None or len(ids) != len(vectors)):
        raise ValueError("Random selection requires stable prompt IDs")
    import numpy as np

    selected, groups = [], []
    by_label = {}
    for i, label in enumerate(labels.tolist()):
        by_label.setdefault(label, []).append(i)
    for label in sorted(by_label):
        indices = np.asarray(by_label[label], dtype=np.int64)
        if len(indices) < min_size:
            continue
        points = vectors[indices]
        centroid = points.mean(axis=0)
        anchor = int((points @ centroid).argmax())
        chosen = [anchor]
        while selection == "max-min" and len(chosen) < min(cap, len(indices)):
            similarities = (points @ points[chosen].T).max(axis=1)
            similarities[chosen] = float("inf")
            chosen.append(int(similarities.argmin()))
        if selection == "random":
            members = sorted((ids[int(i)], int(i)) for i in indices)
            cluster_seed = int(
                digest({"seed": seed, "members": [ident for ident, _ in members]}), 16
            )
            picks = random.Random(cluster_seed).sample(
                [i for _, i in members], min(cap, len(members))
            )
        else:
            picks = [int(indices[j]) for j in chosen]
        selected.extend(picks)
        groups.append((indices.tolist(), picks, int(indices[anchor])))
    return selected, groups


def cluster(
    input_path,
    output,
    *,
    distance,
    cap=1,
    min_size=1,
    selection="random",
    seed=42,
    model="sentence-transformers/all-MiniLM-L6-v2",
    revision=None,
    device="cpu",
    max_matrix_gib=4.0,
    execute=False,
    allow_download=False,
    allow_partial=False,
    embedder=None,
):
    if not 0 < distance <= 2 or cap < 1 or min_size < 1 or max_matrix_gib <= 0:
        raise ValueError("Invalid clustering settings")
    if selection not in {"random", "max-min"}:
        raise ValueError("Unknown cluster selection method")
    if Path(output).exists():
        raise ValueError("Output exists; use a new dataset version")
    data, _ = load_input(input_path, "cluster", allow_partial)
    records = sorted(data["kept"], key=lambda row: row["id"])
    # Float64 matrix only; sklearn/scipy allocate more on top.
    matrix_gib = len(records) ** 2 * 8 / 2**30
    settings = {
        "distance": distance,
        "max_per_cluster": cap,
        "min_cluster_size": min_size,
        "model": model,
        "revision": revision,
        "device": device,
        "method": "agglomerative",
        "metric": "cosine",
        "linkage": "average",
        "selection": selection,
        "seed": seed,
        "matrix_gib": matrix_gib,
    }
    if not execute:
        return {"n_input": len(records), **settings, "execute": False}
    if matrix_gib > max_matrix_gib:
        raise ValueError(
            f"Distance matrix requires {matrix_gib:.2f} GiB, above --max-matrix-gib; total peak is larger"
        )
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    if records:
        if embedder is None:
            if not revision:
                raise ValueError("Pin --embedding-revision to an immutable model commit")
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(
                model, revision=revision, device=device, local_files_only=not allow_download
            )

            def embedder(texts):
                return encoder.encode(
                    texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True
                )

        vectors = np.asarray(embedder([r["prompt"] for r in records]), dtype=np.float64)
        if vectors.ndim != 2 or len(vectors) != len(records) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding matrix")
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ValueError("Zero embedding")
        vectors /= norm
        if len(records) == 1:
            labels = np.zeros(1, dtype=int)
        else:
            distances = vectors @ vectors.T
            distances *= -1
            distances += 1
            np.clip(distances, 0, 2, out=distances)
            np.fill_diagonal(distances, 0)
            labels = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance,
                metric="precomputed",
                linkage="average",
            ).fit_predict(distances)
        _, groups = select_members(
            vectors,
            labels,
            cap,
            min_size,
            selection=selection,
            seed=seed,
            ids=[r["id"] for r in records],
        )
        n_before_minimum = len(set(labels.tolist()))
    else:
        groups, n_before_minimum = [], 0
    kept, clusters = [], []
    # A cluster ID hashes its member IDs, so it changes whenever membership changes.
    for indices, picks, anchor in groups:
        cid = digest(sorted(records[i]["id"] for i in indices))
        rep = records[anchor]["prompt"]
        clusters.append(
            {
                "cluster_id": cid,
                "size": len(indices),
                "representative": rep,
                "member_ids": [records[i]["id"] for i in indices],
                "selected_ids": [records[i]["id"] for i in picks],
            }
        )
        kept.extend({**records[i], "cluster_id": cid, "cluster_rep": rep} for i in picks)
    kept.sort(key=lambda r: r["id"])
    result = {
        "schema_version": SCHEMA,
        "kind": "clustered_pool",
        "metadata": {
            "input_hash": digest(data),
            "upstream_run_hash": data["run_hash"],
            "upstream_complete": data["metadata"]["full_pipeline_complete"],
            "settings": settings,
            "environment": environment(),
            "n_input": len(records),
            "n_kept": len(kept),
            "n_removed": len(records) - len(kept),
            "n_clusters_before_minimum": n_before_minimum,
            "n_clusters": len(clusters),
            "variation_axis_count_distribution": dict(
                Counter(str(len(r["variation_axes"])) for r in kept)
            ),
        },
        "kept": kept,
        "cluster_summary": clusters,
    }
    atomic_json(output, result)
    return result["metadata"]
