"""CADS math and Group Inference objective; tensor imports are lazy."""

import math


def gamma(time, tau1, tau2):
    if not 0 <= time <= 1 or not 0 <= tau1 < tau2 <= 1:
        raise ValueError("Invalid CADS time or schedule")
    return min(1.0, max(0.0, (tau2 - time) / (tau2 - tau1)))


def cads_time(step, steps, scheduler_time, mode):
    """Return the CADS annealing time for one denoising step.

    `sampling-progress` runs from 1 at the first step to 0 at the last (1 for a
    one-step run), independent of SD3's shifted model time.
    `scheduler-timestep/1000` returns `scheduler_time`.
    """
    if steps < 1 or not 0 <= step < steps or not 0 <= scheduler_time <= 1:
        raise ValueError("Invalid CADS step/time")
    if mode == "sampling-progress":
        return 1.0 - step / max(steps - 1, 1)
    if mode == "scheduler-timestep/1000":
        return scheduler_time
    raise ValueError("Unsupported CADS time convention")


def corrupt(clean, amount, noise_scale, psi, generator):
    import torch

    if amount == 1:
        return clean  # Exact identity, consumes no randomness.
    # One call per candidate/branch/tensor; preserve per-sample statistics.
    value = clean.float()
    noise = torch.randn(value.shape, generator=generator, device="cpu", dtype=torch.float32).to(
        value.device
    )
    changed = math.sqrt(amount) * value + noise_scale * math.sqrt(1 - amount) * noise
    if psi:
        dims = tuple(range(1, value.ndim))
        std = changed.std(dim=dims, keepdim=True, correction=0)
        if (std <= 1e-12).any():
            raise ValueError("Degenerate CADS rescale variance")
        rescaled = (changed - changed.mean(dim=dims, keepdim=True)) / std
        rescaled = rescaled * value.std(dim=dims, keepdim=True, correction=0) + value.mean(
            dim=dims, keepdim=True
        )
        changed = psi * rescaled + (1 - psi) * changed
    return changed.to(clean.dtype)


def clean_prediction(latent_before_step, velocity, sigma):
    # Flow matching: x_sigma = (1-sigma)*x0 + sigma*noise, v=noise-x0.
    return latent_before_step.float() - float(sigma) * velocity.float()


def next_size(current, k, retention, last=False):
    if current < k or k < 1 or not 0 < retention < 1:
        raise ValueError("Invalid pruning schedule")
    return (
        k
        if last
        else (max(k, min(current - 1, math.ceil(current * retention))) if current > k else k)
    )


def cardinality_kernel(pairwise):
    """Return an equivalent PSD quadratic form for fixed-cardinality binary selection.

    Only the upper triangle defines the original pair objective. With its
    symmetric zero-diagonal matrix D, G=11' - D has unit diagonal. For binary
    x with sum(x)=k, sum_{i<j} D_ij*x_i*x_j = (k*k - x'Gx)/2.
    Adding delta*I to G and delta*k to the constant preserves every feasible
    objective exactly while making G positive semidefinite. Cosine distances
    already yield a PSD G up to numerical error; arbitrary distances may need
    a larger shift, which can weaken the continuous bound.
    """
    import numpy as np

    d = np.asarray(pairwise, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not len(d) or not np.isfinite(d).all():
        raise ValueError("Invalid pairwise matrix")
    upper = np.triu(d, 1)
    kernel = np.ones_like(d) - upper - upper.T
    shift = max(0.0, -float(np.linalg.eigvalsh(kernel)[0])) + 1e-9
    kernel += shift * np.eye(len(d))
    return kernel, shift


def solve_subset(unary, pairwise, count, weight, seconds, accept_incumbent=False):
    import numpy as np
    import gurobipy as gp

    u, d = np.asarray(unary, dtype=float), np.asarray(pairwise, dtype=float)
    n = len(u)
    if u.shape != (n,) or d.shape != (n, n) or not np.isfinite(u).all() or not np.isfinite(d).all():
        raise ValueError("Invalid group scores")
    if (
        type(count) is not int
        or not 1 <= count <= n
        or not math.isfinite(weight)
        or weight < 0
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("Invalid group selection")
    kernel, shift = cardinality_kernel(d)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            model.Params.TimeLimit = seconds
            model.Params.Threads = 1
            model.Params.Seed = 0
            model.Params.MIPGap = 0.0
            # Keep the convex MIQP relaxation; linearizing pair products gives a looser MILP.
            model.Params.PreQLinearize = 0
            model.Params.NonConvex = 0
            x = model.addMVar(n, vtype=gp.GRB.BINARY)
            model.addConstr(x.sum() == count)
            objective = u @ x + 0.5 * weight * (count**2 + shift * count - x @ kernel @ x)
            model.setObjective(objective, gp.GRB.MAXIMIZE)
            model.optimize()
            if not model.SolCount or (model.Status != gp.GRB.OPTIMAL and not accept_incumbent):
                raise ValueError(
                    f"Group solver did not prove optimality: status={model.Status}, solutions={model.SolCount}"
                )
            selected = np.flatnonzero(x.X > 0.5).tolist()
            if len(selected) != count:
                raise ValueError("Solver returned wrong subset size")
            original_objective = sum(float(u[i]) for i in selected) + weight * sum(
                float(d[i, j]) for a, i in enumerate(selected) for j in selected[a + 1 :]
            )
            if not math.isclose(original_objective, model.ObjVal, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("Reformulated solver objective disagrees with original objective")
            return selected, {
                "status": int(model.Status),
                "optimal": model.Status == gp.GRB.OPTIMAL,
                "objective": original_objective,
                "gap": model.MIPGap,
                "runtime": model.Runtime,
                "bound": model.ObjBound,
                "node_count": model.NodeCount,
                "formulation": "cardinality-psd-v1",
                "diagonal_shift": shift,
            }
