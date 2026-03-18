"""Movement kernels on the triangular mesh."""
from __future__ import annotations

import numpy as np

from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.mesh import TriMesh


def execute_movement(pool, mesh, fields, seed=None, n_micro_steps=3, cwr_threshold=16.0):
    rng = np.random.default_rng(seed)
    alive = pool.alive & ~pool.arrived

    water_nbrs = mesh._water_nbrs
    water_nbr_count = mesh._water_nbr_count

    # --- RANDOM (vectorized) ---
    mask_random = alive & (pool.behavior == Behavior.RANDOM)
    if mask_random.any():
        idx = np.where(mask_random)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_random_vec(tri_buf, water_nbrs, water_nbr_count, rng, n_micro_steps)
        pool.tri_idx[idx] = tri_buf

    # --- Per-agent fallback for UPSTREAM, DOWNSTREAM, TO_CWR ---
    for i in np.where(alive & (pool.behavior >= Behavior.TO_CWR))[0]:
        beh = pool.behavior[i]
        tri = pool.tri_idx[i]
        if beh == Behavior.UPSTREAM:
            pool.tri_idx[i] = _step_directed(tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=False)
        elif beh == Behavior.DOWNSTREAM:
            pool.tri_idx[i] = _step_directed(tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=True)
        elif beh == Behavior.TO_CWR:
            pool.tri_idx[i] = _step_to_cwr(tri, mesh, fields["temperature"], rng, n_micro_steps, cwr_threshold=cwr_threshold)

    _apply_current_advection(pool, mesh, fields, np.where(alive)[0], rng)


def _step_random_vec(tri_indices, water_nbrs, water_nbr_count, rng, steps):
    """Vectorized random walk for a batch of agents."""
    n = len(tri_indices)
    for _ in range(steps):
        current = tri_indices
        counts = water_nbr_count[current]
        has_nbrs = counts > 0
        if not has_nbrs.any():
            break
        rand_idx = rng.integers(0, np.maximum(counts, 1))
        chosen = water_nbrs[current, rand_idx]
        tri_indices[has_nbrs] = chosen[has_nbrs]


def _step_random(tri, mesh, rng, steps):
    current = tri
    for _ in range(steps):
        nbrs = mesh.water_neighbors(current)
        if not nbrs:
            break
        current = rng.choice(nbrs)
    return current


def _step_directed(tri, mesh, field, rng, steps, ascending):
    current = tri
    for s in range(steps):
        if s % 2 == 0:
            # gradient following
            nbrs = mesh.water_neighbors(current)
            if not nbrs:
                break
            vals = np.array([field[n] for n in nbrs])
            if ascending:
                best = nbrs[np.argmax(vals)]
            else:
                best = nbrs[np.argmin(vals)]
            current = best
        else:
            # random jitter
            nbrs = mesh.water_neighbors(current)
            if nbrs:
                current = rng.choice(nbrs)
    return current


def _step_to_cwr(tri, mesh, temperature, rng, steps, cwr_threshold=16.0):
    current = tri
    for _ in range(steps):
        if temperature[current] < cwr_threshold:
            break
        nbrs = mesh.water_neighbors(current)
        if not nbrs:
            break
        temps = np.array([temperature[n] for n in nbrs])
        current = nbrs[np.argmin(temps)]
    return current


def _apply_current_advection(pool, mesh, fields, alive_idx, rng):
    u = fields.get("u_current")
    v = fields.get("v_current")
    if u is None or v is None:
        return

    for i in alive_idx:
        tri = pool.tri_idx[i]
        speed = np.sqrt(u[tri]**2 + v[tri]**2)
        if speed < 0.01:
            continue
        nbrs = mesh.water_neighbors(tri)
        if not nbrs:
            continue
        flow_dir = np.array([v[tri], u[tri]])
        flow_dir /= np.linalg.norm(flow_dir) + 1e-12
        best_dot = -999.0
        best_nbr = tri
        c0 = mesh.centroids[tri]
        for n in nbrs:
            cn = mesh.centroids[n]
            d = cn - c0
            d /= np.linalg.norm(d) + 1e-12
            dot = np.dot(d, flow_dir)
            if dot > best_dot:
                best_dot = dot
                best_nbr = n
        if rng.random() < min(speed * 5.0, 0.8):
            pool.tri_idx[i] = best_nbr
