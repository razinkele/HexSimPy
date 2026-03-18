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

    # --- UPSTREAM (vectorized) ---
    mask_up = alive & (pool.behavior == Behavior.UPSTREAM)
    if mask_up.any():
        idx = np.where(mask_up)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(tri_buf, water_nbrs, water_nbr_count,
                           fields["ssh"], rng, n_micro_steps, ascending=False)
        pool.tri_idx[idx] = tri_buf

    # --- DOWNSTREAM (vectorized) ---
    mask_down = alive & (pool.behavior == Behavior.DOWNSTREAM)
    if mask_down.any():
        idx = np.where(mask_down)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(tri_buf, water_nbrs, water_nbr_count,
                           fields["ssh"], rng, n_micro_steps, ascending=True)
        pool.tri_idx[idx] = tri_buf

    # --- TO_CWR (vectorized) ---
    mask_cwr = alive & (pool.behavior == Behavior.TO_CWR)
    if mask_cwr.any():
        idx = np.where(mask_cwr)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_to_cwr_vec(tri_buf, water_nbrs, water_nbr_count,
                         fields["temperature"], n_micro_steps, cwr_threshold)
        pool.tri_idx[idx] = tri_buf

    _apply_current_advection_vec(pool, mesh, fields, alive, rng)


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


def _step_directed_vec(tri_indices, water_nbrs, water_nbr_count, field,
                       rng, steps, ascending):
    """Vectorized gradient-following for a batch of agents.

    Even micro-steps: move to neighbor with best field value.
    Odd micro-steps: random jitter (move to random neighbor).
    """
    n = len(tri_indices)

    for s in range(steps):
        current = tri_indices
        counts = water_nbr_count[current]
        has_nbrs = counts > 0
        if not has_nbrs.any():
            break

        if s % 2 == 0:
            # Gradient step: gather field values at all neighbors
            nbr_matrix = water_nbrs[current]
            safe_idx = np.where(nbr_matrix >= 0, nbr_matrix, 0)
            nbr_vals = field[safe_idx]
            invalid = nbr_matrix < 0
            if ascending:
                nbr_vals[invalid] = -np.inf
                best_local = np.argmax(nbr_vals, axis=1)
            else:
                nbr_vals[invalid] = np.inf
                best_local = np.argmin(nbr_vals, axis=1)
            chosen = nbr_matrix[np.arange(n), best_local]
            tri_indices[has_nbrs] = chosen[has_nbrs]
        else:
            # Random jitter step
            rand_idx = rng.integers(0, np.maximum(counts, 1))
            chosen = water_nbrs[current, rand_idx]
            tri_indices[has_nbrs] = chosen[has_nbrs]


def _step_to_cwr_vec(tri_indices, water_nbrs, water_nbr_count, temperature,
                     steps, cwr_threshold):
    """Vectorized cold-water refuge seeking for a batch of agents."""
    n = len(tri_indices)

    for _ in range(steps):
        current = tri_indices
        above_thresh = temperature[current] >= cwr_threshold
        counts = water_nbr_count[current]
        active = above_thresh & (counts > 0)
        if not active.any():
            break

        nbr_matrix = water_nbrs[current]
        safe_idx = np.where(nbr_matrix >= 0, nbr_matrix, 0)
        nbr_temps = temperature[safe_idx]
        nbr_temps[nbr_matrix < 0] = np.inf
        best_local = np.argmin(nbr_temps, axis=1)
        chosen = nbr_matrix[np.arange(n), best_local]
        tri_indices[active] = chosen[active]


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


def _apply_current_advection_vec(pool, mesh, fields, alive_mask, rng):
    """Vectorized current advection for all alive agents."""
    u = fields.get("u_current")
    v = fields.get("v_current")
    if u is None or v is None:
        return

    alive_idx = np.where(alive_mask)[0]
    if len(alive_idx) == 0:
        return

    tris = pool.tri_idx[alive_idx]
    speeds = np.sqrt(u[tris]**2 + v[tris]**2)

    moving = speeds >= 0.01
    if not moving.any():
        return

    mov_idx = alive_idx[moving]
    mov_tris = pool.tri_idx[mov_idx]
    mov_speeds = speeds[moving]
    water_nbrs = mesh._water_nbrs
    water_nbr_count = mesh._water_nbr_count

    counts = water_nbr_count[mov_tris]
    has_nbrs = counts > 0
    if not has_nbrs.any():
        return

    # Flow direction for each agent: (v, u) normalized
    flow_y = v[mov_tris].copy()
    flow_x = u[mov_tris].copy()
    flow_norm = np.sqrt(flow_y**2 + flow_x**2) + 1e-12
    flow_y /= flow_norm
    flow_x /= flow_norm

    # Gather neighbor centroids and compute direction vectors
    nbr_matrix = water_nbrs[mov_tris]
    safe_idx = np.where(nbr_matrix >= 0, nbr_matrix, 0)

    c0 = mesh.centroids[mov_tris]                            # (n, 2)
    cn = mesh.centroids[safe_idx]                            # (n, max_nbrs, 2)
    dy = cn[:, :, 0] - c0[:, np.newaxis, 0]                 # (n, max_nbrs)
    dx = cn[:, :, 1] - c0[:, np.newaxis, 1]                 # (n, max_nbrs)
    dnorm = np.sqrt(dy**2 + dx**2) + 1e-12
    dy /= dnorm
    dx /= dnorm

    # Dot product with flow direction
    dots = dy * flow_y[:, np.newaxis] + dx * flow_x[:, np.newaxis]
    dots[nbr_matrix < 0] = -999.0

    # Best neighbor per agent
    best_local = np.argmax(dots, axis=1)
    best_nbr = nbr_matrix[np.arange(len(mov_tris)), best_local]

    # Probabilistic drift
    drift_prob = np.minimum(mov_speeds * 5.0, 0.8)
    drift = rng.random(len(mov_tris)) < drift_prob
    update = has_nbrs & drift

    pool.tri_idx[mov_idx[update]] = best_nbr[update]


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
