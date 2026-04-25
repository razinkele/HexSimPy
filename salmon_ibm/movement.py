"""Movement kernels on the triangular mesh."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range

FORCE_NUMPY = False


def _use_numba():
    return HAS_NUMBA and not FORCE_NUMPY


from salmon_ibm.agents import Behavior


def execute_movement(
    pool,
    mesh,
    fields,
    seed=None,
    n_micro_steps=3,
    cwr_threshold=16.0,
    barrier_arrays=None,
):
    """Execute movement with optional barrier enforcement.

    Parameters
    ----------
    barrier_arrays : tuple (mort, defl, trans) from BarrierMap.to_arrays(), or None
    """
    rng = np.random.default_rng(seed)
    alive = pool.alive & ~pool.arrived

    water_nbrs = mesh._water_nbrs
    water_nbr_count = mesh._water_nbr_count

    # Save pre-movement positions for barrier resolution
    pre_move = pool.tri_idx.copy() if barrier_arrays is not None else None

    # Pre-bucket agents by behavior in a single pass (avoids 4 separate
    # mask computations + np.where calls — ~4x fewer array operations).
    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        _apply_current_advection_vec(pool, mesh, fields, alive, rng)
        return
    alive_beh = pool.behavior[alive_idx]
    buckets = {}
    for b in np.unique(alive_beh):
        buckets[int(b)] = alive_idx[alive_beh == b]

    # --- RANDOM ---
    idx = buckets.get(int(Behavior.RANDOM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        _step_random_vec(tri_buf, water_nbrs, water_nbr_count, rng, n_micro_steps)
        pool.tri_idx[idx] = tri_buf

    # --- UPSTREAM ---
    idx = buckets.get(int(Behavior.UPSTREAM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(
            tri_buf,
            water_nbrs,
            water_nbr_count,
            fields["ssh"],
            rng,
            n_micro_steps,
            ascending=False,
        )
        pool.tri_idx[idx] = tri_buf

    # --- DOWNSTREAM ---
    idx = buckets.get(int(Behavior.DOWNSTREAM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(
            tri_buf,
            water_nbrs,
            water_nbr_count,
            fields["ssh"],
            rng,
            n_micro_steps,
            ascending=True,
        )
        pool.tri_idx[idx] = tri_buf

    # --- TO_CWR ---
    idx = buckets.get(int(Behavior.TO_CWR))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        _step_to_cwr_vec(
            tri_buf,
            water_nbrs,
            water_nbr_count,
            fields["temperature"],
            n_micro_steps,
            cwr_threshold,
        )
        pool.tri_idx[idx] = tri_buf

    # --- Barrier enforcement (after all movement, before advection) ---
    if barrier_arrays is not None:
        alive_idx = np.where(alive)[0]
        if len(alive_idx) > 0:
            mort, defl, trans = barrier_arrays
            current = pre_move[alive_idx]
            proposed = pool.tri_idx[alive_idx]
            final, died = _resolve_barriers_vec(
                current, proposed, mort, defl, trans, water_nbrs, rng
            )
            pool.tri_idx[alive_idx] = final
            pool.alive[alive_idx[died]] = False

    _apply_current_advection_vec(pool, mesh, fields, alive, rng)


@njit(cache=True, parallel=True)
def _step_random_numba(tri_indices, water_nbrs, water_nbr_count, rand_vals, steps):
    n = len(tri_indices)
    for s in range(steps):
        for i in prange(n):
            c = tri_indices[i]
            cnt = water_nbr_count[c]
            if cnt > 0:
                idx = int(rand_vals[s, i] * cnt)
                if idx >= cnt:
                    idx = cnt - 1
                tri_indices[i] = water_nbrs[c, idx]


@njit(cache=True, parallel=True)
def _step_directed_numba(
    tri_indices, water_nbrs, water_nbr_count, field, rand_vals, steps, ascending
):
    n = len(tri_indices)
    for s in range(steps):
        for i in prange(n):
            c = tri_indices[i]
            cnt = water_nbr_count[c]
            if cnt <= 0:
                continue
            if s % 2 == 0:
                best_nbr = water_nbrs[c, 0]
                best_val = field[best_nbr]
                for k in range(1, cnt):
                    nbr = water_nbrs[c, k]
                    val = field[nbr]
                    if ascending:
                        if val > best_val:
                            best_val = val
                            best_nbr = nbr
                    else:
                        if val < best_val:
                            best_val = val
                            best_nbr = nbr
                tri_indices[i] = best_nbr
            else:
                idx = int(rand_vals[s, i] * cnt)
                if idx >= cnt:
                    idx = cnt - 1
                tri_indices[i] = water_nbrs[c, idx]


@njit(cache=True, parallel=True)
def _step_to_cwr_numba(
    tri_indices, water_nbrs, water_nbr_count, temperature, steps, cwr_threshold
):
    n = len(tri_indices)
    for s in range(steps):
        for i in prange(n):
            c = tri_indices[i]
            if temperature[c] < cwr_threshold:
                continue
            cnt = water_nbr_count[c]
            if cnt <= 0:
                continue
            best_nbr = water_nbrs[c, 0]
            best_temp = temperature[best_nbr]
            for k in range(1, cnt):
                nbr = water_nbrs[c, k]
                t = temperature[nbr]
                if t < best_temp:
                    best_temp = t
                    best_nbr = nbr
            tri_indices[i] = best_nbr


@njit(cache=True, parallel=True)
def _advection_numba(
    tri_indices,
    water_nbrs,
    water_nbr_count,
    centroids,
    u,
    v,
    speeds,
    rand_drift,
    scale_x,
    scale_y,
    speed_threshold=0.01,
):
    """Nearest-neighbour advection in flow direction.

    ``scale_x`` / ``scale_y`` convert centroid diffs from whatever units
    the mesh uses (degrees for TriMesh/H3Mesh, meters for HexMesh) into
    meters before the dot-product with the flow vector. See
    ``Mesh.metric_scale(lat)`` — HexMesh returns (1.0, 1.0).
    """
    n = len(tri_indices)
    for i in prange(n):
        if speeds[i] < speed_threshold:
            continue
        c = tri_indices[i]
        cnt = water_nbr_count[c]
        if cnt <= 0:
            continue
        flow_norm = (u[c] ** 2 + v[c] ** 2) ** 0.5 + 1e-12
        flow_x = u[c] / flow_norm
        flow_y = v[c] / flow_norm
        best_dot = -999.0
        best_nbr = c
        cx = centroids[c, 1]
        cy = centroids[c, 0]
        for k in range(cnt):
            nbr = water_nbrs[c, k]
            dx = (centroids[nbr, 1] - cx) * scale_x
            dy = (centroids[nbr, 0] - cy) * scale_y
            dnorm = (dx**2 + dy**2) ** 0.5 + 1e-12
            dot = (dx / dnorm) * flow_x + (dy / dnorm) * flow_y
            if dot > best_dot:
                best_dot = dot
                best_nbr = nbr
        drift_prob = min(speeds[i] * 5.0, 0.8)
        if rand_drift[i] < drift_prob:
            tri_indices[i] = best_nbr


def _step_random_vec(tri_indices, water_nbrs, water_nbr_count, rng, steps):
    """Vectorized random walk for a batch of agents."""
    if _use_numba():
        rand_vals = rng.random((steps, len(tri_indices)))
        _step_random_numba(tri_indices, water_nbrs, water_nbr_count, rand_vals, steps)
    else:
        # Original NumPy implementation
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


def _step_directed_vec(
    tri_indices, water_nbrs, water_nbr_count, field, rng, steps, ascending
):
    """Vectorized gradient-following for a batch of agents.

    Even micro-steps: move to neighbor with best field value.
    Odd micro-steps: random jitter (move to random neighbor).
    """
    if _use_numba():
        rand_vals = rng.random((steps, len(tri_indices)))
        _step_directed_numba(
            tri_indices, water_nbrs, water_nbr_count, field, rand_vals, steps, ascending
        )
    else:
        # Original NumPy implementation
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


def _step_to_cwr_vec(
    tri_indices, water_nbrs, water_nbr_count, temperature, steps, cwr_threshold
):
    """Vectorized cold-water refuge seeking for a batch of agents."""
    if _use_numba():
        _step_to_cwr_numba(
            tri_indices, water_nbrs, water_nbr_count, temperature, steps, cwr_threshold
        )
    else:
        # Original NumPy implementation
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


def _apply_current_advection_vec(pool, mesh, fields, alive_mask, rng):
    """Vectorized current advection for all alive agents."""
    u = fields.get("u_current")
    v = fields.get("v_current")
    if u is None or v is None:
        return

    if _use_numba():
        alive_idx = np.where(alive_mask)[0]
        if len(alive_idx) == 0:
            return
        tris = pool.tri_idx[alive_idx].copy()
        speeds = np.sqrt(u[tris] ** 2 + v[tris] ** 2)
        rand_drift = rng.random(len(tris))
        # Derive metric scale from the mesh — converts centroid diffs to
        # meters regardless of whether centroids are in degrees (TriMesh/H3)
        # or meters (HexMesh).  See Mesh.metric_scale.
        lat_mean = float(mesh.centroids[:, 0].mean())
        scale_x, scale_y = mesh.metric_scale(lat_mean)
        _advection_numba(
            tris,
            mesh._water_nbrs,
            mesh._water_nbr_count,
            mesh.centroids_c,
            u,
            v,
            speeds,
            rand_drift,
            scale_x,
            scale_y,
        )
        pool.tri_idx[alive_idx] = tris
    else:
        # Original NumPy implementation
        alive_idx = np.where(alive_mask)[0]
        if len(alive_idx) == 0:
            return

        tris = pool.tri_idx[alive_idx]
        speeds = np.sqrt(u[tris] ** 2 + v[tris] ** 2)

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

        c0 = mesh.centroids[mov_tris]  # (n, 2)
        cn = mesh.centroids[safe_idx]  # (n, max_nbrs, 2)
        dy = cn[:, :, 0] - c0[:, np.newaxis, 0]  # (n, max_nbrs)
        dx = cn[:, :, 1] - c0[:, np.newaxis, 1]  # (n, max_nbrs)
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


@njit(cache=True, parallel=True)
def _resolve_barriers_numba(
    current, proposed, barrier_mort, barrier_defl, neighbors, rand_vals
):
    """Numba kernel for barrier resolution."""
    n = len(current)
    final = proposed.copy()
    died = np.zeros(n, dtype=np.bool_)
    max_nbrs = neighbors.shape[1]
    for i in prange(n):
        if current[i] == proposed[i]:
            continue
        slot = -1
        for k in range(max_nbrs):
            if neighbors[current[i], k] == proposed[i]:
                slot = k
                break
        if slot < 0:
            continue
        p_mort = barrier_mort[current[i], slot]
        p_defl = barrier_defl[current[i], slot]
        if p_mort <= 0.0 and p_defl <= 0.0:
            continue
        r = rand_vals[i]
        if r < p_mort:
            died[i] = True
        elif r < p_mort + p_defl:
            final[i] = current[i]
    return final, died


def _resolve_barriers_vec(
    current, proposed, barrier_mort, barrier_defl, barrier_trans, neighbors, rng
):
    """Resolve barrier outcomes for a batch of proposed moves."""
    n = len(current)
    if n == 0:
        return proposed.copy(), np.zeros(n, dtype=bool)

    if _use_numba():
        rand_vals = rng.random(n)
        return _resolve_barriers_numba(
            current, proposed, barrier_mort, barrier_defl, neighbors, rand_vals
        )

    # Original NumPy implementation
    final = proposed.copy()
    died = np.zeros(n, dtype=bool)
    moving = current != proposed
    if not moving.any():
        return final, died
    nbr_matrix = neighbors[current]
    match = nbr_matrix == proposed[:, np.newaxis]
    has_match = match.any(axis=1) & moving
    if not has_match.any():
        return final, died
    slot = np.argmax(match, axis=1)
    p_mort = barrier_mort[current, slot]
    p_defl = barrier_defl[current, slot]
    has_barrier = has_match & ((p_mort > 0) | (p_defl > 0))
    if not has_barrier.any():
        return final, died
    rolls = rng.random(n)
    kill = has_barrier & (rolls < p_mort)
    died[kill] = True
    deflect = has_barrier & ~kill & (rolls < p_mort + p_defl)
    final[deflect] = current[deflect]
    return final, died
