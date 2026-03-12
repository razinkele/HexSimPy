"""Movement kernels on the triangular mesh."""
from __future__ import annotations

import numpy as np

from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.mesh import TriMesh


def execute_movement(pool, mesh, fields, seed=None, n_micro_steps=3, cwr_threshold=16.0):
    rng = np.random.default_rng(seed)
    alive_idx = np.where(pool.alive & ~pool.arrived)[0]

    for i in alive_idx:
        beh = pool.behavior[i]
        tri = pool.tri_idx[i]

        if beh == Behavior.HOLD:
            continue
        elif beh == Behavior.RANDOM:
            pool.tri_idx[i] = _step_random(tri, mesh, rng, n_micro_steps)
        elif beh == Behavior.UPSTREAM:
            pool.tri_idx[i] = _step_directed(tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=False)
        elif beh == Behavior.DOWNSTREAM:
            pool.tri_idx[i] = _step_directed(tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=True)
        elif beh == Behavior.TO_CWR:
            pool.tri_idx[i] = _step_to_cwr(tri, mesh, fields["temperature"], rng, n_micro_steps, cwr_threshold=cwr_threshold)

    _apply_current_advection(pool, mesh, fields, alive_idx, rng)


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
