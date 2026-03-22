# Parity Test Skill Design Spec

**Date**: 2026-03-22
**Status**: Draft
**Goal**: Create a Claude Code skill (`/parity-test`) and backing Python script that runs a HexSim XML scenario on both HexSim 4.0.20 (C++) and HexSimPy (Python), then compares census trajectories and per-agent data probe outputs to verify functional parity.

---

## 1. Motivation

HexSimPy reimplements the Snyder et al. (2019) Columbia River migration corridor model. To verify correctness, we need automated head-to-head comparison against the original EPA HexSim 4.0.20 engine using the same scenario XML, spatial data, and RNG seed. Manual benchmark runs (done 2026-03-22) showed matching population sizes over 1,417 shared steps, but a repeatable, scriptable parity test is needed for regression testing and validation of new features.

---

## 2. Components

### 2.1 `scripts/parity_test.py` — Core comparison engine

A standalone Python script with CLI interface:

```
python scripts/parity_test.py \
  --workspace "Columbia [small]" \
  --scenario "Scenarios/snake_Columbia2017B.xml" \
  --steps 2928 \
  --seed 42 \
  --timeout 3600 \
  --output docs/parity-report.md
```

#### Functions

| Function | Purpose |
|----------|---------|
| `patch_scenario_xml(src, dst, workspace, start_log_step, enable_data_probe)` | Replace hardcoded absolute paths (`F:\Marcia\...`) with local workspace, set `startLogStep=1`, enable Data Probe logging |
| `run_hexsim_engine(patched_xml, seed, timeout)` | Run `HexSimEngine64.exe` via subprocess, return `(elapsed_s, census_dict, probe_dict)` |
| `run_hexsimpy(workspace, scenario_xml, seed, n_steps, probe_interval)` | Run `ScenarioLoader.load()` + `sim.run()`, collect per-step census and periodic agent snapshots, return `(elapsed_s, census_dict, agent_snapshots)` |
| `parse_hexsim_census(result_dir)` | Parse HexSim `*.csv` census files into `{pop_id: {step: {size, traits, lambda}}}` |
| `parse_hexsim_data_probe(result_dir)` | Parse Data Probe CSVs into `{step: {agent_id: {accumulators...}}}` |
| `compare_census(hexsim_census, hexsimpy_census)` | Step-by-step comparison of population size and trait distributions. Returns list of `Divergence(step, pop, metric, hexsim_val, hexsimpy_val, rel_error)` |
| `compare_agents(hexsim_probe, hexsimpy_snapshots, tolerance)` | Compare per-agent accumulator values at sampled timesteps. Returns divergence stats |
| `generate_report(comparisons, timing, output_path)` | Write markdown report with pass/fail verdict, tables, and divergence details |

#### Data flow

```
1. patch_scenario_xml()
   ├── Read XML, replace F:\Marcia\... → local workspace path
   ├── Set startLogStep = 1
   ├── Enable Data Probe output (if available in XML)
   └── Write patched XML

2. run_hexsim_engine() + run_hexsimpy()  [sequential — HexSim first]
   ├── HexSim: subprocess.run(HexSimEngine64.exe -r seed patched.xml)
   │   ├── Output: Results/<name>/<name>.N.csv (census per population)
   │   └── Output: Results/<name>/Data Probe/*.csv (per-agent snapshots)
   └── HexSimPy: ScenarioLoader.load() → sim.run()
       ├── Output: census dict from sim.history
       └── Output: agent snapshots every probe_interval steps

3. parse_hexsim_census() + parse_hexsim_data_probe()
   └── Parse HexSim CSV outputs into structured dicts

4. compare_census() + compare_agents()
   ├── Census: exact match on population size, trait counts per step
   └── Agents: relative tolerance on energy, mass; distribution on positions

5. generate_report()
   └── Markdown with verdict, timing table, divergence table, notes
```

### 2.2 `~/.claude/skills/parity-test/SKILL.md` — Claude Code skill

A personal skill registered at `~/.claude/skills/parity-test/SKILL.md` that teaches Claude how to invoke the parity test. The skill:

- Describes when to use it (after modifying event logic, scenario loader, movement kernels, or bioenergetics)
- Provides the CLI invocation pattern
- Explains how to interpret results (PASS/WARN/FAIL)
- Lists common failure modes and their causes

---

## 3. Comparison Metrics

### 3.1 Census trajectory (exact match expected)

| Metric | HexSim source | HexSimPy source | Tolerance |
|--------|--------------|-----------------|-----------|
| Population size | Census CSV `Population Size` | `pop.n_alive` | Exact (0) |
| Trait distribution | Census CSV `Trait Index N` | `TraitManager.get()` counts | Exact (0) |
| Lambda (growth rate) | Census CSV `Lambda` | Computed from consecutive sizes | < 0.01 |

### 3.2 Per-agent data probe (statistical match)

| Metric | HexSim source | HexSimPy source | Tolerance |
|--------|--------------|-----------------|-----------|
| Mean energy density (kJ/g) | Data Probe CSV | `np.mean(pool.ed_kJ_g[alive])` | < 5% relative |
| Mean mass (g) | Data Probe CSV | `np.mean(pool.mass_g[alive])` | < 5% relative |
| Energy density std dev | Data Probe CSV | `np.std(pool.ed_kJ_g[alive])` | < 10% relative |
| Spatial distribution | Data Probe cell IDs | `pool.tri_idx` histogram | Chi-squared p > 0.05 |

### 3.3 Verdict logic

```python
def verdict(census_divergences, agent_divergences):
    if any(d.metric == "population_size" and d.rel_error > 0 for d in census_divergences):
        return "FAIL"  # census must match exactly
    if any(d.rel_error > 0.15 for d in agent_divergences):
        return "FAIL"  # agent metrics > 15% divergence
    if any(d.rel_error > 0.05 for d in agent_divergences):
        return "WARN"  # agent metrics 5-15% divergence
    return "PASS"
```

---

## 4. XML Patching Strategy

The scenario XML files embed absolute paths from the original machine (`F:\Marcia\Columbia [small]\...`). The patcher must:

1. **String-replace** all occurrences of the original workspace prefix with the local workspace path
2. **Set `startLogStep`** to 1 (default is 2929 which disables logging)
3. **Preserve Data Probe events** — these are already defined in the XML as `data_probe` events; they just need the workspace path fixed

This reuses the proven approach from `scripts/benchmark_columbia_steelhead.py`.

---

## 5. HexSim Data Probe Output Format

HexSim writes Data Probe CSVs to `Results/<scenario>/Data Probe/`. Each file contains per-agent accumulator snapshots:

```csv
Run,Time Step,Population,Individual,Accumulator 1,Accumulator 2,...
0,100,0,1,5.23,3500,...
0,100,0,2,5.11,3420,...
```

The columns correspond to the accumulators defined in the scenario XML, in order.

---

## 6. Probe Interval for HexSimPy

To avoid excessive memory use, HexSimPy snapshots are collected every `probe_interval` steps (default: 100). At each probe step, the script records:

```python
snapshot = {
    "step": t,
    "pop_name": name,
    "n_alive": pop.n_alive,
    "ed_kJ_g_mean": np.mean(pool.ed_kJ_g[alive]),
    "ed_kJ_g_std": np.std(pool.ed_kJ_g[alive]),
    "mass_g_mean": np.mean(pool.mass_g[alive]),
    "mass_g_std": np.std(pool.mass_g[alive]),
    "position_histogram": np.bincount(pool.tri_idx[alive], minlength=mesh.n_cells),
}
```

---

## 7. Report Format

Output is a markdown file:

```markdown
# Parity Test Report

**Scenario**: snake_Columbia2017B.xml
**Steps**: 2928 | **Seed**: 42 | **Date**: 2026-03-22
**Verdict**: PASS / WARN / FAIL

## Timing
| Engine | Load | Run | Total |
|--------|------|-----|-------|

## Census Comparison
| Step | Pop | HexSim Size | HexSimPy Size | Match |
|------|-----|-------------|---------------|-------|

## Agent Comparison (sampled steps)
| Step | Pop | Metric | HexSim | HexSimPy | Rel Error | Status |
|------|-----|--------|--------|----------|-----------|--------|

## Divergences
(details of any WARN/FAIL items)

## Notes
- Known stochastic differences due to RNG implementation
- ...
```

---

## 8. Skill File

`~/.claude/skills/parity-test/SKILL.md`:

```yaml
---
name: parity-test
description: Use when modifying salmon IBM event logic, scenario loader, movement kernels, or bioenergetics — runs same HexSim scenario on both C++ engine and HexSimPy to verify output parity
---
```

The skill body will contain:
- When to use (after code changes to core simulation)
- CLI invocation
- How to interpret PASS/WARN/FAIL
- Common failure causes and fixes
- Quick reference for adding new comparison metrics

---

## 9. File Inventory

| File | Purpose |
|------|---------|
| `scripts/parity_test.py` | Core comparison engine with CLI |
| `~/.claude/skills/parity-test/SKILL.md` | Claude Code skill definition |
| `docs/parity-report.md` | Generated report (not committed) |

---

## 10. Constraints

- HexSim 4.0.20 engine is Windows-only (`HexSimEngine64.exe`)
- Subprocess timeout default: 3600s (1 hour) — sufficient for ~1400 steps on 16M-cell grid
- HexSim census logging must be enabled via XML patch (startLogStep=1)
- Data Probe output depends on the scenario having `data_probe` events defined
- RNG implementations differ between C++ and Python — per-agent positions will diverge stochastically; comparison uses statistical tests not exact match
