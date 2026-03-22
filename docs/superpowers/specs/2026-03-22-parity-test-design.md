# Parity Test Skill Design Spec

**Date**: 2026-03-22
**Status**: Approved (6 review iterations)
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
  --hexsim-exe "HexSim 4.0.20/HexSimEngine64.exe" \
  --output docs/parity-report.md
```

**Note**: `--steps` controls both engines. The XML patcher sets `<timesteps>` to the specified value so HexSim 4.0.20 and HexSimPy run for the same duration.

#### Functions

| Function | Purpose |
|----------|---------|
| `patch_scenario_xml(src, dst, workspace, start_log_step, n_steps)` | Replace hardcoded absolute paths (`F:\Marcia\...`) with local workspace, set `startLogStep`, set `<timesteps>` to `n_steps`. Data Probe events are already in the XML and only need their paths fixed. |
| `run_hexsim_engine(patched_xml, seed, timeout)` | Run `HexSimEngine64.exe -r <seed> <xml>` via subprocess with timeout. The `-r` flag sets the simulation seed (verified: `HexSimEngine [-r simulationSeed] scenario.xml`). Returns `(elapsed_s, census_dict, probe_dict)`. Handles `TimeoutExpired` gracefully by collecting partial results. |
| `run_hexsimpy(workspace, scenario_xml, seed, n_steps, probe_interval)` | Run `ScenarioLoader.load()` then a **manual `sim.step()` loop** (not `sim.run()` — the loop is needed to collect per-population census and periodic agent snapshots). Returns `(elapsed_s, census_dict, agent_snapshots)`. |
| `parse_hexsim_census(result_dir)` | Parse HexSim `*.csv` census files into `{pop_id: {step: {size, traits, lambda}}}` |
| `parse_hexsim_data_probe(result_dir, accumulator_map)` | Parse Data Probe CSVs using per-population `accumulator_map` (see Section 5) into `{pop_id: {step: {agent_id: {named_values...}}}}` |
| `build_accumulator_map(scenario_xml)` | Parse the scenario XML to extract accumulator names **per population** in definition order, returning `{pop_index: {column_index: accumulator_name}}`. Each population has its own accumulator set. |
| `compare_census(hexsim_census, hexsimpy_census)` | Step-by-step comparison of population size and trait distributions, matched by step AND population. Returns list of `Divergence(step, pop, metric, hexsim_val, hexsimpy_val, rel_error)` |
| `compare_agents(hexsim_probe, hexsimpy_snapshots, tolerance)` | Compare per-agent accumulator values at sampled timesteps, matched by step AND population name. Returns divergence stats |
| `generate_report(comparisons, timing, output_path)` | Write markdown report with pass/fail verdict, tables, and divergence details |

#### HexSimPy step loop (not sim.run)

```python
loader = ScenarioLoader()
sim = loader.load(workspace_dir, scenario_xml, rng_seed=seed)
mesh = sim.landscape["mesh"]
history = []
snapshots = []
for step in range(n_steps):
    sim.step()
    # Collect per-population census (sim.history only has aggregate n_alive)
    record = {"step": step}
    for pname, pop in sim.populations.populations.items():
        record[pname] = {"n_alive": pop.n_alive}
        if pop.trait_mgr:
            for tname, tdef in pop.trait_mgr.definitions.items():
                vals = pop.trait_mgr.get(tname)
                record[pname][f"trait_{tname}"] = np.bincount(vals[pop.alive])
    history.append(record)
    # Periodic agent snapshot
    if step % probe_interval == 0:
        for pname, pop in sim.populations.populations.items():
            pool = pop.pool
            alive = pop.alive
            if alive.sum() == 0:
                continue
            snapshots.append({
                "step": step, "pop_name": pname, "n_alive": pop.n_alive,
                "ed_kJ_g_mean": np.mean(pool.ed_kJ_g[alive]),
                "ed_kJ_g_std": np.std(pool.ed_kJ_g[alive]),
                "mass_g_mean": np.mean(pool.mass_g[alive]),
                "mass_g_std": np.std(pool.mass_g[alive]),
                "position_histogram": np.bincount(pool.tri_idx[alive], minlength=mesh.n_cells),
            })
```

#### Data flow

```
1. patch_scenario_xml()
   ├── Read XML, string-replace F:\Marcia\... → local workspace path
   ├── Set startLogStep = 1
   ├── Set <timesteps> to --steps value
   └── Write patched XML

2. Clean Results directory (shutil.rmtree if exists)

3. run_hexsim_engine() [first — sequential]
   ├── subprocess.run(HexSimEngine64.exe -r seed patched.xml, timeout=timeout)
   │   ├── On success: parse census + data probe
   │   ├── On TimeoutExpired: collect partial results from CSVs written so far
   │   └── On engine not found: return None (SKIP HexSim comparison)
   └── Output: Results/<name>/<name>.N.csv + Data Probe/*.csv

4. run_hexsimpy() [second]
   ├── ScenarioLoader.load() → manual sim.step() loop
   ├── Collect per-population census each step
   └── Collect agent snapshots every probe_interval steps

5. build_accumulator_map(scenario_xml)  [runs once, before comparisons — only needs XML]
   └── Map Data Probe CSV column indices to accumulator names

6. parse_hexsim_census() + parse_hexsim_data_probe(accumulator_map)
   └── Parse HexSim CSV outputs into structured dicts
   Note: HexSimPy census/snapshots are already structured from the step loop — no parsing needed

7. compare_census() + compare_agents()
   ├── Census: statistical tolerance on population size (see Section 3.1)
   └── Agents: relative tolerance on energy, mass; JSD on positions

8. generate_report()
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

### 3.1 Census trajectory (statistical match)

Both engines use different RNG implementations (C++ Mersenne Twister vs NumPy PCG64). Since movement, behavior selection, and agent initialization involve stochastic draws, per-step population sizes may diverge slightly due to different random sequences — even with the same seed value. Therefore, census comparison uses a tolerance band rather than exact match.

| Metric | HexSim source | HexSimPy source | Tolerance |
|--------|--------------|-----------------|-----------|
| Population size | Census CSV `Population Size` | `pop.n_alive` | < 5% relative OR < 10 agents |
| Trait distribution | Census CSV `Trait Index N` | `TraitManager.get()` counts | < 10% relative per category |
| Lambda (growth rate) | Census CSV `Lambda` | Computed from consecutive sizes | < 0.05 |

**Rationale for non-exact match**: The HexSim `-r` flag sets a simulation seed, but the C++ engine uses a different PRNG algorithm than Python's `numpy.random.default_rng()`. Even with seed=42 in both, the random sequences differ. This is expected and acceptable — parity means the *dynamics are qualitatively the same*, not that every agent makes the same stochastic choice.

### 3.2 Per-agent data probe (statistical match)

| Metric | HexSim source | HexSimPy source | Tolerance |
|--------|--------------|-----------------|-----------|
| Mean energy density (kJ/g) | Data Probe `Energy Density` column | `np.mean(pool.ed_kJ_g[alive])` | < 5% relative |
| Mean mass (g) | Data Probe `Mass` column | `np.mean(pool.mass_g[alive])` | < 5% relative |
| Energy density std dev | Data Probe | `np.std(pool.ed_kJ_g[alive])` | < 10% relative |
| Mass std dev | Data Probe | `np.std(pool.mass_g[alive])` | < 10% relative |
| Spatial distribution | Data Probe cell IDs | `pool.tri_idx` histogram (occupied cells only) | Jensen-Shannon divergence < 0.3 |

**Accumulator-to-column mapping**: Discovered programmatically by `build_accumulator_map()` which parses `<accumulator name="...">` elements from the scenario XML in definition order. Both engines use the same hex cell indexing (compact water-cell IDs from `HexMesh`), so cell indices are directly comparable.

**Spatial comparison**: Uses Jensen-Shannon divergence on occupied-cell histograms instead of chi-squared, because with ~2000 agents on ~40K water cells most bins are zero. JSD is symmetric, bounded [0, 1], and handles sparse distributions correctly. Unlike Wasserstein/EMD, JSD does not require a meaningful distance metric between cell IDs (compact hex indices have no inherent spatial ordering — cell 100 is not necessarily near cell 101 in physical space). Threshold: JSD < 0.3 (0 = identical, 1 = maximally different).

### 3.3 Verdict logic

```python
def verdict(census_divergences, agent_divergences):
    # Census: population size must be within tolerance band
    census_fails = [d for d in census_divergences
                    if d.metric == "population_size"
                    and d.rel_error > 0.05 and abs(d.hexsim_val - d.hexsimpy_val) > 10]
    if census_fails:
        return "FAIL"
    # Agent metrics
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
3. **Set `<timesteps>`** to the `--steps` value so both engines run the same duration
4. **Preserve Data Probe events** — these are already defined in the XML as `data_probe` events; they only need the workspace path fixed (no `enable_data_probe` parameter needed)

This reuses the proven approach from `scripts/benchmark_columbia_steelhead.py`.

---

## 5. HexSim Data Probe Output Format

HexSim writes Data Probe CSVs to `Results/<scenario>/Data Probe/`. Each file contains per-agent accumulator snapshots:

```csv
Run,Time Step,Population,Individual,Accumulator 1,Accumulator 2,...
0,100,0,1,5.23,3500,...
0,100,0,2,5.11,3420,...
```

The columns correspond to the accumulators defined in the scenario XML **in definition order**. The mapping is discovered programmatically:

```python
def build_accumulator_map(scenario_xml: Path) -> dict[int, dict[int, str]]:
    """Parse XML to map column indices to accumulator names, per population.

    Returns {pop_index: {col_index: accumulator_name}}.
    Each population has its own accumulator set — the Data Probe CSV
    'Population' column identifies which mapping to use for each row.
    """
    tree = ET.parse(scenario_xml)
    result = {}
    for pop_idx, pop in enumerate(tree.findall(".//population")):
        acc_names = [acc.get("name") for acc in pop.findall(".//accumulator")]
        result[pop_idx] = {i: name for i, name in enumerate(acc_names)}
    return result
```

Key accumulator names for comparison (from `snake_Columbia2017B.xml`):
- `Energy Density` → energy density (kJ/g equivalent)
- `Mass` → body mass (g)
- `Mean Temperature [ 3hr ]` → 3-hour mean temperature
- `Fish Location [ hydropower ]` → hydropower zone flag

---

## 6. Probe Interval for HexSimPy

To avoid excessive memory use, HexSimPy snapshots are collected every `probe_interval` steps (default: 100). At each probe step, the script records:

```python
snapshot = {
    "step": step,
    "pop_name": pname,
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
| Step | Pop | HexSim Size | HexSimPy Size | Rel Error | Status |
|------|-----|-------------|---------------|-----------|--------|

## Agent Comparison (sampled steps)
| Step | Pop | Metric | HexSim | HexSimPy | Rel Error | Status |
|------|-----|--------|--------|----------|-----------|--------|

## Divergences
(details of any WARN/FAIL items)

## Notes
- RNG implementations differ (C++ MT vs Python PCG64) — stochastic divergence expected
- Census tolerance: < 5% relative OR < 10 agents absolute
- Agent tolerance: < 5% relative for means, < 10% for std devs
- Spatial: Jensen-Shannon divergence on occupied cells
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

| File | Purpose | Git tracked |
|------|---------|-------------|
| `scripts/parity_test.py` | Core comparison engine with CLI | Yes |
| `~/.claude/skills/parity-test/SKILL.md` | Claude Code skill definition | No (personal) |
| `docs/parity-report.md` | Generated report | No (add to `.gitignore`) |

---

## 10. Error Handling

| Condition | Behavior |
|-----------|----------|
| HexSim engine not found | Print `[SKIP] HexSim engine not found`, run HexSimPy only, report as "HexSimPy-only run" |
| HexSim times out | Collect partial census from CSVs written so far, compare available steps |
| HexSim crashes (non-zero exit) | Report error + stderr, compare any partial output |
| No Data Probe output | Skip agent comparison, report census-only verdict |
| Scenario XML not found | Exit with error message |
| Stale Results directory | `shutil.rmtree` before each run to prevent parsing old output |
| Cell ID mismatch | Sanity check: `assert max_cell_id_from_probe <= mesh.n_cells`. If HexSim uses full-grid IDs while HexSimPy uses compact IDs, spatial comparison is invalid — report error and skip spatial metrics |

---

## 11. Constraints

- HexSim 4.0.20 engine is Windows-only (`HexSimEngine64.exe`)
- HexSim CLI: `HexSimEngine [-d] [-t [-m port -c port]] [-s schema] [-r simulationSeed] scenario.xml` — the `-r` flag sets the simulation seed (verified from engine usage output)
- Subprocess timeout default: 3600s (1 hour) — sufficient for ~1400 steps on 16M-cell grid
- HexSim census logging must be enabled via XML patch (startLogStep=1)
- Data Probe output depends on the scenario having `data_probe` events defined in the XML
- RNG implementations differ between C++ (Mersenne Twister) and Python (PCG64) — per-agent positions will diverge stochastically; comparison uses statistical tolerances, not exact match
- Both engines use the same hex cell compact indexing (water-only cells from `HexMesh.from_hexsim()`), so cell IDs are directly comparable between HexSim Data Probe output and HexSimPy `pool.tri_idx`
