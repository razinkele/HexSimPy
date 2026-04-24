# Curonian Lagoon Data Provenance

This document lists every data file under the Curonian Lagoon study area with its source, fetch date, license, and known limitations. **Required reading** for any publication using this model.

Corresponds to Phase 6 of `docs/superpowers/plans/2026-04-24-curonian-realism-upgrades.md`.

---

## Data files inventory

| File | Source | Product / URL | Fetched | License | Caveats |
|---|---|---|---|---|---|
| `data/curonian_minimal_grid.nc` | pre-existing stub | Unknown origin — predates this project arc | pre-2026 | Unknown | Depth variable source not documented; Phase 2 will regenerate `depth` from real EMODnet. |
| `data/forcing_cmems_phy_stub.nc` | stub | Generated at project bootstrap | pre-2026 | N/A | **Stub.** Phase 3 will replace with real CMEMS Baltic reanalysis (`BALTICSEA_MULTIYEAR_PHY_003_011`). |
| `data/winds_stub.nc` | stub | Generated at project bootstrap | pre-2026 | N/A | **Stub.** Currently unused by the simulation; wire into `seiche_pause()` is deferred to its own plan. |
| `data/nemunas_discharge.nc` | synthetic climatology | `scripts/fetch_nemunas_discharge.py` | 2026-04-24 | N/A (synthetic) | Gaussian peak (Apr 25, 1900 m³/s) + winter baseline (400 m³/s). **Same 365-day pattern repeats** every year for 2011-2024. Replace with Lithuanian EPA Smalininkai station records when access is granted. Calibrated to Valiuškevičius et al. 2019 and Mėžinė et al. 2019. |
| `configs/baltic_salmon_species.yaml` | literature | 8+ peer-reviewed citations inline | 2026-04-24 | Text only (no restriction) | Two citations flagged for verification: LW_a/LW_b provenance (Kallio-Nyberg 2020 is NOT a length-weight paper); cmax_A/B numerics need cross-check against Table 2 of Smith et al. 2009 PDF at hal.science/hal-00482198. |

## Pending data files (Phases 2-3 TODO)

| File | Expected source | Expected fetch | License | Disk estimate |
|---|---|---|---|---|
| `data/curonian_bathymetry.nc` | EMODnet Bathymetry WCS, `emodnet__mean` | `scripts/fetch_emodnet_bathymetry.py` | CC-BY 4.0 (EMODnet) | ~10 MB |
| `data/curonian_forcing_cmems.nc` | CMEMS Baltic `BALTICSEA_MULTIYEAR_PHY_003_011`, daily 2011-2024 | `scripts/fetch_cmems_forcing.py` | CC-BY 4.0 (Copernicus Marine) | 200-800 MB |
| `data/curonian_forcing_shyfem.nc` | SHYFEM (Idzelytė et al. 2023, doi:10.5194/os-19-1047-2023) | `scripts/fetch_shyfem_forcing.py` (stub — needs KU Marine Research Institute contact) | TBD | ~100-500 MB |

## Git strategy

CMEMS forcing (200-800 MB) should be `.gitignore`'d. Regeneration is reproducible via the fetcher scripts. Only the scripts, configs, and small derived products (bathymetry regridded to mesh) are committed.

Suggested `.gitignore` additions when Phase 3 lands:

```
data/curonian_forcing_cmems.nc
data/curonian_forcing_cmems_raw.nc
data/curonian_forcing_shyfem.nc
data/curonian_bathymetry_raw.tif
```

## Reproduction

To regenerate all data from scratch (when all phases have landed):

```bash
# Phase 2: bathymetry
micromamba run -n shiny python scripts/fetch_emodnet_bathymetry.py

# Phase 3: physics (CMEMS, may need account + credentials file)
micromamba run -n shiny python scripts/fetch_cmems_forcing.py

# Phase 3 fallback if CMEMS land-sea mask covers >50% of mesh
# micromamba run -n shiny python scripts/fetch_shyfem_forcing.py

# Phase 4: discharge (no external deps)
micromamba run -n shiny python scripts/fetch_nemunas_discharge.py
```

## Scientific provenance (Baltic species config)

The 8+ citations in `configs/baltic_salmon_species.yaml` include (DOIs cross-validated via scite MCP during the 2026-04-24 plan reviews):

- **Smith, Booker & Wells 2009** — `10.1016/j.marenvres.2008.12.010` (CMax allometric, verified paper exists)
- **Jensen, Jonsson & Forseth 2001** — `10.1046/j.0269-8463.2001.00572.x` (T_OPT peak, primary source)
- **Handeland, Imsland & Stefansson 2008** — `10.1016/j.aquaculture.2008.06.042` (T_AVOID behavioral threshold; v1 plan had the wrong DOI `.03.057` which points to a flounder genetics paper)
- **Elliott & Elliott 2010** + **Smialek, Pander & Geist 2021** `10.1111/fme.12507` (T_ACUTE_LETHAL acute thermal mortality)
- **Brett 1995** + **Breck 2008** `10.1577/t05-240.1` (lipid-first catabolism, ED_TISSUE=36)
- **Heinimaa & Heinimaa 2003** — Baltic Bothnian fecundity reference (1845 eggs/kg)
- **Koskela, Pirhonen & Jobling 1997** — `10.1111/j.1095-8649.1997.tb01976.x` (Baltic salmon low-temperature feeding; supporting evidence for T_OPT, not primary source)

## Known gaps not addressed by current data

See `docs/superpowers/plans/2026-04-24-curonian-realism-upgrades.md` Section "Deferred follow-up plans" and memory file `curonian_deferred.md` for the full list. Highlights:

- **Ice cover Dec-Mar** — no dataset in tree; Idzelytė et al. 2019 remote sensing climatology would be the reference.
- **Grey seal abundance** (for predation events) — HELCOM data, not yet integrated.
- **Round goby density** (egg predator, Rakauskas et al. 2013) — not integrated.
- **Cyanobacteria + summer hypoxia** — lagoon is heavily eutrophic (Mėžinė 2019); DO crashes not modelled.
- **Hatchery vs wild distinction** — most Nemunas salmon are hatchery-origin (Žeimena, Simnas programme since 1997); species config does not distinguish.
