# MD Toolkit

An end-to-end toolkit for running molecular dynamics simulations and analyzing
the resulting trajectories. It pairs an OpenMM-based simulation runner with a
suite of trajectory analyses, producing publication-quality figures through a
single, consistent plotting layer.

## Features

The analysis suite (`analyses.py`) covers the full set of standard MD metrics:

- **Structural metrics** — RMSD, RMSF, and radius of gyration (Rg) via
  `compute_rmsd_rmsf_rg`
- **Dimensionality reduction** — principal component analysis via `compute_pca`
- **Free-energy landscape (FEL)** — projected onto the top principal components
  via `plot_pca_fel`
- **Secondary structure** — per-residue DSSP assignment via `compute_dssp`
- **Hydrogen bonds** — count and occupancy over time via `compute_hbonds`
- **Solvent exposure** — solvent-accessible surface area via `compute_sasa`
- **Residue contacts** — contact maps via `compute_contact_map`

Each `compute_*` function has a matching `plot_*` function, so analysis and
visualization stay cleanly separated.

## Usage

The simulation runner reads a YAML config and drives the full pipeline:
```bash
python simulate.py --config config/config.yaml

Under the hood, `simulate.py` loads the config with `load_config`, then hands
it to `run(cfg)`, which builds the system, runs the MD, and writes the
trajectory.


This reflects the real entry point in `simulate.py` (the `main()` parser takes a required `--config` flag and calls `run(cfg)`) and the actual function names in `analyses.py`.

Want me to add the **Installation** and **Configuration** sections next, or pull the exact function signatures so the API list shows the parameters too?