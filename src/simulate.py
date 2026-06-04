"""
simulate.py
-----------
Run an OpenMM MD simulation from a YAML config file.

Usage
-----
    python src/simulate.py --config config/nfkb.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fix_and_solvate(cfg: dict) -> tuple:
    """PDBFixer pass → returns (modeller, topology, positions)."""
    from pdbfixer import PDBFixer
    import openmm.app as app

    pdb_path  = Path(cfg["project"]["data_dir"]) / cfg["input"]["pdb_file"]
    fixed_pdb = Path(cfg["project"]["output_dir"]) / (
        cfg["project"]["protein"] + "_complex_fixed.pdb"
    )
    Path(cfg["project"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"[fix] Loading {pdb_path}")
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    with open(fixed_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    print(f"[fix] Fixed PDB saved → {fixed_pdb}")

    return fixer.topology, fixer.positions, fixed_pdb


def build_system(cfg: dict, topology, positions, fixed_pdb: Path):
    """Build solvated system and return (system, modeller)."""
    import openmm.app as app
    import openmm.unit as unit

    sc   = cfg["solvation"]
    sys_ = cfg["system"]
    ens  = cfg["ensemble"]
    ff_c = cfg["forcefield"]

    ff = app.ForceField(ff_c["protein"], ff_c["water"])

    pdb      = app.PDBFile(str(fixed_pdb))
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # strip and re-add hydrogens at physiological pH
    modeller.deleteWater()
    to_delete = [
        a for a in modeller.topology.atoms()
        if a.element is not None and a.element.symbol == "H"
    ]
    modeller.delete(to_delete)
    modeller.addHydrogens(ff, pH=sc["ph"])

    # solvate if no periodic box
    if modeller.topology.getPeriodicBoxVectors() is None:
        print("[build] No periodic box found — adding solvent + ions …")
        modeller.addSolvent(
            ff,
            model=ff_c["water_model"],
            padding=sc["padding_nm"] * unit.nanometer,
            ionicStrength=sc["ionic_strength_molar"] * unit.molar,
        )

    solvated_path = (
        Path(cfg["project"]["output_dir"])
        / f"{cfg['project']['protein']}_complex_solvated.pdb"
    )
    with open(solvated_path, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print(f"[build] Solvated structure saved → {solvated_path}")

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=sys_["nonbonded_cutoff_nm"] * unit.nanometer,
        constraints=getattr(app, sys_["constraints"]),
        hydrogenMass=sys_["hydrogen_mass_amu"] * unit.amu,
    )

    import openmm as mm
    system.addForce(
        mm.MonteCarloBarostat(
            ens["pressure_bar"] * unit.bar,
            ens["temperature_K"] * unit.kelvin,
        )
    )

    return system, modeller


def run(cfg: dict) -> None:
    import openmm.app as app
    import openmm as mm
    import openmm.unit as unit

    ens   = cfg["ensemble"]
    run_c = cfg["run"]
    out   = Path(cfg["project"]["output_dir"])

    topology, positions, fixed_pdb = fix_and_solvate(cfg)
    system, modeller               = build_system(cfg, topology, positions, fixed_pdb)

    integrator = mm.LangevinMiddleIntegrator(
        ens["temperature_K"] * unit.kelvin,
        ens["friction_per_ps"] / unit.picosecond,
        ens["timestep_ps"] * unit.picoseconds,
    )

    # --- Platform selection ---
    simulation = None
    for platform_name in run_c["platform_priority"]:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            props    = {}
            if platform_name in ("CUDA", "OpenCL"):
                key        = f"{platform_name}Precision"
                props[key] = run_c["precision"]
            simulation = app.Simulation(
                modeller.topology, system, integrator, platform, props
            )
            print(f"[sim] Using platform: {platform_name}")
            break
        except Exception as e:
            print(f"[sim] {platform_name} unavailable ({e}), trying next …")

    if simulation is None:
        raise RuntimeError("No suitable OpenMM platform found.")

    simulation.context.setPositions(modeller.positions)

    # --- Minimise ---
    print("[sim] Minimising energy …")
    simulation.minimizeEnergy()

    # --- Equilibration ---
    print(f"[sim] Equilibrating ({run_c['equilibration_steps']} steps) …")
    simulation.context.setVelocitiesToTemperature(
        ens["temperature_K"] * unit.kelvin
    )
    simulation.step(run_c["equilibration_steps"])

    # --- Reporters ---
    ri  = run_c["report_interval"]
    ci  = run_c["checkpoint_interval"]
    tot = run_c["production_steps"]

    simulation.reporters.append(
        app.DCDReporter(str(out / "trajectory.dcd"), ri)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout, ri,
            step=True, potentialEnergy=True, temperature=True,
            density=True, progress=True, totalSteps=tot,
            remainingTime=True, speed=True, separator="\t",
        )
    )
    simulation.reporters.append(
        app.CheckpointReporter(str(out / "checkpoint.chk"), ci)
    )

    # --- Production ---
    print(f"[sim] Production run: {tot} steps "
          f"({tot * ens['timestep_ps'] / 1000:.1f} ns) …")
    try:
        simulation.step(tot)
        print("[sim] Done.")
    except Exception as e:
        print(f"[sim] Simulation interrupted: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenMM MD simulation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
