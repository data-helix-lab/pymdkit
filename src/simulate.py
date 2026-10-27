"""Run a protein-ligand MD simulation with OpenMM and CHARMM-GUI/CGenFF data.

The script starts from CHARMM-GUI's ``step1_pdbreader.psf`` and
``step1_pdbreader.pdb``. It never sends the complex through PDBFixer, so the
bound ligand residue and its atom ordering cannot silently disappear or change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration is not a YAML mapping: {config_path}")
    root_setting = Path(cfg.get("project", {}).get("root_dir", ".."))
    project_root = root_setting if root_setting.is_absolute() else config_path.parent / root_setting
    cfg["_config_path"] = config_path
    cfg["_project_root"] = project_root.resolve()
    return cfg


def project_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (cfg["_project_root"] / path).resolve()


def data_path(cfg: dict[str, Any], value: str | Path) -> Path:
    base = project_path(cfg, cfg["project"]["data_dir"])
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def output_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg, cfg["project"]["output_dir"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(cfg: dict[str, Any], *keys: str) -> Any:
    value: Any = cfg
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing configuration key: {'.'.join(walked)}")
        value = value[key]
    return value


def validate_config(cfg: dict[str, Any]) -> None:
    for keys in (
        ("project", "data_dir"), ("project", "output_dir"),
        ("input", "pdb_file"), ("input", "psf_file"),
        ("ligand", "residue_name"), ("ligand", "expected_charge_e"),
        ("ligand", "ffxml_file"),
        ("ligand", "rtf_file"), ("ligand", "parameter_file"),
        ("forcefield", "protein_xml"), ("forcefield", "water_xml"),
        ("solvation", "padding_nm"), ("solvation", "ionic_strength_molar"),
        ("system", "nonbonded_cutoff_nm"), ("system", "constraints"),
        ("ensemble", "temperature_K"), ("ensemble", "pressure_bar"),
        ("ensemble", "timestep_ps"), ("run", "production_steps"),
        ("run", "report_interval"), ("analysis", "dt_ps"),
    ):
        _required(cfg, *keys)

    ligand_name = str(cfg["ligand"]["residue_name"]).strip()
    if not ligand_name or any(char.isspace() for char in ligand_name):
        raise ValueError("ligand.residue_name must be a non-empty residue name without spaces")
    if cfg["solvation"]["padding_nm"] <= 0:
        raise ValueError("solvation.padding_nm must be positive")
    if cfg["ensemble"]["timestep_ps"] <= 0:
        raise ValueError("ensemble.timestep_ps must be positive")
    expected_dt = cfg["run"]["report_interval"] * cfg["ensemble"]["timestep_ps"]
    if abs(cfg["analysis"]["dt_ps"] - expected_dt) > 1e-12:
        raise ValueError(
            "analysis.dt_ps must equal run.report_interval * ensemble.timestep_ps "
            f"({expected_dt:g} ps for this configuration)"
        )


def ligand_residues(topology, residue_name: str):
    return [residue for residue in topology.residues() if residue.name == residue_name]


def check_ligand(topology, cfg: dict[str, Any], stage: str) -> None:
    name = cfg["ligand"]["residue_name"]
    residues = ligand_residues(topology, name)
    expected_count = int(cfg["ligand"].get("expected_residue_count", 1))
    if len(residues) != expected_count:
        raise ValueError(
            f"{stage}: expected {expected_count} residue(s) named {name}, found {len(residues)}"
        )
    if expected_count != 1:
        raise ValueError(
            "This workflow currently supports one bound ligand residue per complex; "
            "set ligand.expected_residue_count to 1"
        )
    atoms = list(residues[0].atoms())
    heavy = [atom for atom in atoms if atom.element is not None and atom.element.symbol != "H"]
    expected_atoms = cfg["ligand"].get("expected_atom_count")
    expected_heavy = cfg["ligand"].get("expected_heavy_atom_count")
    if expected_atoms is not None and len(atoms) != int(expected_atoms):
        raise ValueError(
            f"{stage}: {name} must contain {expected_atoms} atoms; found {len(atoms)}"
        )
    if expected_heavy is not None and len(heavy) != int(expected_heavy):
        raise ValueError(
            f"{stage}: {name} must contain {expected_heavy} heavy atoms; found {len(heavy)}"
        )
    print(f"[{stage}] {name}: {len(atoms)} atoms ({len(heavy)} heavy)")


def parse_cgenff_rtf(path: Path, residue_name: str) -> dict[str, Any]:
    atoms: dict[str, tuple[str, float]] = {}
    bonds: set[tuple[str, str]] = set()
    residue_charge: float | None = None
    parameter_penalty: float | None = None
    charge_penalty: float | None = None
    in_residue = False

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if raw.lstrip().startswith("RESI "):
            fields = raw.split()
            in_residue = fields[1] == residue_name
            if in_residue:
                residue_charge = float(fields[2])
                if "param penalty=" in raw:
                    parameter_penalty = float(raw.split("param penalty=", 1)[1].split(";", 1)[0])
                if "charge penalty=" in raw:
                    charge_penalty = float(raw.split("charge penalty=", 1)[1].split()[0])
            continue
        if not in_residue or not line:
            continue
        fields = line.split()
        if fields[0] == "ATOM":
            atoms[fields[1]] = (fields[2], float(fields[3]))
        elif fields[0] == "BOND":
            for index in range(1, len(fields), 2):
                pair = tuple(sorted((fields[index], fields[index + 1])))
                bonds.add(pair)
        elif fields[0] == "END":
            break

    if residue_charge is None or not atoms or not bonds:
        raise ValueError(f"Could not parse residue {residue_name} from {path}")
    return {
        "charge": residue_charge,
        "atoms": atoms,
        "bonds": bonds,
        "parameter_penalty": parameter_penalty,
        "charge_penalty": charge_penalty,
    }


def validate_ligand_files(cfg: dict[str, Any]) -> dict[str, Path]:
    ligand = cfg["ligand"]
    paths = {
        "rtf": data_path(cfg, ligand["rtf_file"]),
        "prm": data_path(cfg, ligand["parameter_file"]),
        "xml": data_path(cfg, ligand["ffxml_file"]),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing ligand {label.upper()} file: {path}")

    name = ligand["residue_name"]
    rtf = parse_cgenff_rtf(paths["rtf"], name)
    if abs(rtf["charge"] - float(ligand["expected_charge_e"])) > 1e-6:
        raise ValueError(f"RTF charge is {rtf['charge']}, expected {ligand['expected_charge_e']}")
    max_penalty = ligand.get("max_parameter_penalty")
    if max_penalty is not None:
        if rtf["parameter_penalty"] is None:
            raise ValueError("CGenFF parameter penalty is missing from the ligand RTF")
        if rtf["parameter_penalty"] > float(max_penalty):
            raise ValueError(
                f"CGenFF parameter penalty {rtf['parameter_penalty']} exceeds the configured "
                f"maximum {max_penalty}"
            )

    root = ET.parse(paths["xml"]).getroot()
    xml_residue = next(
        (node for node in root.findall("./Residues/Residue") if node.attrib.get("name") == name),
        None,
    )
    if xml_residue is None:
        raise ValueError(f"No {name} residue template in {paths['xml']}")
    xml_atoms = {
        atom.attrib["name"]: float(atom.attrib["charge"])
        for atom in xml_residue.findall("Atom")
    }
    if set(xml_atoms) != set(rtf["atoms"]):
        raise ValueError(f"{name} atom names differ between CGenFF RTF and OpenMM XML")
    for atom_name, (_, charge) in rtf["atoms"].items():
        if abs(xml_atoms[atom_name] - charge) > 1e-8:
            raise ValueError(f"{name} charge mismatch for atom {atom_name}")
    xml_bonds = {
        tuple(sorted((bond.attrib["atomName1"], bond.attrib["atomName2"])))
        for bond in xml_residue.findall("Bond")
    }
    if xml_bonds != rtf["bonds"]:
        raise ValueError(f"{name} bonds differ between CGenFF RTF and OpenMM XML")
    parameter_penalty = (
        "not reported" if rtf["parameter_penalty"] is None
        else f"{rtf['parameter_penalty']:.3f}"
    )
    charge_penalty = (
        "not reported" if rtf["charge_penalty"] is None
        else f"{rtf['charge_penalty']:.3f}"
    )
    print(
        f"[params] CGenFF {name}: charge={rtf['charge']:+.0f}, "
        f"parameter penalty={parameter_penalty}, charge penalty={charge_penalty}"
    )
    return paths


def load_charmm_gui_complex(cfg: dict[str, Any]):
    from openmm.app import CharmmPsfFile, PDBFile

    psf_path = data_path(cfg, cfg["input"]["psf_file"])
    pdb_path = data_path(cfg, cfg["input"]["pdb_file"])
    for path in (psf_path, pdb_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing CHARMM-GUI input file: {path}")

    print(f"[input] PSF: {psf_path}")
    print(f"[input] PDB: {pdb_path}")
    psf = CharmmPsfFile(str(psf_path))
    pdb = PDBFile(str(pdb_path))
    if psf.topology.getNumAtoms() != pdb.topology.getNumAtoms():
        raise ValueError("PSF/PDB atom counts do not match")

    psf_atoms = list(psf.topology.atoms())
    pdb_atoms = list(pdb.topology.atoms())
    for index, (psf_atom, pdb_atom) in enumerate(zip(psf_atoms, pdb_atoms), start=1):
        if (psf_atom.name, psf_atom.residue.name) != (pdb_atom.name, pdb_atom.residue.name):
            raise ValueError(
                f"PSF/PDB atom order mismatch at atom {index}: "
                f"{psf_atom.residue.name}:{psf_atom.name} versus "
                f"{pdb_atom.residue.name}:{pdb_atom.name}"
            )
    check_ligand(psf.topology, cfg, "input")
    print(
        f"[input] {psf.topology.getNumAtoms()} atoms, "
        f"{psf.topology.getNumResidues()} residues, "
        f"{sum(1 for _ in psf.topology.chains())} segments"
    )
    return psf, pdb, psf_path, pdb_path


def _register_charmm_gui_terminal_templates(forcefield, psf, ligand_name: str) -> int:
    """Register NTER/CTER signatures already encoded in the CHARMM-GUI PSF."""
    unmatched = forcefield.getUnmatchedResidues(psf.topology)
    if not unmatched:
        return 0
    allowed_terminal_atoms = {"H2", "H3", "OXT"}
    for residue in unmatched:
        atom_names = {atom.name for atom in residue.atoms()}
        if residue.name == ligand_name or not (atom_names & allowed_terminal_atoms):
            raise ValueError(
                f"Unparameterized non-terminal residue: {residue.name} {residue.id} "
                f"in segment {residue.chain.id}"
            )

    templates, representatives = forcefield.generateTemplatesForUnmatchedResidues(psf.topology)
    for template_number, (template, residue) in enumerate(zip(templates, representatives)):
        template.name = f"CGGUI_TERMINUS_{template_number}_{residue.name}"
        real_atoms = {atom.name: atom for atom in residue.atoms()}
        for template_atom in template.atoms:
            topology_atom = real_atoms[template_atom.name]
            psf_atom = psf.atom_list[topology_atom.index]
            candidates = [
                atom_type_name
                for atom_type_name in forcefield._atomClasses[psf_atom.attype]
                if forcefield._atomTypes[atom_type_name].element == template_atom.element
            ]
            if not candidates:
                raise ValueError(
                    f"No CHARMM36 atom type for class {psf_atom.attype} "
                    f"({residue.name}:{template_atom.name})"
                )
            template_atom.type = sorted(candidates)[0]
            template_atom.parameters["charge"] = psf_atom.charge
        forcefield.registerResidueTemplate(template)

    remaining = forcefield.getUnmatchedResidues(psf.topology)
    if remaining:
        details = ", ".join(f"{r.name}{r.id}" for r in remaining)
        raise ValueError(f"Residues still lack force-field templates: {details}")
    return len(unmatched)


def build_forcefield(cfg: dict[str, Any], psf):
    from openmm.app import ForceField

    ff_cfg = cfg["forcefield"]
    ligand_xml = data_path(cfg, cfg["ligand"]["ffxml_file"])
    forcefield = ForceField(ff_cfg["protein_xml"], ff_cfg["water_xml"], str(ligand_xml))
    terminal_count = _register_charmm_gui_terminal_templates(
        forcefield, psf, cfg["ligand"]["residue_name"]
    )
    print(
        f"[forcefield] {ff_cfg['protein_xml']} + {cfg['ligand']['method']} + "
        f"{ff_cfg['water_model']}"
    )
    print(f"[forcefield] matched {terminal_count} CHARMM-GUI terminal residues")
    return forcefield


def run_preflight(cfg: dict[str, Any]) -> None:
    from openmm.app import NoCutoff

    validate_config(cfg)
    ligand_files = validate_ligand_files(cfg)
    psf, _, psf_path, pdb_path = load_charmm_gui_complex(cfg)
    forcefield = build_forcefield(cfg, psf)
    system = forcefield.createSystem(psf.topology, nonbondedMethod=NoCutoff, constraints=None)
    if system.getNumParticles() != psf.topology.getNumAtoms():
        raise RuntimeError("Preflight particle count mismatch")
    print(f"[preflight] system creation successful: {system.getNumParticles()} particles")
    print(f"[preflight] PSF SHA256: {sha256(psf_path)}")
    print(f"[preflight] PDB SHA256: {sha256(pdb_path)}")
    print(f"[preflight] PRM SHA256: {sha256(ligand_files['prm'])}")
    print("[preflight] PASS")


def prepare_system(cfg: dict[str, Any], psf, pdb, forcefield):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    solvation = cfg["solvation"]
    system_cfg = cfg["system"]
    ensemble = cfg["ensemble"]
    out = output_path(cfg)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(int(cfg["run"]["random_seed"]))
    modeller = app.Modeller(psf.topology, pdb.positions)
    check_ligand(modeller.topology, cfg, "before solvation")
    print("[prepare] Adding CHARMM TIP3P water and ions ...")
    modeller.addSolvent(
        forcefield,
        model=cfg["forcefield"]["water_model"],
        padding=float(solvation["padding_nm"]) * unit.nanometer,
        ionicStrength=float(solvation["ionic_strength_molar"]) * unit.molar,
        positiveIon=solvation.get("positive_ion", "Na+"),
        negativeIon=solvation.get("negative_ion", "Cl-"),
        neutralize=bool(solvation.get("neutralize", True)),
        boxShape=solvation.get("box_shape", "dodecahedron"),
    )
    check_ligand(modeller.topology, cfg, "solvated")

    constraints = getattr(app, system_cfg["constraints"])
    create_kwargs: dict[str, Any] = {
        "nonbondedMethod": app.PME,
        "nonbondedCutoff": float(system_cfg["nonbonded_cutoff_nm"]) * unit.nanometer,
        "constraints": constraints,
        "rigidWater": bool(system_cfg.get("rigid_water", True)),
        "hydrogenMass": float(system_cfg["hydrogen_mass_amu"]) * unit.amu,
        "ewaldErrorTolerance": float(system_cfg.get("ewald_error_tolerance", 5e-4)),
    }
    switch_nm = system_cfg.get("switch_distance_nm")
    if switch_nm is not None:
        create_kwargs["switchDistance"] = float(switch_nm) * unit.nanometer
    system = forcefield.createSystem(modeller.topology, **create_kwargs)
    if system.getNumParticles() != modeller.topology.getNumAtoms():
        raise RuntimeError("System particle count does not match solvated topology")

    barostat = mm.MonteCarloBarostat(
        float(ensemble["pressure_bar"]) * unit.bar,
        float(ensemble["temperature_K"]) * unit.kelvin,
        int(ensemble.get("barostat_interval", 25)),
    )
    barostat.setRandomNumberSeed(int(cfg["run"]["random_seed"]))
    system.addForce(barostat)

    solvated_path = out / f"{cfg['project']['protein']}_complex_solvated.pdb"
    system_path = out / "system.xml"
    with solvated_path.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    system_path.write_text(mm.XmlSerializer.serialize(system), encoding="utf-8")

    residues = list(modeller.topology.residues())
    waters = sum(residue.name in {"HOH", "TIP3", "WAT"} for residue in residues)
    ions = sum(residue.name in {"NA", "CL", "SOD", "CLA"} for residue in residues)
    print(
        f"[prepare] {modeller.topology.getNumAtoms()} atoms; "
        f"{waters} waters; {ions} ions"
    )
    print(f"[prepare] solvated PDB: {solvated_path}")
    print(f"[prepare] serialized system: {system_path}")
    return system, modeller, solvated_path, system_path


def _new_simulation(cfg: dict[str, Any], topology, system):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    ensemble = cfg["ensemble"]
    run_cfg = cfg["run"]
    errors: list[str] = []
    for name in run_cfg["platform_priority"]:
        try:
            platform = mm.Platform.getPlatformByName(name)
            available = set(platform.getPropertyNames())
            properties: dict[str, str] = {}
            if "Precision" in available and name in {"CUDA", "OpenCL"}:
                properties["Precision"] = str(run_cfg.get("precision", "mixed"))
            if "Threads" in available and run_cfg.get("cpu_threads"):
                properties["Threads"] = str(run_cfg["cpu_threads"])
            integrator = mm.LangevinMiddleIntegrator(
                float(ensemble["temperature_K"]) * unit.kelvin,
                float(ensemble["friction_per_ps"]) / unit.picosecond,
                float(ensemble["timestep_ps"]) * unit.picoseconds,
            )
            integrator.setRandomNumberSeed(int(run_cfg["random_seed"]))
            simulation = app.Simulation(topology, system, integrator, platform, properties)
            print(f"[sim] platform: {platform.getName()} {properties}")
            return simulation
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"[sim] {name} unavailable: {exc}")
    raise RuntimeError("No usable OpenMM platform. " + " | ".join(errors))


def _save_context_pdb(simulation, path: Path) -> None:
    import openmm.app as app

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    with path.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)


def _add_reporters(simulation, cfg: dict[str, Any], resume: bool) -> list[Any]:
    import openmm.app as app

    out = output_path(cfg)
    run_cfg = cfg["run"]
    report_interval = int(run_cfg["report_interval"])
    checkpoint_interval = int(run_cfg["checkpoint_interval"])
    total_steps = int(run_cfg["equilibration_steps"]) + int(run_cfg["production_steps"])
    trajectory = out / "trajectory.dcd"
    simulation.reporters.append(
        app.DCDReporter(str(trajectory), report_interval, append=resume and trajectory.exists())
    )
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout, report_interval,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, volume=True, density=True,
            speed=True, progress=True, remainingTime=True,
            totalSteps=total_steps, separator="\t",
        )
    )
    state_log_handle = (out / "state_data.csv").open("a" if resume else "w", encoding="utf-8")
    simulation.reporters.append(
        app.StateDataReporter(
            state_log_handle, report_interval,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, volume=True, density=True,
            speed=True, separator=",",
        )
    )
    simulation.reporters.append(
        app.CheckpointReporter(str(out / "checkpoint.chk"), checkpoint_interval)
    )
    return [state_log_handle]


def run(cfg: dict[str, Any], *, prepare_only: bool = False, resume: bool = False) -> None:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    validate_config(cfg)
    validate_ligand_files(cfg)
    psf, pdb, psf_path, pdb_path = load_charmm_gui_complex(cfg)
    out = output_path(cfg)
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = cfg["run"]

    solvated_path = out / f"{cfg['project']['protein']}_complex_solvated.pdb"
    system_path = out / "system.xml"
    checkpoint_path = out / "checkpoint.chk"
    can_resume = resume and all(path.is_file() for path in (solvated_path, system_path, checkpoint_path))

    if can_resume:
        print("[resume] loading saved topology, System, and checkpoint")
        prepared_pdb = app.PDBFile(str(solvated_path))
        topology = prepared_pdb.topology
        system = mm.XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
        positions = prepared_pdb.positions
    else:
        if resume:
            print("[resume] complete restart files not found; starting a fresh run")
        forcefield = build_forcefield(cfg, psf)
        system, modeller, solvated_path, system_path = prepare_system(cfg, psf, pdb, forcefield)
        topology = modeller.topology
        positions = modeller.positions

    if prepare_only:
        print("[prepare-only] preparation completed; dynamics were not started")
        return

    simulation = _new_simulation(cfg, topology, system)
    open_handles: list[Any] = []
    try:
        if can_resume:
            simulation.context.setPositions(positions)
            simulation.loadCheckpoint(str(checkpoint_path))
            print(f"[resume] restored at step {simulation.currentStep}")
        else:
            simulation.context.setPositions(positions)
            print("[sim] energy minimization ...")
            simulation.minimizeEnergy(
                tolerance=float(run_cfg["minimization_tolerance_kj_mol_nm"])
                * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=int(run_cfg["minimization_max_iterations"]),
            )
            _save_context_pdb(simulation, out / "minimized.pdb")

            print(f"[sim] NPT equilibration: {run_cfg['equilibration_steps']} steps")
            simulation.context.setVelocitiesToTemperature(
                float(cfg["ensemble"]["temperature_K"]) * unit.kelvin,
                int(run_cfg["random_seed"]),
            )
            equilibration_steps = int(run_cfg["equilibration_steps"])
            if equilibration_steps > 0:
                equilibration_reporter = app.StateDataReporter(
                    sys.stdout,
                    min(int(run_cfg["report_interval"]), equilibration_steps),
                    step=True, time=True, temperature=True, speed=True,
                    progress=True, remainingTime=True,
                    totalSteps=equilibration_steps, separator="\t",
                )
                simulation.reporters.append(equilibration_reporter)
                simulation.step(equilibration_steps)
                simulation.reporters.remove(equilibration_reporter)
            simulation.saveCheckpoint(str(checkpoint_path))
            _save_context_pdb(simulation, out / "equilibrated.pdb")

        total_target = int(run_cfg["equilibration_steps"]) + int(run_cfg["production_steps"])
        remaining = total_target - int(simulation.currentStep)
        if remaining < 0:
            raise ValueError(
                f"Checkpoint step {simulation.currentStep} exceeds configured target {total_target}"
            )
        if remaining == 0:
            print("[sim] configured production target has already been reached")
        else:
            open_handles = _add_reporters(simulation, cfg, resume=can_resume)
            ns_remaining = remaining * float(cfg["ensemble"]["timestep_ps"]) / 1000.0
            print(f"[sim] production remaining: {remaining} steps ({ns_remaining:.3f} ns)")
            simulation.step(remaining)

        simulation.saveCheckpoint(str(checkpoint_path))
        simulation.saveState(str(out / "final_state.xml"))
        _save_context_pdb(simulation, out / "final_structure.pdb")
        metadata = {
            "input_psf": str(psf_path), "input_pdb": str(pdb_path),
            "ligand": cfg["ligand"], "forcefield": cfg["forcefield"],
            "temperature_K": cfg["ensemble"]["temperature_K"],
            "pressure_bar": cfg["ensemble"]["pressure_bar"],
            "timestep_ps": cfg["ensemble"]["timestep_ps"],
            "completed_step": int(simulation.currentStep),
            "production_ns": (
                int(run_cfg["production_steps"]) * float(cfg["ensemble"]["timestep_ps"]) / 1000.0
            ),
        }
        (out / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[sim] DONE")
    except Exception:
        try:
            simulation.saveCheckpoint(str(out / "emergency_checkpoint.chk"))
        except Exception:
            pass
        raise
    finally:
        for handle in open_handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic OpenMM protein-ligand MD using CHARMM-GUI/CGenFF inputs"
    )
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument(
        "--preflight", action="store_true",
        help="validate every input and build an unsolvated test System, then exit",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="build and save the solvated System, but do not minimize or simulate",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from outputs/checkpoint.chk when matching preparation files exist",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.preflight:
        run_preflight(cfg)
    else:
        run(cfg, prepare_only=args.prepare_only, resume=args.resume)


if __name__ == "__main__":
    main()
