"""
analysis.py
-----------
All MD analysis routines.  Each function is self-contained:
  - receives a pre-loaded MDTraj trajectory (protein-only, superposed)
  - returns numpy arrays  (no side effects)
  - companion `plot_*` function saves the figure and returns the path

Usage
-----
    from src.analysis import (
        compute_rmsd_rmsf_rg,
        compute_pca,
        compute_dssp,
        compute_hbonds,
        compute_sasa,
        compute_contact_map,
        compute_bfactor_comparison,
        build_fel_rmsd_rg,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mdtraj as md
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, pearsonr
from sklearn.decomposition import PCA

from src.plotting import (
    CMAPS,
    PALETTE,
    add_caption,
    add_colorbar,
    apply_style,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)


# ============================================================
# Helpers
# ============================================================

def _unit_scale(plot_unit: str) -> tuple[float, str]:
    """Return (scale_factor, label_string) for Å or nm output."""
    if plot_unit.upper() in ("A", "Å", "ANGSTROM"):
        return 10.0, "Å"
    return 1.0, "nm"


def _savefig(fig: plt.Figure, path: str | Path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")
    return str(path)


def _kBT_kJ(T_K: float = 310.0) -> float:
    R = 0.008314462618   # kJ/mol/K
    return R * T_K


# ============================================================
# 1. RMSD / RMSF / Rg
# ============================================================

def compute_rmsd_rmsf_rg(
    traj: md.Trajectory,
    dt_ps: float = 20.0,
    plot_unit: str = "A",
) -> dict:
    """
    Compute RMSD (backbone), RMSF (Cα), and radius of gyration.

    Returns
    -------
    dict with keys: time_ns, rmsd, rmsf, rg, res_indices, unit_label
    """
    scale, unit_label = _unit_scale(plot_unit)

    backbone  = traj.topology.select("backbone")
    ca        = traj.topology.select("name CA")
    ref       = traj.slice(0)

    rmsd = md.rmsd(traj, ref, atom_indices=backbone) * scale
    rmsf = md.rmsf(traj, ref, atom_indices=ca)       * scale
    rg   = md.compute_rg(traj)                        * scale

    time_ns     = np.arange(traj.n_frames) * dt_ps / 1000.0
    res_indices = np.array([
        traj.topology.atom(i).residue.index for i in ca
    ])

    return dict(
        time_ns=time_ns, rmsd=rmsd, rmsf=rmsf, rg=rg,
        res_indices=res_indices, unit_label=unit_label,
    )


def plot_rmsd_rmsf_rg(data: dict, outdir: str, protein: str) -> str:
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    ul = data["unit_label"]

    # -- RMSD --
    ax = axes[0]
    ax.plot(data["time_ns"], data["rmsd"], color=PALETTE["rmsd"], lw=1.2)
    ax.axhline(np.mean(data["rmsd"]), ls="--", color=PALETTE["rmsd"],
               alpha=0.55, lw=1.0, label=f"mean = {np.mean(data['rmsd']):.2f} {ul}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(f"RMSD ({ul})")
    ax.set_title(f"{protein} — Backbone RMSD")
    ax.legend()

    # -- RMSF --
    ax = axes[1]
    ax.fill_between(data["res_indices"], data["rmsf"],
                    alpha=0.35, color=PALETTE["rmsf"])
    ax.plot(data["res_indices"], data["rmsf"], color=PALETTE["rmsf"], lw=1.0)
    ax.set_xlabel("Residue index")
    ax.set_ylabel(f"RMSF ({ul})")
    ax.set_title(f"{protein} — Cα RMSF")

    # -- Rg --
    ax = axes[2]
    ax.plot(data["time_ns"], data["rg"], color=PALETTE["rg"], lw=1.2)
    ax.axhline(np.mean(data["rg"]), ls="--", color=PALETTE["rg"],
               alpha=0.55, lw=1.0, label=f"mean = {np.mean(data['rg']):.2f} {ul}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(f"Rg ({ul})")
    ax.set_title(f"{protein} — Radius of Gyration")
    ax.legend()

    add_caption(fig, f"{protein} | 310 K | AMBER14 + TIP3P")
    return _savefig(fig, Path(outdir) / "MD_Analysis_Summary.png")


# ============================================================
# 2. PCA + Free Energy Landscape in PC space
# ============================================================

def compute_pca(
    traj: md.Trajectory,
    dt_ps: float = 20.0,
    plot_unit: str = "A",
    n_components: int = 2,
    random_state: int = 0,
) -> dict:
    scale, unit_label = _unit_scale(plot_unit)
    backbone = traj.topology.select("backbone")
    coords   = (traj.xyz[:, backbone, :] * scale).reshape(traj.n_frames, -1)

    pca = PCA(n_components=n_components, random_state=random_state)
    pc  = pca.fit_transform(coords)

    time_ns = np.arange(traj.n_frames) * dt_ps / 1000.0

    print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

    return dict(
        pc=pc, time_ns=time_ns,
        explained=pca.explained_variance_ratio_,
        unit_label=unit_label,
        pca_obj=pca,
    )


def _pc_fel(pc: np.ndarray, T_K: float = 310.0) -> tuple:
    """KDE-based free energy in PC space."""
    xy      = pc[:, :2].T
    kde     = gaussian_kde(xy)
    density = kde(xy)
    density = np.clip(density, 1e-300, None)
    fe      = -_kBT_kJ(T_K) * np.log(density)
    fe     -= fe.min()
    return fe


def plot_pca_fel(data: dict, outdir: str, protein: str, T_K: float = 310.0) -> tuple[str, str]:
    apply_style()
    pc      = data["pc"]
    time_ns = data["time_ns"]
    exp     = data["explained"]
    fe      = _pc_fel(pc, T_K)
    out     = Path(outdir) / "PCA_FEL"
    out.mkdir(parents=True, exist_ok=True)

    ul = data["unit_label"]

    # -- FEL tricontourf --
    fig, ax = plt.subplots(figsize=(7, 6))
    tcf = ax.tricontourf(pc[:, 0], pc[:, 1], fe, levels=200, cmap=CMAPS["fel"])
    fig.colorbar(tcf, ax=ax, label="Free Energy (kJ/mol)")
    ax.set_xlabel(f"PC1 ({exp[0]*100:.1f}%) ({ul})")
    ax.set_ylabel(f"PC2 ({exp[1]*100:.1f}%) ({ul})")
    ax.set_title(f"{protein} — Free Energy Landscape (PCA)")
    add_caption(fig, f"{protein} | KDE | T = {T_K:.0f} K")
    p1 = _savefig(fig, out / "FEL_PCA_Backbone.png")

    # -- Scatter coloured by time --
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pc[:, 0], pc[:, 1], s=6, c=time_ns,
                    cmap=CMAPS["scatter"], alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Time (ns)")
    ax.set_xlabel(f"PC1 ({exp[0]*100:.1f}%) ({ul})")
    ax.set_ylabel(f"PC2 ({exp[1]*100:.1f}%) ({ul})")
    ax.set_title(f"{protein} — PCA Trajectory (coloured by time)")
    add_caption(fig, f"{protein} | Backbone atoms")
    p2 = _savefig(fig, out / "PCA_scatter_time.png")

    return p1, p2


# ============================================================
# 3. DSSP – Secondary Structure Evolution
# ============================================================

def compute_dssp(
    traj: md.Trajectory,
    stride: int = 10,
    dt_ps: float = 20.0,
) -> dict:
    sub   = traj[::stride]
    dssp  = md.compute_dssp(sub, simplified=True)   # shape (frames, residues)

    n_res   = dssp.shape[1]
    helix_f = (dssp == "H").sum(axis=1) / n_res
    sheet_f = (dssp == "E").sum(axis=1) / n_res
    coil_f  = (dssp == "C").sum(axis=1) / n_res

    # time axis
    if hasattr(sub, "time") and sub.time is not None and len(sub.time) > 0:
        time_ns = sub.time / 1000.0
    else:
        frame_dt = dt_ps * stride
        time_ns  = np.arange(sub.n_frames) * frame_dt / 1000.0

    return dict(
        time_ns=time_ns,
        helix=helix_f, sheet=sheet_f, coil=coil_f,
        dssp_raw=dssp,
    )


def plot_dssp(data: dict, outdir: str, protein: str) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(data["time_ns"], data["helix"], color=PALETTE["helix"],
            lw=1.5, label="Helix")
    ax.plot(data["time_ns"], data["sheet"], color=PALETTE["sheet"],
            lw=1.5, label="Sheet")
    ax.plot(data["time_ns"], data["coil"],  color=PALETTE["coil"],
            lw=1.2, label="Coil", ls="--")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Fraction of residues")
    ax.set_title(f"{protein} — Secondary Structure Evolution (DSSP)")
    ax.legend(loc="upper right")
    add_caption(fig, f"{protein} | simplified DSSP | protein-only")
    return _savefig(fig, Path(outdir) / "DSSP_SecondaryStructure_Fractions.png")


# ============================================================
# 4. FEL – RMSD vs Rg  (2D & 3D)
# ============================================================

def build_fel_rmsd_rg(
    rmsd: np.ndarray,
    rg: np.ndarray,
    bins: int = 30,
    sigma: float = 1.9,
    T_K: float = 310.0,
) -> dict:
    h, xedge, yedge = np.histogram2d(rmsd, rg, bins=bins, density=True)
    pseudo  = h.copy()
    nonzero = pseudo > 0
    pseudo[~nonzero] = pseudo[nonzero].min() * 1e-6

    fe          = -_kBT_kJ(T_K) * np.log(pseudo)
    fe         -= fe.min()
    fe_smooth   = gaussian_filter(fe.T, sigma=sigma)

    xc = 0.5 * (xedge[:-1] + xedge[1:])
    yc = 0.5 * (yedge[:-1] + yedge[1:])
    X, Y = np.meshgrid(xc, yc)

    return dict(fe=fe_smooth, X=X, Y=Y, xc=xc, yc=yc)


def plot_fel_2d(
    fel: dict, outdir: str, protein: str,
    unit_label: str = "Å",
) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(fel["X"], fel["Y"], fel["fe"], levels=40, cmap=CMAPS["fel_2d"])
    fig.colorbar(cf, ax=ax, label="Free Energy (kJ/mol)")
    ax.set_xlabel(f"RMSD ({unit_label})")
    ax.set_ylabel(f"Rg ({unit_label})")
    ax.set_title(f"{protein} — 2D Free Energy Landscape")
    add_caption(fig, f"{protein} | T = 310 K | RMSD vs Rg")
    return _savefig(fig, Path(outdir) / "FEL_RMSD_RG" / "FEL_RMSD_RG_2D.png")


def plot_fel_3d(
    fel: dict, outdir: str, protein: str,
    unit_label: str = "Å",
) -> str:
    apply_style()
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        fel["X"], fel["Y"], fel["fe"],
        cmap=CMAPS["fel_2d"], edgecolor="none", alpha=0.92,
    )
    # floor contour projection
    z_off = fel["fe"].min() - 0.5
    ax.contourf(fel["X"], fel["Y"], fel["fe"],
                zdir="z", offset=z_off, cmap=CMAPS["fel_2d"], alpha=0.4)

    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="Free Energy (kJ/mol)")
    ax.set_xlabel(f"RMSD ({unit_label})")
    ax.set_ylabel(f"Rg ({unit_label})")
    ax.set_zlabel("ΔG (kJ/mol)")
    ax.set_title(f"{protein} — 3D Free Energy Landscape")
    ax.view_init(elev=25, azim=55)

    # clean panes
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("0.75")

    add_caption(fig, f"{protein} | T = 310 K | Gaussian-smoothed (σ = 1.9)")
    return _savefig(fig, Path(outdir) / "FEL_RMSD_RG" / "FEL_RMSD_RG_3D.png")


# ============================================================
# 5. Hydrogen Bond Analysis
# ============================================================

def compute_hbonds(
    traj: md.Trajectory,
    dt_ps: float = 20.0,
    freq_threshold: float = 0.3,
) -> dict:
    """
    Count intramolecular H-bonds per frame using Baker-Hubbard criterion.

    Parameters
    ----------
    freq_threshold : float
        Minimum occupancy (0–1) for a bond to be listed as persistent.

    Returns
    -------
    dict with keys: time_ns, hbond_count, persistent_bonds, occupancy
    """
    # shape: (n_hbonds, 3) — donor-H-acceptor indices per bond
    # returns list of arrays; one array per frame
    hbonds_per_frame = [
        md.baker_hubbard(traj.slice(i), periodic=False)
        for i in range(traj.n_frames)
    ]
    hbond_count = np.array([len(h) for h in hbonds_per_frame])
    time_ns     = np.arange(traj.n_frames) * dt_ps / 1000.0

    # persistent bonds: collect (donor_res, acceptor_res) tuples
    bond_occupancy: dict[tuple, int] = {}
    for bonds in hbonds_per_frame:
        seen = set()
        for d, h, a in bonds:
            key = (
                traj.topology.atom(d).residue.index,
                traj.topology.atom(a).residue.index,
            )
            if key not in seen:
                bond_occupancy[key] = bond_occupancy.get(key, 0) + 1
                seen.add(key)

    total = traj.n_frames
    occupancy = {k: v / total for k, v in bond_occupancy.items()}
    persistent = {k: v for k, v in occupancy.items() if v >= freq_threshold}

    print(f"  Mean H-bonds/frame : {hbond_count.mean():.1f} ± {hbond_count.std():.1f}")
    print(f"  Persistent bonds (≥{freq_threshold*100:.0f}% occupancy): {len(persistent)}")

    return dict(
        time_ns=time_ns,
        hbond_count=hbond_count,
        persistent_bonds=persistent,
        occupancy=occupancy,
    )


def plot_hbonds(data: dict, outdir: str, protein: str) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(data["time_ns"], data["hbond_count"],
            color=PALETTE["hbond"], lw=1.0, alpha=0.75)
    ax.axhline(np.mean(data["hbond_count"]), ls="--",
               color=PALETTE["hbond"], alpha=0.6, lw=1.2,
               label=f"mean = {np.mean(data['hbond_count']):.1f}")

    # rolling average
    window = min(50, len(data["hbond_count"]) // 5)
    if window > 1:
        rolling = np.convolve(
            data["hbond_count"],
            np.ones(window) / window, mode="same",
        )
        ax.plot(data["time_ns"], rolling, color="k", lw=1.8,
                label=f"rolling avg ({window} frames)")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Intramolecular H-bonds")
    ax.set_title(f"{protein} — Hydrogen Bond Count")
    ax.legend()
    add_caption(fig, f"{protein} | Baker-Hubbard criterion")
    return _savefig(fig, Path(outdir) / "HBond_Count.png")


# ============================================================
# 6. Solvent Accessible Surface Area (SASA)
# ============================================================

def compute_sasa(
    traj: md.Trajectory,
    dt_ps: float = 20.0,
    plot_unit: str = "A",
) -> dict:
    """
    Compute total SASA per frame using Shrake-Rupley algorithm.
    MDTraj returns SASA in nm²; we optionally convert to Å².
    """
    sasa_nm2 = md.shrake_rupley(traj, mode="residue").sum(axis=1)   # nm²

    if plot_unit.upper() in ("A", "Å"):
        sasa   = sasa_nm2 * 100.0   # nm² → Å²
        ul     = "Å²"
    else:
        sasa   = sasa_nm2
        ul     = "nm²"

    time_ns = np.arange(traj.n_frames) * dt_ps / 1000.0

    print(f"  Mean SASA : {sasa.mean():.1f} ± {sasa.std():.1f} {ul}")
    return dict(time_ns=time_ns, sasa=sasa, unit_label=ul)


def plot_sasa(data: dict, outdir: str, protein: str) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(data["time_ns"], data["sasa"],
                    alpha=0.25, color=PALETTE["sasa"])
    ax.plot(data["time_ns"], data["sasa"],
            color=PALETTE["sasa"], lw=1.2)
    ax.axhline(np.mean(data["sasa"]), ls="--",
               color=PALETTE["sasa"], alpha=0.65, lw=1.2,
               label=f"mean = {np.mean(data['sasa']):.1f} {data['unit_label']}")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(f"SASA ({data['unit_label']})")
    ax.set_title(f"{protein} — Solvent Accessible Surface Area")
    ax.legend()
    add_caption(fig, f"{protein} | Shrake-Rupley | probe r = 0.14 nm")
    return _savefig(fig, Path(outdir) / "SASA.png")


# ============================================================
# 7. Residue–Residue Contact Map
# ============================================================

def compute_contact_map(
    traj: md.Trajectory,
    scheme: str = "ca",
    threshold_nm: float = 0.8,
) -> dict:
    """
    Compute average pairwise Cα contact map.

    Parameters
    ----------
    scheme       : MDTraj contact scheme ('ca', 'closest', 'closest-heavy')
    threshold_nm : distance cutoff for a 'contact' (default 0.8 nm = 8 Å)

    Returns
    -------
    dict with keys: contact_map (n_res × n_res), pairs, n_res
    """
    distances, pairs = md.compute_contacts(traj, scheme=scheme)
    # distances shape: (frames, n_pairs)

    n_res = traj.topology.n_residues
    avg_dist = distances.mean(axis=0)          # mean over trajectory

    # Convert to square matrix
    cmap = np.zeros((n_res, n_res))
    for (i, j), d in zip(pairs, avg_dist):
        cmap[i, j] = d
        cmap[j, i] = d

    # Binary contact matrix (1 = in contact, avg distance < threshold)
    contact_binary = (cmap < threshold_nm) & (cmap > 0)

    print(f"  Residues          : {n_res}")
    print(f"  Contacting pairs  : {contact_binary.sum() // 2}")

    return dict(
        contact_map=cmap,
        contact_binary=contact_binary.astype(float),
        pairs=pairs,
        n_res=n_res,
        threshold_nm=threshold_nm,
    )


def plot_contact_map(data: dict, outdir: str, protein: str) -> str:
    apply_style()
    n   = data["n_res"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # -- average distance --
    ax = axes[0]
    im = ax.imshow(data["contact_map"], cmap=CMAPS["contact"],
                   origin="lower", aspect="auto",
                   vmin=0, vmax=data["contact_map"].max())
    fig.colorbar(im, ax=ax, label="Avg Cα distance (nm)")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    ax.set_title(f"{protein} — Average Distance Map")

    # -- binary contact map --
    ax = axes[1]
    im2 = ax.imshow(data["contact_binary"], cmap="Greys",
                    origin="lower", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im2, ax=ax, label=f"Contact (d < {data['threshold_nm']*10:.0f} Å)")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    ax.set_title(f"{protein} — Binary Contact Map")

    add_caption(fig, f"{protein} | Cα scheme | threshold = {data['threshold_nm']*10:.0f} Å")
    return _savefig(fig, Path(outdir) / "ContactMap.png")


# ============================================================
# 8. MD B-factor vs Crystallographic B-factor
# ============================================================

def compute_bfactor_comparison(
    traj: md.Trajectory,
    rmsf_angstrom: np.ndarray,
    pdb_path: str,
) -> dict:
    """
    Convert MD RMSF → theoretical B-factor and compare to PDB B-factors.

    B = (8π²/3) × RMSF²    (units: Å² if RMSF in Å)

    Parameters
    ----------
    rmsf_angstrom : 1-D array of per-residue RMSF in Å (from compute_rmsd_rmsf_rg)
    pdb_path      : path to the original (unsolvated) PDB for reading B-factors

    Returns
    -------
    dict with keys: bfactor_md, bfactor_xray, res_indices, pearson_r, p_value
    """
    # MD → theoretical B-factor
    bfactor_md = (8 * np.pi**2 / 3) * rmsf_angstrom**2

    # Read X-ray B-factors from PDB
    import mdtraj as md2
    ref_traj = md2.load(pdb_path)
    ca_idx   = ref_traj.topology.select("name CA")
    bfactor_xray = ref_traj.topology.to_dataframe()[0].loc[ca_idx, "bfactor"].values

    # Align lengths (crystal may be shorter/longer)
    min_len        = min(len(bfactor_md), len(bfactor_xray))
    bfactor_md     = bfactor_md[:min_len]
    bfactor_xray   = bfactor_xray[:min_len]
    res_indices    = np.arange(min_len)

    # Mask residues with zero X-ray B-factor (missing data)
    mask = bfactor_xray > 0
    r, p = pearsonr(bfactor_md[mask], bfactor_xray[mask])

    print(f"  Pearson r (MD vs X-ray B-factor): {r:.3f}  (p = {p:.2e})")

    return dict(
        bfactor_md=bfactor_md,
        bfactor_xray=bfactor_xray,
        res_indices=res_indices,
        pearson_r=r,
        p_value=p,
    )


def plot_bfactor(data: dict, outdir: str, protein: str) -> str:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # -- overlay --
    ax = axes[0]
    ax.plot(data["res_indices"], data["bfactor_xray"],
            color="dimgrey", lw=1.2, label="X-ray B-factor")
    ax.plot(data["res_indices"], data["bfactor_md"],
            color=PALETTE["rmsf"], lw=1.2, ls="--",
            label="MD B-factor")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("B-factor (Å²)")
    ax.set_title(f"{protein} — MD vs Crystallographic B-factor  "
                 f"(r = {data['pearson_r']:.2f})")
    ax.legend()

    # -- scatter correlation --
    ax = axes[1]
    ax.scatter(data["bfactor_xray"], data["bfactor_md"],
               s=18, alpha=0.55, color=PALETTE["rmsf"], edgecolors="none")
    lims = [
        min(data["bfactor_xray"].min(), data["bfactor_md"].min()),
        max(data["bfactor_xray"].max(), data["bfactor_md"].max()),
    ]
    ax.plot(lims, lims, "k--", lw=1.0, alpha=0.5, label="y = x")
    ax.set_xlabel("X-ray B-factor (Å²)")
    ax.set_ylabel("MD B-factor (Å²)")
    ax.set_title(f"Correlation  r = {data['pearson_r']:.3f},  "
                 f"p = {data['p_value']:.2e}")
    ax.legend()

    add_caption(fig, f"{protein} | B = (8π²/3) × RMSF²")
    return _savefig(fig, Path(outdir) / "BFactor_Comparison.png")

# ============================================================
# Command-line entry point
# ============================================================

import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_trajectory(
    cfg: dict,
    traj_path: Optional[str] = None,
    top_path: Optional[str] = None,
) -> md.Trajectory:
    """Load a trajectory, keep protein atoms only, superpose on frame 0."""
    out     = Path(cfg["project"]["output_dir"])
    protein = cfg["project"]["protein"]

    if traj_path is None:
        traj_path = out / "trajectory.dcd"
    if top_path is None:
        top_path = out / f"{protein}_complex_solvated.pdb"

    print(f"[load] trajectory : {traj_path}")
    print(f"[load] topology   : {top_path}")

    traj = md.load(str(traj_path), top=str(top_path))

    # protein-only + superpose on first frame using Cα
    protein_idx = traj.topology.select("protein")
    traj = traj.atom_slice(protein_idx)
    ca = traj.topology.select("name CA")
    traj.superpose(traj, 0, atom_indices=ca)

    print(f"[load] {traj.n_frames} frames | {traj.n_atoms} protein atoms")
    return traj


def run_analysis(
    cfg: dict,
    traj_path: Optional[str] = None,
    top_path: Optional[str] = None,
) -> None:
    apply_style()

    a         = cfg["analysis"]
    protein   = cfg["project"]["protein"]
    outdir    = cfg["project"]["output_dir"]
    T_K       = cfg["ensemble"]["temperature_K"]
    dt_ps     = a["dt_ps"]
    plot_unit = a["plot_unit"]

    traj = load_trajectory(cfg, traj_path, top_path)

    print("\n[1/8] RMSD / RMSF / Rg …")
    rrr = compute_rmsd_rmsf_rg(traj, dt_ps=dt_ps, plot_unit=plot_unit)
    plot_rmsd_rmsf_rg(rrr, outdir, protein)

    print("\n[2/8] PCA + FEL …")
    pca = compute_pca(traj, dt_ps=dt_ps, plot_unit=plot_unit)
    plot_pca_fel(pca, outdir, protein, T_K=T_K)

    print("\n[3/8] DSSP …")
    dssp = compute_dssp(traj, stride=a["dssp_stride"], dt_ps=dt_ps)
    plot_dssp(dssp, outdir, protein)

    print("\n[4/8] FEL (RMSD vs Rg) …")
    fel = build_fel_rmsd_rg(
        rrr["rmsd"], rrr["rg"],
        bins=a["fel"]["bins"], sigma=a["fel"]["sigma_smooth"], T_K=T_K,
    )
    plot_fel_2d(fel, outdir, protein, unit_label=rrr["unit_label"])
    plot_fel_3d(fel, outdir, protein, unit_label=rrr["unit_label"])

    print("\n[5/8] Hydrogen bonds …")
    hb = compute_hbonds(traj, dt_ps=dt_ps,
                        freq_threshold=a.get("hbond_threshold", 0.3))
    plot_hbonds(hb, outdir, protein)

    print("\n[6/8] SASA …")
    sasa = compute_sasa(traj, dt_ps=dt_ps, plot_unit=plot_unit)
    plot_sasa(sasa, outdir, protein)

    print("\n[7/8] Contact map …")
    cm = compute_contact_map(
        traj,
        scheme=a.get("contact_scheme", "ca"),
        threshold_nm=a.get("contact_cutoff_nm", 0.8),
    )
    plot_contact_map(cm, outdir, protein)

    print("\n[8/8] B-factor comparison …")
    pdb_path = Path(cfg["project"]["data_dir"]) / cfg["input"]["pdb_file"]
    try:
        bf = compute_bfactor_comparison(traj, rrr["rmsf"], str(pdb_path))
        plot_bfactor(bf, outdir, protein)
    except Exception as e:
        print(f"  [warn] B-factor comparison skipped: {e}")

    print(f"\n[done] All figures saved under: {outdir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MD trajectory analysis")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--traj", default=None,
                        help="Override trajectory file (.dcd/.xtc/.h5)")
    parser.add_argument("--top", default=None,
                        help="Override topology file (solvated .pdb/.prmtop)")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    run_analysis(cfg, traj_path=args.traj, top_path=args.top)


if __name__ == "__main__":
    main()
