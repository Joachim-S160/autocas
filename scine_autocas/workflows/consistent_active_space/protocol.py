# -*- coding: utf-8 -*-
__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details. """

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Tuple
from optparse import OptionParser
from scine_autocas.workflows.consistent_active_space.configuration import ConsistentActiveSpaceConfiguration
from scine_autocas.cas_selection.cas_combination import combine_active_spaces
from scine_autocas.interfaces import Molcas, Serenity
from scine_autocas.utils.defaults import CasMethods, PostCasMethods
from scine_autocas.workflows.consistent_active_space.prepare_orbitals import (
    load_orbitals_from_serenity, construct_molecules, print_orbital_map
)
from scine_autocas.utils.defaults import Defaults
from scine_autocas.workflows.consistent_active_space.run_autocas import run_autocas
from scine_autocas.io import FileHandler


def plot_ibo_distribution(serenity: Serenity) -> None:
    """
    Generate IBO orbital distribution plots using IAO-constrained classification.

    Shows the proper IAO constraint: nValVirt = nMINAO - nOcc

    Parameters
    ----------
    serenity : Serenity
        The Serenity interface with initialized systems.
    """
    if not hasattr(serenity, 'systems') or not serenity.systems:
        return
    if not FileHandler.check_project_dir_exists():
        return

    # Get first system's HDF5 file
    sys_zero = serenity.systems[0]
    sys_name = sys_zero.getSystemName()
    sys_path = sys_zero.getSettings().path
    h5file = Path(sys_path) / sys_name / f"{sys_name}.scf.h5"

    if not h5file.exists():
        print(f"[WARNING] HDF5 file not found for IBO plot: {h5file}")
        return

    # Find IBO_distr_IAO.py script (5 parents up from this file to reach autoCAS4HE)
    script_path = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "IBO_distr_IAO.py"
    if not script_path.exists():
        print(f"[WARNING] IBO distribution script not found: {script_path}")
        return

    print("Plotting IBO orbital distribution (IAO-constrained)")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(h5file)],
            cwd=str(h5file.parent), check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[WARNING] IBO plot script error: {result.stderr[:200]}")

        # Copy generated plots to project directory
        project_path = FileHandler.get_project_path()
        for pdf in h5file.parent.glob("*_IBO_IAO*.pdf"):
            dest = f"{project_path}/{FileHandler.PlotNames.ibo_distribution_file}"
            shutil.copy2(pdf, dest)
            print(f"  Saved: {dest}")
    except Exception as e:
        print(f"[WARNING] IBO distribution analysis failed: {e}")


def _isolate_worker_scratch(base_scratch: str, worker_scratch: str, project_name: str = "") -> None:
    """Create worker_scratch and copy pre-computed integral files for this system.

    OpenMolcas creates a transient INPORB symlink (unprefixed) in WorkDir.  If two
    OpenMolcas processes share the same WorkDir they overwrite each other's INPORB,
    causing each to read the wrong orbital file.  This helper gives each parallel
    worker its own WorkDir.

    We copy (not symlink) files that match this system's project_name prefix.
    Copying avoids HDF5 file-lock conflicts: symlinks to the same inode are locked
    as a unit, so two processes using symlinks to the same .h5 file both get errno=11.
    Only files for THIS system are copied; other systems' files stay in base_scratch.
    Non-prefixed driver files (coord.inp, molcas.control, …) are created fresh by
    OpenMolcas when it starts — they do not need to be pre-populated.
    """
    os.makedirs(worker_scratch, exist_ok=True)
    if not os.path.isdir(base_scratch):
        return
    prefix = project_name + "." if project_name else ""
    for fname in os.listdir(base_scratch):
        if prefix and not fname.startswith(prefix):
            continue
        src = os.path.join(base_scratch, fname)
        dst = os.path.join(worker_scratch, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def _run_parallel(jobs: list, n_workers: int) -> list:
    """Run zero-arg callables in parallel forked processes (at most n_workers at once).

    Uses fork so complex non-picklable objects (Molcas interfaces, closures) work as-is.
    Each worker puts (idx, result, traceback_or_None) into a shared queue.
    Results are returned in input order. Raises RuntimeError if any job fails.
    """
    if n_workers <= 1:
        return [fn() for fn in jobs]

    q: mp.Queue = mp.Queue()
    sem = mp.Semaphore(n_workers)
    procs = []

    for i, fn in enumerate(jobs):
        sem.acquire()

        def _child(idx=i, f=fn):
            result = tb = None
            try:
                result = f()
            except Exception:
                tb = traceback.format_exc()
            q.put((idx, result, tb))
            sem.release()

        p = mp.Process(target=_child)
        p.start()
        procs.append(p)

    # Collect exactly len(jobs) results; blocks until each worker puts its item.
    raw: dict = {}
    errors: list = []
    for _ in range(len(jobs)):
        idx, result, tb = q.get()
        if tb is not None:
            errors.append(f"[worker {idx}] {tb}")
        else:
            raw[idx] = result

    for p in procs:
        p.join()

    if errors:
        raise RuntimeError("Parallel workers failed:\n" + "\n".join(errors))

    return [raw[i] for i in range(len(jobs))]


def run_from_command_line() -> None:
    """
    Run the consistent active space protocol from the command line.
    """
    parser = OptionParser(description="Run the consistent active space protocol for a set of molecules. The input can"
                                      " be either a set of XYZ files and command line options or a yaml file provided"
                                      " through the option -y. ")
    parser.add_option("-s", "--load_orbitals", dest="load_orbitals", action="store_true", default=False,
                      help="If true, paths to the Serenity system directories are expected instead of the"
                           " path to the XYZ files. The orbitals are then loaded from these directories.")
    parser.add_option("-m", "--cas_method", dest="cas_method", action="store", default="CASPT2")
    parser.add_option("-i", "--autocas_indices", dest="autocas_molecule_indices", default="",
                      help="The molecule indices used for the autocas selection. By default the CAS"
                           " is selected for all molecules.")
    parser.add_option("-b", "--basis", dest="basis_set", default=Defaults.Interface.basis_set,
                      help="The atomic basis set label.")
    parser.add_option("-l", "--large_active_space", dest="large_active_space", default=True, action="store_false",
                      help="If given, the large active space protocoll in autocas is not used. By default it is used.")
    parser.add_option("-c", "--exclude_core", dest="exclude_core", default=True, action="store_false",
                      help="If given, core orbitals can be selected for the active space. By default, all core"
                           " orbitals are removed.")
    parser.add_option("-u", "--unmapable", dest="always_include_unmapables", default=False, action="store_true")
    parser.add_option("-e", "--external_orbitals", dest="use_external_orbitals", action="store_true", default=False,
                      help="If true, external orbital files (e.g., from OpenMolcas with DKH2) are used instead of "
                           "running Serenity SCF. Use with -o to specify the orbital files.")
    parser.add_option("-o", "--orbital_files", dest="external_orbital_files", default="", type="str",
                      help="Comma-separated list of paths to external orbital files (e.g., OpenMolcas .ScfOrb files). "
                           "One file per system is required. Use with -e flag.")
    parser.add_option("-L", "--localization", dest="localization_method", default="IBO", type="str",
                      help="Orbital localization method. Options: IBO (default), PIPEK_MEZEY, BOYS, "
                           "EDMINSTON_RUEDENBERG. Use PIPEK_MEZEY or BOYS for heavy elements where IBO fails.")
    parser.add_option("-f", "--force-cas", dest="force_cas", action="store_true", default=False,
                      help="Force active space selection even when single-orbital entropies indicate a "
                           "single-reference system. Useful for systems with low initial entropies.")
    parser.add_option("-S", "--skip-localization", dest="skip_localization", action="store_true", default=False,
                      help="Skip orbital localization entirely. Use canonical orbitals directly. "
                           "Useful when localization causes issues (e.g., NaN coefficients).")
    parser.add_option("-y", "--yaml", dest="yaml_file", default="", type="str",
                      help="The configuration yaml file to use. If given, xyz files/loading paths must not be set"
                           " and all other options provided through the command line are ignored.")
    parser.add_option("-n", "--create_no_yaml", dest="create_yaml", default=True, action="store_false",
                      help="If true, the configuration will be written to a yaml file. By default, a yaml"
                           " file is created.")
    parser.add_option("--rasscf_sx_max_iter", dest="rasscf_sx_max_iter", default=None, type="int",
                      help="Override max SX inner iterations per RASSCF macro-step (ITERations keyword, 2nd value). "
                           "Default: use Molcas.Settings default (100). Use e.g. 150 to test higher values.")
    parser.add_option("--rasscf-level-shift", dest="rasscf_level_shift", default=None, type="float",
                      help="Override RASSCF LEVShift in Eh (LEVShift keyword). "
                           "Default: use Molcas.Settings default (0.5 Eh). Use 0.1 when DMRG "
                           "orbitals are already a good starting point to avoid oscillation "
                           "(e.g. PbO near ionic/covalent crossing at R=7 Ang).")
    parser.add_option("--restart-from-dmrg", dest="restart_from_dmrg", action="store_true", default=False,
                      help="Skip orbital prep, IBO, DMRG, and CAS combination. Read combined_cas_spaces "
                           "from the existing autocas_project/dmrg/ directory and re-run only the final "
                           "CASSCF. Use with --rasscf-level-shift to fix convergence failures without "
                           "repeating the expensive DMRG step.")
    parser.add_option("-M", "--multiplicity", dest="spin_multiplicity", default=1, type="int",
                      help="Spin multiplicity (2S+1) of the molecule. Default: 1 (singlet). "
                           "Use 3 for triplet, appropriate for heavy diatomics at the dissociation "
                           "limit (e.g. PbO, Pb2, Po2 with Pb/Po in 3P ground state).")
    parser.add_option("--ibo-minao-basis", dest="ibo_minao_basis", default="MINAO", type="string",
                      help="Minimal basis for IAO/IBO construction. Default: MINAO. "
                           "Use MINAO1/MINAO2/MINAO3 for open-shell systems where nOcc_alpha > nMINAO/2 "
                           "(e.g. triplet PbO: use MINAO1 for 5 virtual valence slots instead of 2).")
    parser.add_option("--save-per-geom-casscf", dest="save_per_geom_casscf", action="store_true", default=False,
                      help="After per-geometry DMRG selection, run a CASSCF with each geometry's own "
                           "entropy-selected CAS and save rasscf.h5 to pergeom/. "
                           "Allows Pegamoid inspection before the union CAS is applied.")
    parser.add_option("--n-workers", dest="n_workers", default=1, type="int",
                      help="Number of parallel worker processes for per-geometry loops (DMRG entropy, "
                           "per-geom CASSCF, final CASSCF). Default: 1 (serial). "
                           "Uses forked subprocesses — safe for os.chdir()-heavy code. "
                           "Set to the number of available CPU cores for maximum throughput.")
    (options, args) = parser.parse_args()
    if options.yaml_file:
        if len(args) > 0:
            raise ValueError("If a yaml file is given, no other arguments must be set.")
        configuration = ConsistentActiveSpaceConfiguration.from_file(options.yaml_file)
    else:
        configuration = ConsistentActiveSpaceConfiguration.from_options(options, args)
        if options.create_yaml:
            configuration.write_yaml_file()
    run_consistent_active_space_protocol(configuration)


def _parse_combined_cas_spaces(path: str) -> Tuple[List[List[int]], List[List[int]]]:
    """Parse a combined_cas_spaces file and return (occupations, indices).

    Parameters
    ----------
    path : str
        Absolute path to the combined_cas_spaces file.

    Returns
    -------
    Tuple[List[List[int]], List[List[int]]]
        (occupations, indices) — one entry per system in the same order as the file.
    """
    import ast
    occupations: List[List[int]] = []
    indices: List[List[int]] = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        if line.startswith("combined cas indices:"):
            indices.append(ast.literal_eval(line.split(":", 1)[1].strip()))
        elif line.startswith("combined occupation:"):
            occupations.append(ast.literal_eval(line.split(":", 1)[1].strip()))
    if len(occupations) != len(indices):
        raise ValueError(
            f"Malformed combined_cas_spaces: {len(indices)} index lines, "
            f"{len(occupations)} occupation lines in {path}."
        )
    return occupations, indices


def run_consistent_active_space_protocol(configuration: ConsistentActiveSpaceConfiguration)\
        -> Tuple[List[List[int]], List[List[int]], List[float]]:
    """
    Run the consistent active space protocol.

    Parameters
    ----------
    configuration: ConsistentActiveSpaceConfiguration
        The configuration for the protocol.

    Returns
    -------
    Tuple[List[List[int]], List[List[int]], List[float]]
        The combined active space occupations, indices, and energies for each structure.
    """
    initial_project_name = FileHandler.DirectoryNames.project_name
    FileHandler.DirectoryNames.project_name = configuration.project_name
    FileHandler.setup_project()
    print("Final energy evaluation method: " + configuration.cas_method)
    print("*******************************************************************************************")
    print("*                                                                                         *")
    print("*                               Orbital Preparation                                       *")
    print("*                                                                                         *")
    print("*******************************************************************************************")
    # Load orbitals if desired.
    serenity = None
    settings = configuration.get_serenity_interface_settings()
    if configuration.load_orbitals:
        load_paths = configuration.get_serenity_load_paths()
        serenity, molecules = load_orbitals_from_serenity(load_paths, settings)
        orbital_map, unmappable_orbitals = serenity.get_orbital_map()
    # Generate the Molcas orbital files.
    interfaces, molecules = construct_molecules(configuration)

    # Apply CLI overrides to molcas settings
    if configuration.rasscf_sx_max_iter is not None:
        for molcas in interfaces:
            molcas.settings.rasscf_sx_max_iter = configuration.rasscf_sx_max_iter
    if configuration.rasscf_level_shift is not None:
        for molcas in interfaces:
            molcas.settings.rasscf_level_shift = configuration.rasscf_level_shift

    # Restart from existing DMRG results: skip orbital prep, IBO, DMRG, and CAS combination.
    if configuration.restart_from_dmrg:
        dmrg_dir = os.path.join(FileHandler.get_project_path(),
                                FileHandler.DirectoryNames.initial_dmrg)
        cas_file = os.path.join(dmrg_dir, "combined_cas_spaces")
        if not os.path.exists(cas_file):
            raise FileNotFoundError(
                f"--restart-from-dmrg: cannot find {cas_file}. "
                f"Run the full protocol first to generate DMRG results.")
        print(f"[restart] Reading combined active spaces from {cas_file}")
        combined_occupations, combined_indices = _parse_combined_cas_spaces(cas_file)
        n_systems = len(configuration.xyz_files)
        if len(combined_occupations) != n_systems:
            raise ValueError(
                f"--restart-from-dmrg: {len(combined_occupations)} entries in combined_cas_spaces "
                f"but {n_systems} xyz files given. Check that the same xyz list is provided.")
        for molcas, name in zip(interfaces, configuration.system_names):
            sel_file = os.path.join(dmrg_dir, f"{name}.scf.h5_sel")
            if not os.path.exists(sel_file):
                raise FileNotFoundError(
                    f"--restart-from-dmrg: orbital file not found: {sel_file}")
            molcas.orbital_file = os.path.abspath(sel_file)
        print("*******************************************************************************************")
        print("*                                                                                         *")
        print("*                    Final Calculations (restarted from DMRG)                            *")
        print("*                                                                                         *")
        print("*******************************************************************************************")
        def _make_restart_job(iface, cas_occ, cas_idx, method):
            base_scratch = iface.environment.molcas_scratch_dir
            def job():
                worker_scratch = os.path.join(base_scratch, "restart_" + iface.project_name)
                _isolate_worker_scratch(base_scratch, worker_scratch, iface.project_name)
                iface.environment.molcas_scratch_dir = worker_scratch
                _, _, energy = run_final_calculation(iface, cas_occ, cas_idx, method)
                return energy, iface.orbital_file
            return job

        restart_jobs = [
            _make_restart_job(iface, cas_occ, cas_idx, configuration.cas_method)
            for iface, cas_occ, cas_idx in zip(interfaces, combined_occupations, combined_indices)
        ]
        restart_results = _run_parallel(restart_jobs, configuration.n_workers)
        energies: List[float] = []
        for i, (energy, orb_file) in enumerate(restart_results):
            energies.append(energy)
            interfaces[i].orbital_file = orb_file
        os.chdir(FileHandler.get_project_path())
        f = open("energies.dat", "w")
        for e in energies:
            f.write(str(e) + "\n")
        f.close()
        FileHandler.DirectoryNames.project_name = initial_project_name
        return combined_occupations, combined_indices, energies

    # If the orbitals are loaded from existing files, Serenity will be initialized.
    # Otherwise, we run the SCF calculation with Serenity.
    if serenity is None:
        serenity = Serenity(molecules, settings)
        # Check if using external orbitals (e.g., from OpenMolcas with DKH2)
        if configuration.use_external_orbitals:
            print("Loading external orbitals (skipping Serenity SCF)...")
            print(f"External orbital files: {configuration.external_orbital_files}")
            # Copy external orbital files to the initial directory with correct system names
            # Serenity reads from .scf.h5 files (HDF5 format), NOT .ScfOrb (ASCII)
            # The .scf.h5 files contain MO_ENERGIES and MO_VECTORS datasets
            initial_dir = serenity.settings.molcas_orbital_files[0]  # All systems use same initial dir
            for i, (ext_orb_file, sys_name) in enumerate(zip(configuration.external_orbital_files,
                                                              configuration.system_names)):
                # Determine the correct destination file based on the source file extension
                # We support both .scf.h5 (preferred) and .ScfOrb files
                if ext_orb_file.endswith('.scf.h5'):
                    dest_file = os.path.join(initial_dir, f"{sys_name}.scf.h5")
                elif ext_orb_file.endswith('.ScfOrb'):
                    # Also copy .ScfOrb for compatibility, but warn that .scf.h5 is preferred
                    dest_file = os.path.join(initial_dir, f"{sys_name}.ScfOrb")
                    print(f"  WARNING: Using .ScfOrb file. For eigenvalues to be read correctly, use .scf.h5 files.")
                else:
                    # Try to guess based on file existence
                    dest_file = os.path.join(initial_dir, f"{sys_name}.scf.h5")
                print(f"  Copying {ext_orb_file} -> {dest_file}")
                shutil.copy2(ext_orb_file, dest_file)
            # Load the external orbitals from the copied HDF5 files
            serenity.load_or_write_molcas_orbitals()
            # Skip Serenity SCF - we use the external orbitals directly
        else:
            serenity.load_or_write_molcas_orbitals()
            serenity.calculate()
        orbital_map, unmappable_orbitals = serenity.get_orbital_map()
    names = serenity.settings.system_names
    print_orbital_map(orbital_map)
    serenity.load_or_write_molcas_orbitals(True)

    # Generate IBO distribution plot (IAO-constrained classification)
    try:
        plot_ibo_distribution(serenity)
    except Exception as e:
        print(f"[WARNING] IBO distribution plot failed (non-fatal): {e}")

    cas_occupations: List[List[int]] = [[] for _ in names]
    cas_indices: List[List[int]] = [[] for _ in names]

    # Absolute paths for output text files — avoid CWD-dependency in parallel mode
    # (parallel workers fork and os.chdir independently; parent CWD may not advance)
    _dmrg_dir = os.path.join(FileHandler.get_project_path(), FileHandler.DirectoryNames.initial_dmrg)
    _final_dir = os.path.join(FileHandler.get_project_path(), FileHandler.DirectoryNames.final_calc)

    # Run autoCAS (DMRG entropy) — parallel over geometries
    print(f"Running DMRG entropy for {len(configuration.autocas_indices)} geometries "
          f"({configuration.n_workers} worker(s))")

    def _make_dmrg_job(mol, iface, nm, large_cas, force):
        base_scratch = iface.environment.molcas_scratch_dir
        def job():
            worker_scratch = os.path.join(base_scratch, iface.project_name)
            _isolate_worker_scratch(base_scratch, worker_scratch, iface.project_name)
            iface.environment.molcas_scratch_dir = worker_scratch
            iface.set_initial_cas_state(False)
            cas_occ, cas_idx = run_autocas(mol, iface, nm, large_cas, force)
            return cas_occ, cas_idx, iface.orbital_file
        return job

    dmrg_jobs = [
        _make_dmrg_job(molecules[i], interfaces[i], names[i],
                       configuration.large_active_space, configuration.force_cas)
        for i in configuration.autocas_indices
    ]
    dmrg_results = _run_parallel(dmrg_jobs, configuration.n_workers)
    for i, (cas_occ, cas_idx, orb_file) in zip(configuration.autocas_indices, dmrg_results):
        cas_occupations[i] = cas_occ  # type: ignore
        cas_indices[i] = cas_idx
        interfaces[i].orbital_file = orb_file  # sync mutation from forked worker

    # Log per-geometry CAS selections before combining (for inspection/debugging)
    os.makedirs(_dmrg_dir, exist_ok=True)
    per_geom_log = open(os.path.join(_dmrg_dir, "per_geometry_cas_spaces"), "w")
    for cas_idx, cas_occ, nm in zip(cas_indices, cas_occupations, names):
        n_e = sum(cas_occ) if cas_occ else 0
        per_geom_log.write(f"system {nm}: CAS({n_e},{len(cas_idx)}) indices: {cas_idx}\n")
        per_geom_log.write(f"system {nm}: occupation: {cas_occ}\n")
        print(f"  per-geom CAS {nm}: CAS({n_e},{len(cas_idx)})")
    per_geom_log.close()

    if configuration.save_per_geom_casscf:
        print(f"*** Running per-geometry CASSCF (pre-union), {configuration.n_workers} worker(s) ***")
        prev_dir = os.getcwd()
        orig_final_name = FileHandler.DirectoryNames.final_calc
        orig_orbital_files = [m.orbital_file for m in interfaces]
        FileHandler.DirectoryNames.final_calc = FileHandler.DirectoryNames.per_geom_calc
        try:
            def _make_pergeom_job(iface, cas_occ, cas_idx, method):
                base_scratch = iface.environment.molcas_scratch_dir
                def job():
                    worker_scratch = os.path.join(base_scratch, "pergeom_" + iface.project_name)
                    _isolate_worker_scratch(base_scratch, worker_scratch, iface.project_name)
                    iface.environment.molcas_scratch_dir = worker_scratch
                    run_final_calculation(iface, cas_occ, cas_idx, method)
                    return iface.orbital_file
                return job

            pergeom_jobs = [
                _make_pergeom_job(iface, cas_occ, cas_idx, configuration.cas_method)
                for iface, cas_occ, cas_idx in zip(interfaces, cas_occupations, cas_indices)
                if cas_occ
            ]
            active_ifaces = [iface for iface, cas_occ in zip(interfaces, cas_occupations) if cas_occ]
            pergeom_results = _run_parallel(pergeom_jobs, configuration.n_workers)
            for iface, orb_file in zip(active_ifaces, pergeom_results):
                iface.orbital_file = orb_file
        finally:
            FileHandler.DirectoryNames.final_calc = orig_final_name
            for molcas_iface, orig_orb in zip(interfaces, orig_orbital_files):
                molcas_iface.orbital_file = orig_orb
            os.chdir(prev_dir)

    print("*******************************************************************************************")
    print("*                                                                                         *")
    print("*                               Combined Active Spaces                                    *")
    print("*                                                                                         *")
    print("*******************************************************************************************")

    # Combine CAS
    combined_occupations, combined_indices = combine_active_spaces(cas_occupations, cas_indices, orbital_map)
    # Always include all unmappable orbitals if required.
    if configuration.unmappable:
        unmappable_occupied = unmappable_orbitals[0]
        unmappable_virtuals: List[List[int]] = []
        occ = 2
        if len(unmappable_orbitals) > 1:
            unmappable_virtuals = unmappable_orbitals[1]
        for cas_index, cas_occ, u_occ, u_virt in zip(combined_indices, combined_occupations, unmappable_occupied,
                                                     unmappable_virtuals):
            for i in u_occ:
                if i not in cas_index:
                    cas_index.append(i)
                    cas_occ.append(occ)
            for i in u_virt:
                if i not in cas_index:
                    cas_index.append(i)
                    cas_occ.append(0)

    if configuration.exclude_core:
        n_core_orbitals = molecules[0].core_orbitals
        for active_space, occupations in zip(combined_indices, combined_occupations):
            for i in range(n_core_orbitals):
                if i in active_space:
                    idx = active_space.index(i)
                    active_space.remove(i)
                    occupations.remove(occupations[idx])

    combined_file = open(os.path.join(_dmrg_dir, "combined_cas_spaces"), "w")
    for cas_index, cas_occ in zip(combined_indices, combined_occupations):
        print(f"combined cas indices: {cas_index}")
        print(f"combined occupation:  {cas_occ}")
        combined_file.write(f"combined cas indices: {cas_index}\n")
        combined_file.write(f"combined occupation:  {cas_occ}\n")
    combined_file.close()

    print("*******************************************************************************************")
    print("*                                                                                         *")
    print("*                               Final Calculations                                        *")
    print("*                                                                                         *")
    print("*******************************************************************************************")

    print(f"Running final CASSCF for {len(interfaces)} geometries ({configuration.n_workers} worker(s))")

    def _make_final_job(iface, cas_occ, cas_idx, method):
        base_scratch = iface.environment.molcas_scratch_dir
        def job():
            worker_scratch = os.path.join(base_scratch, "final_" + iface.project_name)
            _isolate_worker_scratch(base_scratch, worker_scratch, iface.project_name)
            iface.environment.molcas_scratch_dir = worker_scratch
            _, _, energy = run_final_calculation(iface, cas_occ, cas_idx, method)
            return energy, iface.orbital_file
        return job

    final_jobs = [
        _make_final_job(iface, cas_occ, cas_idx, configuration.cas_method)
        for iface, cas_occ, cas_idx in zip(interfaces, combined_occupations, combined_indices)
    ]
    final_results = _run_parallel(final_jobs, configuration.n_workers)
    energies: List[float] = []
    for i, (energy, orb_file) in enumerate(final_results):
        energies.append(energy)
        interfaces[i].orbital_file = orb_file

    os.makedirs(_final_dir, exist_ok=True)
    f = open(os.path.join(_final_dir, "energies.dat"), "w")
    for e in energies:
        f.write(str(e) + "\n")
    f.close()
    FileHandler.DirectoryNames.project_name = initial_project_name
    return combined_occupations, combined_indices, energies


def run_final_calculation(molcas: Molcas, cas_occ: List[int], cas_index: List[int], method: str)\
        -> Tuple[List[int], List[int], float]:
    """
    Run the final calculation with the given method.

    Parameters
    ----------
    molcas: Molcas
        The molcas interface.
    cas_occ: List[int]
        The CAS occupations.
    cas_index: List[int]
        The orbital indices for the CAS.
    method: str
        The method to use for the final calculation, e.g., "CASSCF", "CASCI", "DMRGCI", "DMRGSCF", or "CASPT2".

    Returns
    -------
    Tuple[List[int], List[int], float]
        The CAS occupations, CAS indices, and the final energy.
    """
    # cas and hyphen do not matter for method names
    if "PT2" in method:
        print("Running CASPT2")
        molcas.settings.post_cas_method = PostCasMethods.CASPT2
        molcas.settings.cas_method = CasMethods.CASSCF
    else:
        if method == "CASSCF":
            molcas.settings.cas_method = CasMethods.CASSCF
        elif method == "CASCI":
            molcas.settings.cas_method = CasMethods.CASCI
        elif method == "DMRGCI":
            molcas.settings.cas_method = CasMethods.DMRGCI
        elif method == "DMRGSCF":
            molcas.settings.cas_method = CasMethods.DMRGSCF
        else:
            raise NotImplementedError(f"Method: {method} is not available")

    molcas.settings.dmrg_bond_dimension = Defaults.Interface.dmrg_bond_dimension
    molcas.settings.dmrg_sweeps = Defaults.Interface.dmrg_sweeps
    # ensure interface is set correctly
    molcas.set_initial_cas_state(True)
    molcas.set_orbital_state(True)

    # Do a calculation with this CAS
    final_energy, final_s1, final_s2, final_mut_inf = molcas.calculate(cas_occ, cas_index)

    # use results
    n_electrons = sum(cas_occ)
    n_orbitals = len(cas_occ)
    print(f"final energy:      {final_energy}")
    print(f"final CAS(e, o):  ({n_electrons}, {n_orbitals})")
    print(f"final cas indices: {cas_index}")
    print(f"final occupation:  {cas_occ}")
    print(f"final s1:          {final_s1}")
    print(f"final s2: \n{final_s2}")
    print(f"final mut_inf: \n{final_mut_inf}")
    if len(final_energy) < 1:
        raise RuntimeError("Final energy calculation did not return a valid energy value.")
    return cas_occ, cas_index, final_energy[0]
