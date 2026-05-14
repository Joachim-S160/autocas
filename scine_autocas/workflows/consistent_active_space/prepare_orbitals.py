# -*- coding: utf-8 -*-
__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details. """

import os
from typing import List, Optional, Tuple
from scine_autocas.workflows.consistent_active_space.configuration import ConsistentActiveSpaceConfiguration
from scine_autocas import Molecule
from scine_autocas.interfaces import Molcas, Serenity

from scine_autocas.utils.defaults import Defaults


def setup_molcas_and_molecule(xyz_file: str, name: str, basis_set: str,
                               spin_multiplicity: int = 1,
                               external_orbital_file: Optional[str] = None,
                               relativistic: str = Defaults.Interface.relativistic) -> Tuple[Molcas, Molecule]:
    """
    Setup Molcas interface and molecule from an XYZ file.

    Parameters
    ----------
    xyz_file: str
        The path to the XYZ file containing the molecule.
    name: str
        The system name.
    basis_set: str
        The AO basis set.
    spin_multiplicity: int
        Spin multiplicity of the molecule.
    external_orbital_file: str, optional
        Path to an existing .scf.h5 file. When provided, the OpenMolcas &SCF block is
        skipped (only &GATEWAY + &SEWARD run to produce the RunFile); the supplied file
        is used as-is for the subsequent DMRG/RASSCF steps via FILEORB.
    relativistic: str
        Relativistic Hamiltonian string passed to &SEWARD (default "R02O" for DKH2).
        Pass empty string or "NONE" to disable.

    Returns
    -------
    Tuple[Molcas, Molecule]
        The Molcas interface and the Molecule object.
    """
    molecule = Molecule(xyz_file, spin_multiplicity=spin_multiplicity)
    molcas = Molcas(molecule)
    molcas.project_name = name
    molcas.settings.basis_set = basis_set
    molcas.set_cas_method("dmrgci")
    molcas.settings.dmrg_bond_dimension = Defaults.Interface.init_dmrg_bond_dimension
    molcas.settings.dmrg_sweeps = Defaults.Interface.init_dmrg_sweeps
    molcas.settings.relativistic = relativistic

    if external_orbital_file is not None and os.path.exists(external_orbital_file):
        # SCF-less init: GATEWAY + SEWARD only — DMRG/RASSCF reads user orbitals via FILEORB.
        # Running the full SCF and then overwriting the result is wasted compute.
        molcas.settings.skip_scf_block = True
        molcas.calculate()
        molcas.orbital_file = os.path.abspath(external_orbital_file)
        molcas.hdf5_utils.read_hdf5(molcas.orbital_file)
        print(f"  [external orbitals] {name}: SCF-less init; using {os.path.basename(external_orbital_file)}")
    else:
        molcas.calculate()
    return molcas, molecule


def print_orbital_map(orbital_map: List[List[List[int]]]) -> None:
    """
    Print the orbital map in a readable format.

    Parameters
    ----------
    orbital_map: List[List[List[int]]]
        The orbital map.
    """
    print("Orbital groups")
    for group in orbital_map:
        n_orbitals = len(group[0])
        to_out = ''
        for i_orb in range(n_orbitals):
            for idx in group:
                orb = idx[i_orb]
                to_out += f'{orb:8}  '
            to_out += '\n'
        print(to_out)


def construct_molecules(configuration: ConsistentActiveSpaceConfiguration,
                        external_orbital_files: Optional[List[str]] = None) -> Tuple[List[Molcas], List[Molecule]]:
    """
    Construct the Molcas interfaces and Molecule objects from the configuration.

    Parameters
    ----------
    configuration: ConsistentActiveSpaceConfiguration
        The calculation configuration.
    external_orbital_files: list of str, optional
        Pre-existing .scf.h5 files (one per geometry). When provided, the IBO pymolcas
        step is skipped for each geometry that has a matching file.

    Returns
    -------
    Tuple[List[Molcas], List[Molecule]]
        A tuple containing the list of Molcas interfaces and the list of Molecule objects.
    """
    interfaces = []
    molecules = []
    for i, (xyz, name) in enumerate(zip(configuration.xyz_files, configuration.system_names)):
        ext_orb = external_orbital_files[i] if (external_orbital_files and i < len(external_orbital_files)) else None
        molcas, molecule = setup_molcas_and_molecule(
            xyz, name, configuration.basis_set, configuration.spin_multiplicity,
            external_orbital_file=ext_orb,
            relativistic=configuration.relativistic,
        )
        molecules.append(molecule)
        interfaces.append(molcas)
    return interfaces, molecules


def load_orbitals_from_serenity(load_paths: List[str], settings: dict):
    """
    Load orbitals from Serenity and return the Serenity object and the list of molecules.

    Parameters
    ----------
    load_paths: List[str]
        The paths to load the orbitals/systems from.
    settings: dict
        The serenity interface settings.

    Returns
    -------
    Tuple[Serenity, List[Molecule]]
        A tuple containing the Serenity object and the list of Molecule objects.
    """
    serenity = Serenity([], settings, load_paths)
    serenity.settings.molcas_orbital_files = settings["Interface"]["molcas_orbital_files"]
    molecules = serenity.molecules
    return serenity, molecules
