# -*- coding: utf-8 -*-
__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details. """

import os
import shutil
from typing import List, Optional, Tuple
from scine_autocas.workflows.consistent_active_space.configuration import ConsistentActiveSpaceConfiguration
from scine_autocas import Molecule
from scine_autocas.interfaces import Molcas, Serenity

from scine_autocas.utils.defaults import Defaults


def setup_molcas_and_molecule(xyz_file: str, name: str, basis_set: str,
                               spin_multiplicity: int = 1,
                               external_orbital_file: Optional[str] = None) -> Tuple[Molcas, Molecule]:
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
        Path to an existing .scf.h5 file. When provided, IBO still runs (SEWARD is
        required to generate the RunFile for DMRG), but the resulting orbital file is
        replaced with this external file before the DMRG step.

    Returns
    -------
    Tuple[Molcas, Molecule]
        The Molcas interface and the Molecule object.
    """
    # create a molecule
    molecule = Molecule(xyz_file, spin_multiplicity=spin_multiplicity)
    # initialize autoCAS and Molcas interface
    molcas = Molcas(molecule)
    # setup interface
    molcas.project_name = name
    molcas.settings.basis_set = basis_set
    molcas.set_cas_method("dmrgci")
    molcas.settings.dmrg_bond_dimension = Defaults.Interface.init_dmrg_bond_dimension
    molcas.settings.dmrg_sweeps = Defaults.Interface.init_dmrg_sweeps

    if external_orbital_file is not None and os.path.exists(external_orbital_file):
        # Run IBO (GATEWAY+SEWARD+SCF) to generate the RunFile and integrals in scratch —
        # DMRG requires system_N.RunFile which is only created by SEWARD.
        molcas.calculate()
        # Replace the IBO-generated orbital file with the external (Serenity) one.
        shutil.copy2(external_orbital_file, molcas.orbital_file)
        molcas.hdf5_utils.read_hdf5(molcas.orbital_file)
        print(f"  [external orbitals] {name}: replaced IBO orbitals with {os.path.basename(external_orbital_file)}")
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
            external_orbital_file=ext_orb
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
