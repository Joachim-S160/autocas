# -*- coding: utf-8 -*-
__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details. """

import os.path
from typing import List, Dict, Optional
import yaml
from scine_autocas.utils.defaults import Defaults
from scine_autocas.io import FileHandler


class ConsistentActiveSpaceConfiguration:
    """
    Configuration class for the consistent active space workflow.
    """
    __slots__ = (
        "load_orbitals",
        "cas_method",
        "autocas_indices",
        "basis_set",
        "large_active_space",
        "exclude_core",
        "unmappable",
        "xyz_files",
        "system_names",
        "base_load_path",
        "project_name",
        "use_external_orbitals",
        "external_orbital_files",
        "localization_method",
        "force_cas",
        "skip_localization",
        "rasscf_sx_max_iter",
        "rasscf_level_shift",
        "restart_from_dmrg",
        "spin_multiplicity",
        "ibo_minao_basis",
        "save_per_geom_casscf",
        "n_workers",
    )

    def __init__(self):
        self.load_orbitals: bool = False
        """
        bool
            If true, the orbitals are loaded from the Serenity HDF5 files.
        """
        self.cas_method: str = "CASPT2"
        """
        str
            The post CAS method to be used. Default is "CASPT2".
        """
        self.autocas_indices: List[int] = []
        """
        List[int]
            The system indicies to run the autoCAS workflow on. If None, all systems are used.
        """
        self.basis_set: str = Defaults.Interface.basis_set
        """
        str
            The AO basis set set.
        """
        self.large_active_space: bool = False
        """
        bool
            If true, the large active space workflow is used.
        """
        self.exclude_core: bool = True
        """
        bool
            If true, core orbitals are always excluded from the active space.
        """
        self.unmappable: bool = False
        """
        bool
            If true, unmappable orbitals are always included in the active space.
        """
        self.xyz_files: List[str] = []
        """
        List[str]
            The structure files in XYZ format to be used for the workflow.
        """
        self.system_names: List[str] = []
        """
        List[str]
            The system names to be used for the workflow.
        """
        self.base_load_path: str = os.getcwd()
        """
        str
            The base path where the Serenity orbitals are loaded from. This is used when load_orbitals is True.
            By default, it is set to the current directory.
        """
        self.project_name: str = Defaults.DirName.project_name
        """
        str
            The project name. Serenity and Molcas files will be written to this directory.
        """
        self.use_external_orbitals: bool = False
        """
        bool
            If true, external orbital files (e.g., from OpenMolcas with DKH2) are used instead of
            running Serenity SCF. This is useful for heavy elements where relativistic effects
            are important.
        """
        self.external_orbital_files: List[str] = []
        """
        List[str]
            Paths to external orbital files (e.g., OpenMolcas .ScfOrb files) to load into Serenity.
            One file per system is required when use_external_orbitals is True.
        """
        self.localization_method: str = "IBO"
        """
        str
            The orbital localization method to use. Options are:
            - "IBO" (default): Intrinsic Bond Orbitals
            - "PIPEK_MEZEY": Pipek-Mezey localization
            - "BOYS": Foster-Boys localization
            - "EDMINSTON_RUEDENBERG": Edmiston-Ruedenberg localization
        """
        self.force_cas: bool = False
        """
        bool
            If true, force active space selection even when single-orbital entropies
            indicate a single-reference system (no multireference character).
            Useful for systems where the initial DMRG gives low entropies but an
            active space is still desired.
        """
        self.skip_localization: bool = False
        """
        bool
            If true, skip orbital localization entirely. Useful for debugging or when
            localization causes issues (e.g., NaN orbital coefficients with certain
            localization methods for heavy elements).
        """
        self.rasscf_sx_max_iter: Optional[int] = None
        """
        Optional[int]
            Override the max SX inner iterations per RASSCF macro-step (ITERations keyword,
            2nd value). None = use the Molcas.Settings default (currently 100).
            Set to a higher value (e.g. 150) to test whether more inner iterations help
            convergence at stretched geometries.
        """
        self.rasscf_level_shift: Optional[float] = None
        """
        Optional[float]
            Override the RASSCF LEVShift (Eh). None = use Molcas.Settings default (0.5 Eh).
            OpenMolcas undamped default is 0.3 Eh. Use 0.1 when DMRG orbitals are already
            a good starting point and 0.5 causes oscillation (e.g. near ionic/covalent
            crossing such as PbO at R=7.00 Ang).
        """
        self.restart_from_dmrg: bool = False
        """
        bool
            If True, skip orbital preparation, IBO localization, DMRG entropy selection,
            and CAS combination. Read combined_cas_spaces from the existing dmrg/ directory
            and run only the final CASSCF. Requires a previous complete run up to DMRG.
            Use with rasscf_level_shift to change convergence parameters on restart.
        """
        self.spin_multiplicity: int = 1
        """
        int
            The spin multiplicity (2S+1) of the molecule. Default is 1 (singlet).
            Use 3 for triplet, which is the ground state dissociation limit for
            heavy diatomics like PbO, Pb2, Po2 (Pb/Po both have 3P ground state).
        """
        self.ibo_minao_basis: str = "MINAO"
        """
        str
            Minimal basis for IAO/IBO construction. Default is "MINAO" (all occupied shells).
            For open-shell systems where nOcc_alpha > nMINAO/2, use MINAO1 (adds 1 extra shell
            per heavy atom of appropriate l-type), MINAO2 (+2 shells), or MINAO3 (+3 shells).
            These are generated by autoCAS4HE_priv/scripts/generate_minao_spin.py.
        """
        self.save_per_geom_casscf: bool = False
        """
        bool
            If True, run a CASSCF for each geometry using its own entropy-selected active
            space (before the union CAS is taken). Results are saved to pergeom/ alongside
            the existing final/ directory. Allows Pegamoid inspection at both pipeline
            stages: per-geometry CAS and union CAS.
        """
        self.n_workers: int = 1
        """
        int
            Number of parallel worker processes for the per-geometry loops (DMRG entropy,
            per-geom CASSCF, final CASSCF). Default 1 = serial. Uses forked subprocesses
            so each worker has its own CWD and memory — safe for os.chdir()-heavy code.
        """

    def write_yaml_file(self, file_name: str = "consistent_cas.configuration.yaml") -> str:
        """
        Write the configuration to a YAML file.

        Returns
        -------
        str
            The configuration is written to a file in the current directory.
            The filename is returned.
        """
        config_dict = {attr: getattr(self, attr) for attr in self.__slots__}
        with open(file_name, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        return file_name

    @staticmethod
    def input_sanity_checks(config) -> None:
        """
        Perform sanity checks on the input configuration and complete missing/incomplete fields.

        Parameters
        ----------
        config : ConsistentActiveSpaceConfiguration
            The configuration object to be checked and completed.
        """
        n_systems = max(len(config.xyz_files), len(config.system_names))
        if n_systems < 2:
            raise ValueError("At least two XYZ files or system names are required for the consistent active"
                             " space workflow.")
        if not config.autocas_indices:
            config.autocas_indices = list(range(n_systems))

        if max(config.autocas_indices) > n_systems - 1:
            raise ValueError("The autocas_indices must not exceed the number of XYZ files/system names minus"
                             " one.")
        if not config.system_names:
            config.system_names = [f"system_{i}" for i in range(n_systems)]
        if config.load_orbitals:
            config.xyz_files = [os.path.join(config.base_load_path, name, name + ".xyz")
                                for name in config.system_names]
        if len(config.xyz_files) != len(config.system_names):
            raise ValueError("The number of XYZ files must match the number of system names.")
        for p in config.xyz_files:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"The file {p} does not exist.")
        # Validate external orbital files if using external orbitals
        if config.use_external_orbitals:
            if not config.external_orbital_files:
                raise ValueError("External orbital files must be provided when use_external_orbitals is True.")
            if len(config.external_orbital_files) != n_systems:
                raise ValueError(f"Number of external orbital files ({len(config.external_orbital_files)}) "
                                 f"must match number of systems ({n_systems}).")
            for p in config.external_orbital_files:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"The external orbital file {p} does not exist.")

    @classmethod
    def from_options(cls, options, xyz_files: List[str]):
        """
        Create a configuration from command line options.

        Parameters
        ----------
        options
            The command line options.
        xyz_files : List[str]
            The list of XYZ files to be used in the workflow.

        Returns
        -------
        ConsistentActiveSpaceConfiguration
            The configuration created from the options.
        """
        config = cls()
        config.load_orbitals = options.load_orbitals
        config.cas_method = options.cas_method
        config.autocas_indices = [int(i) for i in options.autocas_molecule_indices.split()]
        config.basis_set = options.basis_set
        config.large_active_space = options.large_active_space
        config.exclude_core = options.exclude_core
        config.unmappable = options.always_include_unmapables
        config.xyz_files = [os.path.abspath(p) if p[0] != "/" else p for p in xyz_files]
        # Handle external orbitals option
        config.use_external_orbitals = options.use_external_orbitals
        if options.external_orbital_files:
            config.external_orbital_files = [os.path.abspath(p) if p[0] != "/" else p
                                              for p in options.external_orbital_files.split(",")]
        # Handle localization method option
        config.localization_method = options.localization_method
        # Handle force_cas option
        config.force_cas = options.force_cas
        # Handle skip_localization option
        config.skip_localization = options.skip_localization
        # Handle rasscf_sx_max_iter override (None = use Molcas.Settings default)
        config.rasscf_sx_max_iter = options.rasscf_sx_max_iter
        # Handle rasscf_level_shift override (None = use Molcas.Settings default)
        config.rasscf_level_shift = options.rasscf_level_shift
        # Handle restart_from_dmrg flag
        config.restart_from_dmrg = options.restart_from_dmrg
        # Handle spin multiplicity
        config.spin_multiplicity = options.spin_multiplicity
        # Handle IBO MINAO basis variant
        config.ibo_minao_basis = options.ibo_minao_basis
        # Handle save_per_geom_casscf flag
        config.save_per_geom_casscf = options.save_per_geom_casscf
        config.n_workers = options.n_workers
        ConsistentActiveSpaceConfiguration.input_sanity_checks(config)
        config.base_load_path = os.path.join(*config.xyz_files[0].split("/")[:-1])  # type: ignore
        return config

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the configuration from a file.

        Parameters
        ----------
        filename : str
            The path to the configuration file.

        Returns
        -------
        ConsistentActiveSpaceConfiguration
            The loaded configuration.
        """
        with open(filename, 'r') as file:
            configuration_dict = yaml.safe_load(file)
        config = cls()
        for key, value in configuration_dict.items():
            setattr(config, key, value)
        config.xyz_files = [os.path.abspath(p) if p[0] != "/" else p for p in config.xyz_files]
        ConsistentActiveSpaceConfiguration.input_sanity_checks(config)
        return config

    def get_serenity_interface_settings(self) -> Dict:
        """
        Get the Serenity interface settings for the consistent active space workflow.

        Returns
        -------
        Dict
            The settings for the Serenity interface.
        """
        # Virtual localization is only supported for IBO/IAO in Serenity
        # For PM, FB, ER we must disable it
        localize_virtuals = self.localization_method in ["IBO", "IAO"]
        if not localize_virtuals:
            print(f"  Note: Virtual localization disabled for {self.localization_method} "
                  "(only supported for IBO/IAO)")
        settings = {
            "Interface": {
                # UHF only when Serenity runs its own SCF. For external (ROHF) orbitals,
                # keep restricted IBO throughout — restricted→restricted write-back is consistent.
                "uhf": self.spin_multiplicity != 1 and not self.use_external_orbitals,
                "localisation_method": self.localization_method,
                "alignment": True,
                "localize_virtuals": localize_virtuals,
                "optimized_mapping": True,
                "work_dir": "serenity/",
                "basis_set": self.basis_set,
                "score_start": 1.0,
                "skip_localization": self.skip_localization,
                "system_names": self.system_names,
                "molcas_orbital_files": [FileHandler.get_project_path() + "/initial/" for _ in self.system_names]
            }
        }
        if self.xyz_files:
            settings["Interface"]["xyz_files"] = self.xyz_files
        if self.ibo_minao_basis != "MINAO":
            settings["Interface"]["ibo_minao_basis"] = self.ibo_minao_basis
        # Add external orbital settings
        if self.use_external_orbitals:
            settings["Interface"]["use_external_orbitals"] = True
            settings["Interface"]["external_orbital_files"] = self.external_orbital_files
        return settings

    def get_serenity_load_paths(self) -> List[str]:
        """
        Get the paths to load Serenity orbitals from.

        Returns
        -------
        List[str]
            The list of paths to load Serenity orbitals from.
        """
        return [self.base_load_path for _ in range(len(self.system_names))]
