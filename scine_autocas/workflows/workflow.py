# -*- coding: utf-8 -*-
__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details. """
from typing import Any, Dict, Optional

from scine_autocas.cas_selection import Autocas
from scine_autocas.interfaces.interface import Interface
from scine_autocas.io import FileHandler, logger
from scine_autocas.plots import EntanglementPlot, ThresholdPlot
from scine_autocas.utils.exceptions import SingleReferenceException


class Workflow:
    """Base worfklow class"""

    def __init__(self, autocas: Autocas, interface: Optional[Interface] = None):
        """Base workflow

        Parameters
        ----------
        autocas : Autocas
            initialized autocas object
        interface : Interface, default = None
            set the interface if something has to be calculated
        """
        self.autocas: Autocas = autocas
        """autocas object"""
        self.interface: Optional[Interface] = None
        """Interface object"""
        if interface:
            self.interface = interface

        self.results: Dict[str, Any] = {}
        """Save results here"""

    def run(self):
        """Run the workflow"""
        assert self.interface
        print(f"Start {self.__class__.__name__}")
        self._initial_orbitals_impl()
        try:
            tmp_cas = self.interface.settings.cas_method
            try:
                tmp_post_cas = self.interface.settings.post_cas_method
            except AttributeError:
                tmp_post_cas = None
            self.interface.set_cas_method("dmrgci")
            self.interface.settings.post_cas_method = None

            self._initial_dmrg_impl()
            self.interface.settings.cas_method = tmp_cas
            self.interface.settings.post_cas_method = tmp_post_cas
        except SingleReferenceException:
            self.results["final_orbital_indices"] = None
            self.results["final_occupation"] = None
            self.plotting()
        self._final_calc_impl()
        self.print_results()

    def _initial_orbitals_impl(self) -> None:
        """Run initial orbital calculation."""
        assert self.interface
        logger.frame("Initial orbital calculation")
        FileHandler.make_initial_orbital_dir()
        # get initial orbitals, e.g. RHF/UHF
        self.interface.calculate()
        logger.empty_line()

        # get valence space
        logger.frame("Initial CAS")
        (
            initial_occupation, initial_orbital_indices,
        ) = self.autocas.make_initial_active_space()
        logger.init_cas(initial_occupation, initial_orbital_indices)
        logger.empty_line()

        self.results["initial_occupation"] = initial_occupation
        self.results["initial_orbital_indices"] = initial_orbital_indices

    def _initial_dmrg_impl(self) -> None:
        raise NotImplementedError("""
            <Workflow> is just the base class.

            In order to create custom workflows, inherite from this class
            and implement this function. """)

    def _final_calc_impl(self) -> None:
        assert self.interface
        logger.frame("Final calculation")
        FileHandler.make_final_calc_dir()
        final_energy = self.interface.calculate(
            self.results["final_occupation"], self.results["final_orbital_indices"]
        )[0]
        self.results["final_energy"] = final_energy
        logger.empty_line()

    def plotting(self) -> None:
        """Plot entanglement and threshold"""
        print("Plotting entanglement diagram from initial DMRG calculation")
        entanglement_plot = EntanglementPlot()
        plt = entanglement_plot.plot(
            self.results["initial_s1"],
            self.results["initial_mutual_information"],
            self.results["initial_orbital_indices"],
        )
        if FileHandler.check_project_dir_exists():
            entang_diag_path = FileHandler.get_project_path() + f"/{FileHandler.PlotNames.entanglement_file}"
            print(f"Plotting in {entang_diag_path}")
            plt.savefig(entang_diag_path)  # type: ignore
        else:
            print("Disable plotting, since no project folder was set")
        # reset plot
        plt.clf()  # type: ignore
        print("Plotting threshold diagram from initial DMRG calculation")
        threshold_plot = ThresholdPlot()
        thresh_plt = threshold_plot.plot(self.results["initial_s1"])
        if FileHandler.check_project_dir_exists():
            threh_diag_path = FileHandler.get_project_path() + f"/{FileHandler.PlotNames.threshold_file}"
            print(f"Plotting in {threh_diag_path}")
            thresh_plt.savefig(threh_diag_path)  # type: ignore
        else:
            print("Disable plotting, since no project folder was set")
        # IBO distribution plot
        self._plot_ibo_distribution()

    def _plot_ibo_distribution(self) -> None:
        """Generate IBO orbital distribution plots using IAO-constrained classification.

        Shows the proper IAO constraint: nValVirt = nMINAO - nOcc
        """
        if not hasattr(self.interface, 'systems') or not self.interface.systems:
            return
        if not FileHandler.check_project_dir_exists():
            return

        import subprocess
        import sys
        import shutil
        from pathlib import Path

        sys_zero = self.interface.systems[0]
        sys_name = sys_zero.getSystemName()
        sys_path = sys_zero.getSettings().path
        h5file = Path(sys_path) / sys_name / f"{sys_name}.scf.h5"

        if not h5file.exists():
            return

        # Use the IAO-constrained script
        script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "IBO_distr_IAO.py"
        if not script_path.exists():
            # Fallback to old script if new one not found
            script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "IBO_distr.py"
            if not script_path.exists():
                return

        print("Plotting IBO orbital distribution (IAO-constrained)")
        try:
            subprocess.run(
                [sys.executable, str(script_path), str(h5file)],
                cwd=str(h5file.parent), check=False, capture_output=True
            )
            project_path = FileHandler.get_project_path()
            # Look for both new and old format plot names
            for pdf in h5file.parent.glob("*_IBO*.pdf"):
                if "proposed" in pdf.name:
                    dest = f"{project_path}/{FileHandler.PlotNames.ibo_distribution_proposed_file}"
                else:
                    dest = f"{project_path}/{FileHandler.PlotNames.ibo_distribution_file}"
                shutil.copy2(pdf, dest)
                print(f"  Saved: {dest}")
        except Exception as e:
            print(f"[WARNING] IBO distribution analysis failed: {e}")

    def print_results(self) -> None:
        """Print all results"""
        logger.frame("Results")
        for key, result in self.results.items():
            print(f"{key}:")
            print(f"{result}")
            logger.empty_line()
