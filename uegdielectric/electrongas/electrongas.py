""" A class to represent the electron gas."""

import warnings
from dataclasses import dataclass
from typing import Callable

from uegdielectric.electrongas._inv_fermi_integral import inv_fdint_onehalf as ifdint


@dataclass
class ElectronGas:
    """
    A class for an electron gas.

    Parameters
    ___________
    temperature: float
        The thermal energy (kB*T: kB - Boltzmann constant, T is
        temperature) of the electron gas, in atomic units (a.u.) or units
        of Ha, where Ha = Hartree energy = 27.2114 eV.
    density: float
        Electron density of the electron gas, in a.u. or units of 1/a0^3,
        where a0 = Bohr radius = 0.529 Angstrom.
    DOSratio: function, optional
        Ratio of the nonideal density of states (DOS) and the ideal DOS.
        This is a function of the electronic momentum. In atomic units,
        this is related to the energy by momentum = sqrt(2*energy).
    chemicalpot: float, optional
        The chemical potential of the electron gas in a.u.

    Notes
    _____
    If dosratio is not supplied, the assumption is that the density of states
    of this electron gas is ideal, i.e. that this is a non-interacting/ideal
    Fermi (electron) gas.

    The chemical potential is automatically calculated assuming an ideal DOS
    using the density. If DOSratio is not None, a warning is raised if
    chemicalpot is None, as there might be inconsitencies with the density
    argument and the chemical potential that is computed from the density.
    """

    temperature: float
    density: float
    DOSratio: Callable = None
    chemicalpot: float = None

    def __post_init__(self):
        """Compute the chemical potential if not provided."""
        if self.chemicalpot is None:
            if self.DOSratio is not None:
                warnmssg = (
                    "`DOSratio` is given but not `chemicalpot`. Will use the "
                    + "chemical potential computed using the ideal Density of States."
                )
                warnings.warn(warnmssg, category=RuntimeWarning)
            # compute chemical potential using ideal DOS
            self.chemicalpot = ifdint(self.density, self.temperature)
