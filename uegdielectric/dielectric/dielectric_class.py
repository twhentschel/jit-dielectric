"""Classes to compute the dielectric function of an electron gas"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from numbers import Number
from numpy.typing import ArrayLike

from uegdielectric.dielectric._Mermin_integrals import MerminDielectric
from uegdielectric.electrongas import ElectronGas


class AbstractDielectric(ABC):
    """An abstract base class that defines the interface for the dielectric
    function."""

    @abstractmethod
    def __call__(
        self, wavenum: ArrayLike, frequency: ArrayLike, modifier: Callable = None
    ) -> ArrayLike:
        """Compute the dielectric function.

        This is a function of the wave number `wavenum` and the frequency `frequency`
        of an external perturbation to the electron density. `modifier` is a function
        that can be used to include wave number or frequency parameters within the
        dielectric function.
        """
        pass


@dataclass
class Mermin(AbstractDielectric):
    """
    A class for the Mermin dielecric model.

    Parameters:
    ___________
    electronparams : `ElectronGas`
        `ElectronGas` instance.
    """

    electronparams: ElectronGas

    def __call__(
        self,
        wavenum: ArrayLike,
        frequency: ArrayLike,
        collisionrate: Callable[[int | float], Number] = None,
    ) -> ArrayLike:
        """
        The dielectric function of the electron gas as a function of the wave
        number (`wavenum`) and frequency (`frequency`) modes excited by a perturbation
        If the electron-ion collision rate (`collisionrate`) is not supplied, this
        returns the random phase approximation (RPA) dielectric function. If a
        collision frequency is used, this returns the Mermin dielectric
        function.

        The uniform electron gas is a theoretical creation that does not exist
        in nature, although there are some metals (like aluminum, for example)
        whose electrons almost behave like an free/ideal electron gas. The goal
        of including a collision frequency parameter is to improve the electron
        gas description to better predict the behavior of true materials that
        have ions present. As such, the collision frequency is a material-
        dependent quantity. It can also be frequency dependent (that
        is, dependent on omega).

        All parameters are in atomic units (a.u.).

        Parameters:
        ___________
        wavenum: array_like of real values
            The wave number of the perturbation acting on the electron
            gas. Units are a.u. or units of 1/a0, where
            a0 = Bohr radius = 0.529 Angstrom.
        frequency: array_like of real values
            Temporal frequency of the perturbation acting on the electron
            gas. Units are a.u. or units of Ha, where Ha = Hartree energy = 27.2114 eV.
        collisionrate : Callable, optional
            Electron-ion collision rate, which depends on the material we
            are modelling as an electron gas. Assumed to be function of the frequency
            of the pertubation.

        Returns:
        ________
        ret: ndarray of complex values
            If `wavenum` and `frequency` are both 1D arrays, shape will be
            (size(wavenum), size(frequency)). Otherwise, if only one of these arguments
            is a 1D array of size n and the other is a scalar, the shape is
            (size(n),). If both arguments are scalars, the result is a complex
            scalar as well.
        """
        # Default `collisionrate` is a function that returns 0.
        if collisionrate is None:
            collisionrate = lambda x: 0.0

        ret = MerminDielectric(
            wavenum,
            frequency,
            collisionrate(frequency),
            self.electronparams.temperature,
            self.electronparams.chemicalpot,
            self.electronparams.DOSratio,
        )

        return ret


class RPA(Mermin):
    """Class for the RPA dielectric model.

    The RPA model is equivalent to the Mermin dielectric model with
    collfreq = 0.

    Parameters:
    ___________
    argument : |ElectronGas|
        |ElectronGas| instance.

    Notes:
    ______
     In the collective limit (for example, for small wavenumbers), it can be helpful to
     dampen the RPA response with a small collision rate to avoid numerical
     issues. How small is up to you (the larger the value, the less RPA-like the
     calculation), but a good first attempt is
     0.03675 ~ 1/27.2114 = 1/Ha, where Ha = Hartree energy = 27.2114 eV.
    """
