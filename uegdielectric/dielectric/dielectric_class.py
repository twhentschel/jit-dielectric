"""Classes to compute the dielectric function of an electron gas"""

from abc import ABC, abstractmethod

from uegdielectric.dielectric._Mermin_integrals import MerminDielectric


class AbstractDielectric(ABC):
    """An abstract base class that defines the interface for the dielectric
    function."""

    @abstractmethod
    def dielectric(self, q, omega):
        """Compute the dielectric function"""
        pass


class Mermin(AbstractDielectric):
    """
    A class for the Mermin dielecric model.

    Parameters:
    ___________
    electrongas : |ElectronGas|
        |ElectronGas| instance.
    collfreq : float or function, optional
        Electron-ion collision frequency, which depends on the material we
        are modelling as an electron gas. Can be a function of the frequency
        of the pertubation. If it is a function, it must have only one argument that is of type float and be defined for all nonegative numbers.
    """

    def __init__(
        self,
        electrongas,
        collfreq=None
    ):
        self._electrongas = electrongas
        if collfreq is None:
            self._collfreq = lambda x: 0
        elif isinstance(collfreq, float):
            self._collfreq = lambda x: collfreq
        else:
            self._collfreq = collfreq

    def dielectric(
            self,
            q,
            omega):
        """
        The dielectric function of the electron gas as a function of the wave
        number (q) and frequency (omega) modes excited by a perturbation
        If the electron-ion collision frequency (collfreq) is not supplied, this
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
        q: array_like of real values
            The spatial frequency of the perturbation acting on the electron
            gas. Units are a.u. or units of 1/a0, where
            a0 = Bohr radius = 0.529 Angstrom.
        omega: array_like of real values
            Temporal frequency of the perturbation acting on the electron
            gas. Units are a.u. or units of Ha, where
            Ha = Hartree energy = 27.2114 eV.

        Returns:
        ________
        ret: ndarray of complex values
            If q and omega are both 1D arrays, shape will be
            (size(q), size(omega)). Otherwise, if only one of these arguments is
            a 1D array of size n and the other is a scalar, the shape is
            (size(n),). If both arguments are scalars, the result is a complex
            scalar as well.
        """
        ret = MerminDielectric(
            q,
            omega,
            self._collfreq(omega),
            self._electrongas.temperature(),
            self._electrongas.chempot(),
            self._electrongas.dosratio()
        )

        return ret

    @property
    def collfreq(self):
        """
        The electron-ion collision frequency of the system, in atomic units.
        """
        return self._collfreq

    @collfreq.setter
    def collfreq(self, collisions):
        if collisions is None:
            self._collfreq = lambda x: 0
        elif isinstance(collisions, float):
            self._collfreq = lambda x: collisions
        else:
            self._collfreq = collisions


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
     In the collective limit (for example, for small wavenumbers), it can be helpful to damp the RPA response with a small collision frequency to avoid numerical issues. How small is up to you (the larger the value, the less RPA-like the calculation), but a good first attempt is
     0.03675 ~ 1/27.2114 = 1/Ha, where Ha = Hartree energy = 27.2114 eV. The following example shows how this can be done.

     >>> RPA.collfreq(1/27.2114)
    """

    def __init__(self, argument):
        super().__init__(argument)

    def dielectric(self, q, omega):
        return super().dielectric(q, omega)
