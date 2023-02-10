""" A class to represent the electron gas."""

import warnings
import fdint
import numpy as np
from electrondielectric.dielectric.dielectric import MerminDielectric

class ElectronGas:
    """
    A class for an electron gas.
    
    Parameters
    ___________
    temperature: float
        The thermal energy (kB*T: kB - Boltzmann's constant, T is
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
    def __init__(
        self,
        temperature,
        density,
        DOSratio = None,
        chemicalpot = None
    ):

        self._temp = temperature
        self._density = density
        self._dosratio = DOSratio
    
        
        if chemicalpot is None:
            if self._dosratio is not None:
                warnings.warn("DOSratio is given but not chemicalpot. "\
                    + "Will use the chemical potential computed using the "\
                    + "ideal DOS.")
            # compute chemical potential using ideal DOS
            self._chempot = fdint.ifd1h(2 * np.pi**2 * self._density \
                                        / (2*self._temp)**(3/2)) * self._temp
        else:
            self._chempot = chemicalpot
    
    @property
    def temperature(self):
        """
        The temperature (thermal energy) of the electron gas, in atomic units.
        """
        return self._temp
    
    @property
    def density(self):
        """
        The density of the electron gas, in atomic units.
        """
        return self._density
    
    @property
    def chempot(self):
        """
        The chemical potential of the electron gas, in atomic units.
        """
        return self._chempot

    @property
    def dosratio(self):
        """
        Ratio of the nonideal density of states (DOS) and the ideal DOS.
        This is a function of the electronic momentum. In atomic units,
        this is related to the energy by momentum = sqrt(2*energy).
        """
        return self._dosratio
    
    def dielectric(self, q, omega, collfreq=None):
        """
        The dielectric function of the electron gas as a function of the spatial
        (q) and temporal (omega) frequency modes excited by a perturbation. If
        the electron-ion collision frequency (collfreq) is not supplied, this
        returns the random phase approximation (RPA) dielectric function. If a
        collision frequency is used, this returns the Mermin dielectric
        function.
        
        The uniform electron gas is a theoretical creation that does not exist
        in nature, although there are some metals (like aluminum, for example)
        whose electrons almost behave like an free/ideal electron gas. The goal
        of including a collision frequency parameter is to improve the electron
        gas description to better predict the behavior of true materials that
        have ions present. As such, the collision frequency is a material-
        dependent quantity. It can also be (temporal) frequency dependent (that
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
        collfreq: array_like of real values
            The collision frequency (inverse of the collision rate) between the
            electrons and ions. If 1D array, must be same size as omega.
            
        Returns:
        ________
        ret: ndarray of complex values
            If q and omega are both 1D arrays, shape will be
            (size(q), size(omega)). Otherwise, if only one of these arguments is
            a 1D array of size n and the other is a scalar, the shape is
            (size(n),). If both arguments are scalars, the result is a complex
            scalar as well.
        """
        if collfreq is None:
            collfreq = 0.
            
        return MerminDielectric(
            q,
            omega,
            collfreq,
            self.temperature(),
            self.chempot(),
            self.dosratio()
        )

    # def dielectric(self, momentum, energy, collfreq=0):
    #     """
    #     Returns the dielectric function for the (free) electron gas using the
    #     (RPA) Mermin approximatio, as a function of the momentum (spatial
    #     frequency) and energy (temporal frequency) of the perturbation (these
    #     are typically denoted by :math:`k, \omega`). In the dielectric theory,
    #     which is linear, this perturbation excites the same frequencies in the
    #     dielectric response.
        
    #     For the Mermin dielectric, the collfreq argument represents the
    #     electron-ion collision frequency and is incorporated as a relaxation
    #     parameter.

    #     Parameters:
    #     ___________
    #     momentum: float
    #         :math:`\hbar k`, where :math:`k` is the spatial frequency of
    #         perturbation, in atomic units (a.u.) or units of 1/a0, where
    #         a0 = Bohr radius = 0.529 Angstrom.
    #     energy: array-like or float
    #         :math:`\hbar \omega`, where :math:`\omega` is the (angular) temporal
    #         frequency of the perturbation, in a.u. or units of Ha, where
    #         Ha = Hartree energy = 27.2114 eV.
    #     collfreq: array-like or float
    #         The electron-ion collision frequency that modifies the free electron
    #         approximation (which neglects electron-ion interactions). If
    #         array-like, must be same length at energy. In a.u. or units of Ha.
    #         Default value is 0.
    #     """
    #     return MerminDielectric.MerminDielectric(momentum,
    #                                              energy,
    #                                              collfreq,
    #                                              self.temp,
    #                                              self.chempot,
    #                                              self.dosratio)
    
    # def electronloss(self, momentum, energy, collfreq=0):
    #     """
    #     Returns the electron loss function for the (free) electron gas using the
    #     (RPA) Mermin approximatio, as a function of the momentum (spatial
    #     frequency) and energy (temporal frequency) of the perturbation (these
    #     are typically denoted by :math:`k, \omega`).
        
    #     For the Mermin dielectric, the collfreq argument represents the
    #     electron-ion collision frequency and is incorporated as a relaxation
    #     parameter.

    #     Parameters:
    #     ___________
    #     momentum: float
    #         :math:`\hbar k`, where :math:`k` is the spatial frequency of
    #         perturbation, in atomic units (a.u.) or units of 1/a0, where
    #         a0 = Bohr radius = 0.529 Angstrom.
    #     energy: array-like or float
    #         :math:`\hbar \omega`, where :math:`\omega` is the (angular) temporal
    #         frequency of the perturbation, in a.u. or units of Ha, where
    #         Ha = Hartree energy = 27.2114 eV.
    #     collfreq: array-like or float
    #         The electron-ion collision frequency that modifies the free electron
    #         approximation (which neglects electron-ion interactions). If
    #         array-like, must be same length at energy. In a.u. or units of Ha.
    #         Default value is 0.
    #     """
    #     return MerminDielectric.ELF(momentum,
    #                                 energy,
    #                                 collfreq,
    #                                 self.temp,
    #                                 self.chempot,
    #                                 self.dosratio)
    
    # def stoppingpower(self, velocity, charge):
    #     return 0

