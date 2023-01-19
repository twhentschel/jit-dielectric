""" A class to represent the electron gas."""

import warnings
import fdint
import numpy as np

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
    DOSratio: function, default None
        Ratio of the nonideal density of states (DOS) and the ideal DOS.
        This is a function of the electronic momentum. In atomic units,
        this is related to the energy by momentum = sqrt(2*energy).
    chemicalpot: float, default None
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

        self.temp = temperature
        self.density = density
        self.dosratio = DOSratio
    
        
        if chemicalpot is None:
            if DOSratio is not None:
                warnings.warn("DOSratio is given but not chemicalpot. "\
                    + "Will use the chemical potential computed using the "\
                    + "ideal DOS.")
            # compute chemical potential using ideal DOS
            self.chempot = fdint.ifd1h(2 * np.pi**2 * self.density \
                                       / (2*self.temp)**(3/2)) * self.temp
        else:
            self.chempot = chemicalpot
            

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

