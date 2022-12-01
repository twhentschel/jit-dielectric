import MerminDielectric

class ElectronGas:
    """
    A simple description of the uniform electron gas  (UEG) at finite
    temperatures.

    This code follows the dielectric formalism for describing the UEG, and from
    there calculates two related quantities:
    1. The dynamic structure factor : structurefactor()
       This quantity is closely related to the intensity of scattered photons
       from a material. By comparing the results from this model to experimental
       scattering spectra in the scattering regime where the signal from the
       free-electrons dominate (also known as the collective regime),
       quantities like the temperature and density can be inferred. Here, the
       assumption is that the free-electrons can be treated approximattely as a
       unifrom electron gas.
    2. The charged particle stopping power : stoppingpower()
        This quantity describes the retarding force a charged particle feels as
        is travels through a material. It depends on the velocity and charge
        of the incoming particle, and also the temperature and density of the
        target material. This model returns the electronic stopping power,
        assuming that the electrons behave primarily like a uniform electron
        gas.

    To compute the dielectric function, we rely on two main approximations. The
    first is the random phase approximation (RPA),
    the second builds upon
    the RPA by incorporating a dynamic electron-ion collision frequency. The
    collision frequency describes how the electron density-response is modified
    (in a relaxation-time approximation picture) in the presence of the ions in
    our sample.
    """
    def __init__(self, temperature, chemicalpot, density, DOSratio=None):
        """
        Parameters:
        ___________
        temperature: float
            The thermal energy (kB*T: k - Boltzmann's constant, T is
            temperature) of the electron gas, in atomic units (a.u.) or units
            of Ha, where Ha = Hartree energy = 27.2114 eV.
        chemicalpotential: float
            Chemical potential of the electron gas in a.u. (units of 1/Ha).
        density: float
            Electron density of the electron gas, in a.u. or units of 1/a0^3,
            where a0 = Bohr radius = 0.529 Angstrom.
        dosratio: function, default None
            Ratio of the nonideal density of states (DOS) and the ideal DOS.
            This is a function of the electronic momentum. In atomic units,
            this is related to the energy by momentum = sqrt(2*energy).
        """
        self.temp = temperature
        self.chempot = chemicalpot
        self.density = density
        self.dosratio = DOSratio

    def dielectric(self, momentum, energy, collfreq=0):
        """
        Returns the dielectric function for the (free) electron gas using the
        (RPA) Mermin approximatio, as a function of the momentum (spatial
        frequency) and energy (temporal frequency) of the perturbation (these
        are typically denoted by :math:`k, \omega`). In the dielectric theory,
        which is linear, this perturbation excites the same frequencies in the
        dielectric response.
        
        For the Mermin dielectric, the collfreq argument represents the
        electron-ion collision frequency and is incorporated as a relaxation
        parameter.

        Parameters:
        ___________
        momentum: float
            :math:`\hbar k`, where :math:`k` is the spatial frequency of
            perturbation, in atomic units (a.u.) or units of 1/a0, where
            a0 = Bohr radius = 0.529 Angstrom.
        energy: array-like or float
            :math:`\hbar \omega`, where :math:`\omega` is the (angular) temporal
            frequency of the perturbation, in a.u. or units of Ha, where
            Ha = Hartree energy = 27.2114 eV.
        collfreq: array-like or float
            The electron-ion collision frequency that modifies the free electron
            approximation (which neglects electron-ion interactions). If
            array-like, must be same length at energy. In a.u. or units of Ha.
            Default value is 0.
        """
        return MerminDielectric.MerminDielectric(momentum,
                                                 energy,
                                                 collfreq,
                                                 self.temp,
                                                 self.chempot,
                                                 self.dosratio)
    
    def electronloss(self, momentum, energy, collfreq=0):
        """
        Returns the electron loss function for the (free) electron gas using the
        (RPA) Mermin approximatio, as a function of the momentum (spatial
        frequency) and energy (temporal frequency) of the perturbation (these
        are typically denoted by :math:`k, \omega`).
        
        For the Mermin dielectric, the collfreq argument represents the
        electron-ion collision frequency and is incorporated as a relaxation
        parameter.

        Parameters:
        ___________
        momentum: float
            :math:`\hbar k`, where :math:`k` is the spatial frequency of
            perturbation, in atomic units (a.u.) or units of 1/a0, where
            a0 = Bohr radius = 0.529 Angstrom.
        energy: array-like or float
            :math:`\hbar \omega`, where :math:`\omega` is the (angular) temporal
            frequency of the perturbation, in a.u. or units of Ha, where
            Ha = Hartree energy = 27.2114 eV.
        collfreq: array-like or float
            The electron-ion collision frequency that modifies the free electron
            approximation (which neglects electron-ion interactions). If
            array-like, must be same length at energy. In a.u. or units of Ha.
            Default value is 0.
        """
        return MerminDielectric.ELF(momentum,
                                    energy,
                                    collfreq,
                                    self.temp,
                                    self.chempot,
                                    self.dosratio)
    
    def stoppingpower(self, velocity, charge):
        return 0

