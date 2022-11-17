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
    def __init__(self, temperature, chemicalpot, DOSratio=None):
        """
        Parameters:
        ___________
        temperature: scalar
            The thermal energy (kB*T: k - Boltzmann's constant, T is
            temperature) of the electron gas, in atomic units (a.u.) or units of
            1/Ha, where Ha = Hartree energy = 27.2114 eV.
        chemicalpotential: scalar
            Chemical potential of the electron gas in a.u. (units of 1/Ha).
        dosratio: function, default None
            Ratio of the nonideal density of states (DOS) and the ideal DOS.
            This is a function of the electronic momentum. In atomic units, this
            is related to the energy by momentum = sqrt(2*energy).
        """
                 self.temp = temperature
                 self.chempot = chemicalpot
                 self.dosratio = DOSratio
