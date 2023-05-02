"""Inverse of the complete Fermi-Dirac integral"""

import numpy as np


def inv_fdint_onehalf(density, temp):
    """
    Calculates the chemical potential (:math:`\mu`) given the electron density
    and thermal energy by inverting the complete Fermi-Dirac integral of order
    1/2.

    The inversion uses the two approximate inversion formulas from
    :cite:t:`nilsson:1973,nilsson:1978`. The first is

    .. math::

        \eta = \frac{\log u}{u^2 - 1} + \frac{v}{1 + [0.24 + 1.08 v]^{-2}}

    where :math:`\eta = \mu / (k_B T)`, :math:`k_B T` is the thermal energy
    (Boltzmann constant times the temperature), :math:`u = (4 / 3 \pi^{1/2}
    (\epsilon_F / k T)^{3/2}`, :math:`\epsilon_F = (3 \pi^2)^{2/3} n_e / 2` and
    :math:`n_e` is the electron density. This formula has an absolute accuracy of 0.006
    or better over the full range of values for :math:`\eta`. We use the second formula
    when :math:`\eta < 12.7` as it has an absolute error that is less than 5e-5 for
    this region

    .. math::

        \eta = \log u + \frac{u}{[64 + 0.05524 u (64 + u^{1/2})]^{1/4}}.


    Quantities are in atomic units (a.u.).

    Parameters:
    ___________
    density: float
        Electron density of the electron gas, in a.u. or units of 1/a0^3,
        where a0 = Bohr radius = 0.529 Angstrom.
    temperature: float
        The thermal energy (kB*T: kB - Boltzmann constant, T is
        temperature) of the electron gas, in atomic units (a.u.) or units
        of Ha, where Ha = Hartree energy = 27.2114 eV.

    Returns:
    ________
    chempot: float
        the chemical potential (a.u.)

    """
    # Fermi energy
    eF = (3 * np.pi**2 * density) ** (2 / 3) / 2
    # related quantity
    u = 4 / 3 / np.pi**0.5 * (eF / temp) ** (3 / 2)
    # u = (2 * np.pi**3)**0.5 * density / temp**1.5
    if u < 34.3:
        reducedchempot = np.log(u) + u / (64 + 0.05524 * u * (64 + u**0.5)) ** 0.25
    else:
        # simplifying variable in final expression
        v = (3 * np.pi**0.5 * u / 4) ** (2 / 3)
        reducedchempot = np.log(u) / (1 - u**2) + v / (1 + (0.24 + 1.08 * v) ** (-2))

    chempot = reducedchempot * temp

    return chempot
