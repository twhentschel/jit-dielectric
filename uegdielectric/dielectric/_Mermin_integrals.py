"""
@author: tommy

Helper functions to numerically calculates the Mermin dielectric function.

The dielectric function describe how a material responds to some external
electromagnetic perturbation or force. This is a function of the material
properties (like temperature and density) and also of the frequency modes of the
perturbing force. The wave number or spatial frequency is denoted by k, and is also the
momentum transferred to the material. The temporal frequency is denoted by omega, and
is also the energy transferred to the material. Here, we assume that the material is
isotropic so that we only care about the magnitude of the wave number.

The dielectric function also describes the screening of ions by electrons in
the system, resulting in a shielding of the original charge of the ions. This
process creates 'qausi-particle' like objects (a positively charged
particle surrounded by negatively charged electrons) that weak interaction due
to a reduced and modified Coulomb force.

The random phase approximation is a specific approach to computing the screening
effect of electrons in a material. For simplicity, we assume the electrons form
an electron gas (that is, we ignore the prescense of the ions in our material,
replacing them with a positive background charge to ensure charge neutrality of
the system).

The Mermin dielectric function builds upon the random phase approximation (RPA)
dielectric function by including an electron-ion collision frequency term
(denoted as nu in the code) that can also be omega dependent. The collision
frequency aims to improve the assumption of that electrons are electron gas by
approximating how the electronic response is modified (in a relaxation-
time approximation picture) in the presence of the ions in our sample.

This code follows the work by David Perkins, Andre Souza, Didier Saumon,
and Charles Starrett as produced in the 2013 Final Reports from the Los
Alamos National Laboratory Computational Physics Student Summer Workshop,
in the section titled "Modeling X-Ray Thomson Scattering Spectra
of Warm Dense Matter".

All quantities are in atomic units (a.u.)
"""


import numpy as np


def realintegrand(p, k, omega, nu, kBT, mu, dosratio=1):
    """
    The integrand present in the formula for the real part of the general
    RPA dielectric function.

    Parameters:
    ___________
    p: array_like of real values
        The integration variable, which is also the momemtum of the electronic
        state.
    k: array_like of real values
        The spatial frequency of the perturbation acting on the material.
        Units are a.u. or units of 1/a0, where
        a0 = Bohr radius = 0.529 Angstrom.
    omega: array_like of real values
        Temporal frequency of the perturbation acting on the electron
        gas. Units are a.u. or units of Ha, where
        Ha = Hartree energy = 27.2114 eV.
    nu: array_like of real values
        Collision frequency in a.u. If a 1D array, must has same size as omega.
    kBT: real
        Thermal energy (kb - Boltzmann constant, T is temperature) in a.u.
    mu: real
        Chemical potential in a.u.
    dosratio: function, optional
        Ratio of the nonideal density of states (DOS) and the ideal DOS. None
        implies that the ideal density of states is correct, so dosratio(x) = 1
        for any x >= 0.

    Returns:
    ________
        : array_like of real values
    """

    # delta will help with avoiding singularities if the real part of nu is 0.
    deltamod = 1e-5

    # variables to avoid verbose lines later on.
    pp = (k**2 + 2 * (omega - nu.imag) + 2 * p * k) ** 2 + (
        2 * nu.real + deltamod
    ) ** 2
    pm = (k**2 + 2 * (omega - nu.imag) - 2 * p * k) ** 2 + (
        2 * nu.real + deltamod
    ) ** 2
    mp = (k**2 - 2 * (omega - nu.imag) + 2 * p * k) ** 2 + (
        2 * nu.real + deltamod
    ) ** 2
    mm = (k**2 - 2 * (omega - nu.imag) - 2 * p * k) ** 2 + (
        2 * nu.real + deltamod
    ) ** 2

    logpart = np.log(np.sqrt(pp / pm)) + np.log(np.sqrt(mp / mm))

    FD = 1 / (1 + np.exp((p**2 / 2 - mu) / kBT))

    return logpart * FD * p * dosratio


def DEtransform(u, k, omega, nu, kBT, mu, plim, dosratio):
    r"""
    Transform the real integral using a Double Exponential (DE) change of
    variables.

    The transformation is of the form

    x = tanh(a sinh(u))
    dx = a cosh(u) / cos^2(a sinh(u)) du

    where a = \pi/2.

    This transformation will take an integral from [-1, 1] to
    (-\infty, +\infty).

    We first use the linear tranformation

    2p = (b-a)x + (b+a),

    where plim=(a,b) are the limits of the original integral, to transform the
    integration range to (-1, 1).

    Parameters:
    ___________
    u: array_like of real values
        The integration variable, which is transformation of the the momemtum of
        the electronic state.
    k: array_like of real values
        The spatial frequency of the perturbation acting on the material.
        Units are a.u. or units of 1/a0, where
        a0 = Bohr radius = 0.529 Angstrom.
    omega: array_like of real values
        Temporal frequency of the perturbation acting on the electron
        gas. Units are a.u. or units of Ha, where
        Ha = Hartree energy = 27.2114 eV.
    nu: array_like of real values
        Collision frequency in a.u. If a 1D array, must has same size as omega.
    kBT: real
        Thermal energy (kb - Boltzmann constant, T is temperature) in a.u.
    mu: real
        Chemical potential in a.u.
    plim: list-like of length 2
        Original limits of integration. The limits after the transformation
        will be from (-\infty, +\infty)
    dosratio: function, optional
        Ratio of the nonideal density of states (DOS) and the ideal DOS. None
        implies that the ideal density of states is correct, so dosratio(x) = 1
        for any x >= 0.

    Returns:
    ________
        : array_like of real values
    """

    a, b = plim

    ptrans = ((b - a) * np.tanh(np.pi / 2 * np.sinh(u)) + (b + a)) / 2

    transfactor = (
        (b - a) / 2 * np.pi / 2 * np.cosh(u) / np.cosh(np.pi / 2 * np.sinh(u)) ** 2
    )

    return transfactor * realintegrand(ptrans, k, omega, nu, kBT, mu, dosratio(ptrans))


def imagintegrand(p, k, omega, nu, kBT, mu, dosratio=1):
    """
    The integrand present in the formula for the imaginary part of the general
    RPA dielectric function.

    k: array_like of real values
        The spatial frequency of the perturbation acting on the material.
        Units are a.u. or units of 1/a0, where
        a0 = Bohr radius = 0.529 Angstrom.
    omega: array_like of real values
        Temporal frequency of the perturbation acting on the electron
        gas. Units are a.u. or units of Ha, where
        Ha = Hartree energy = 27.2114 eV.
    nu: array_like of real values
        Collision frequency in a.u. If a 1D array, must has same size as omega.
    kBT: real
        Thermal energy (kb - Boltzmann constant, T is temperature) in a.u.
    mu: real
        Chemical potential in a.u.
    dosratio: function, optional
        Ratio of the nonideal density of states (DOS) and the ideal DOS. None
        implies that the ideal density of states is correct, so dosratio(x) = 1
        for any x >= 0.

    Returns:
    ________
        : array_like of real values
    """

    # variables to avoid verbose lines later on.
    pp = k**2 + 2 * (omega - nu.imag) + 2 * p * k
    pm = k**2 + 2 * (omega - nu.imag) - 2 * p * k
    mp = k**2 - 2 * (omega - nu.imag) + 2 * p * k
    mm = k**2 - 2 * (omega - nu.imag) - 2 * p * k

    arctanpart = (
        np.arctan2(2.0 * nu.real, pp)
        - np.arctan2(2.0 * nu.real, pm)
        + np.arctan2(-2.0 * nu.real, mp)
        - np.arctan2(-2.0 * nu.real, mm)
    )

    FD = 1 / (1 + np.exp((p**2 / 2 - mu) / kBT))

    return arctanpart * FD * p * dosratio


def generalRPAdielectric(k, omega, nu, kBT, mu, dosratio=None):
    """
    Numerically calculates the dielectric function  in Random Phase
    Approximation (RPA), epsilon_{RPA}(k, omega + i*nu). This function is
    labelled general becuase the frequency argument is made complex to account
    for collision frequency parameter, nu. This alone is not a correct
    expression for the dielectric function, and is used in calculating the
    Mermin dielectric function.

    Parameters:
    ___________
    k: array_like of real values
        The spatial frequency of the perturbation acting on the material.
        Units are a.u. or units of 1/a0, where
        a0 = Bohr radius = 0.529 Angstrom.
    omega: array_like of real values
        Temporal frequency of the perturbation acting on the electron
        gas. Units are a.u. or units of Ha, where
        Ha = Hartree energy = 27.2114 eV.
    nu: array_like of real values
        Collision frequency in a.u. If a 1D array, must has same size as omega.
    kBT: real
        Thermal energy (kb - Boltzmann constant, T is temperature) in a.u.
    mu: real
        Chemical potential in a.u.
    dosratio: function, optional
        Ratio of the nonideal density of states (DOS) and the ideal DOS. None
        implies that the ideal density of states is correct, so dosratio(x) = 1
        for any x >= 0.

    Returns:
    ________
    ret: ndarray of complex values
        If k and omega are both 1D arrays, shape will be (size(k), size(omega)).
        Otherwise, if only one of these arguments is a 1D array of size n and
        the other is a scalar, the shape is (size(n),). If both arguments are
        scalars, the result is a complex scalar as well.
    """

    # To handle both scalar and array inputs
    k = np.asarray(k)
    omega = np.asarray(omega)
    nu = np.asarray(nu)
    scalar_input = False
    if k.ndim == 0:
        k = np.expand_dims(k, axis=0)  # Makes k 1D
        scalar_input = True
    if omega.ndim == 0:
        omega = np.expand_dims(omega, axis=0)
        scalar_input = True
    if nu.ndim == 0:
        nu = np.expand_dims(nu, axis=0)
        scalar_input = True
    # Lengths of array inputs
    M = k.size
    N = omega.size

    # Meshgrid for broadcasting k, omega
    k, omega = np.meshgrid(k, omega, indexing="ij", sparse=True)

    if dosratio is None:
        # Make dosratio the constant function returning 1
        dosratio = lambda x: 1

    # A small nu causes some problems when integrating the real and imaginary
    # parts of the dielectric.
    # When nu is small, the imaginary integrand is like a modulated step
    # function between p1 and p2, while the real part develops sharp peaks at
    # p1 and p2 (the peaks should go to infinity, but I damp them with the
    # small delta term in the integrand).
    # (p3 essentially defines the point at which the Fermi-Dirac exponential,
    # 1 / (1 + np.exp((p^2/2 - mu)/kBT), starts to drop off considerably.)
    p1 = abs(k**2 - 2 * omega) / (2 * k)
    p2 = (k**2 + 2 * omega) / (2 * k)
    p3 = np.sqrt(abs(2 * mu))

    # Integral for real part of the dielectric function #

    # Transformed integrand for real part
    realint = lambda x, lims: DEtransform(x, k, omega, nu, kBT, mu, lims, dosratio)

    # All transformed integrations fall roughly within the same region in the
    # transformed space
    t = np.linspace(-2.5 * np.ones((M, N)), 2.5 * np.ones((M, N)), 200, axis=0)
    tempwidth = np.sqrt(2 * np.abs(mu + 10 * kBT))
    realsolve = (
        np.trapz(realint(t, (np.zeros(N), p1)), t, axis=0)
        + np.trapz(realint(t, (p1, p2)), t, axis=0)
        + np.trapz(realint(t, (p2, 2 * p2 + tempwidth)), t, axis=0)
    )

    # Integral for the imag part of the dielectric function #

    imagint = lambda x: imagintegrand(x, k, omega, nu, kBT, mu, dosratio(x))

    # Explicitly identify difficult points in integration range, plus the
    # "widths" around each point (1e-4 gently smoothes peaks when nu.real == 0)
    nuwidth = nu.real + 1e-4
    # Put these into an array
    pdiff = np.zeros((8, M, N))
    pdiff[1] = np.maximum(p1 - nuwidth, np.zeros((M, N)))
    pdiff[2] = p1 + nuwidth
    pdiff[3] = np.maximum(p2 - nuwidth, np.zeros((M, N)))
    pdiff[4] = p2 + nuwidth
    pdiff[5] = np.maximum(p3 - tempwidth, np.zeros((M, N)))
    pdiff[6] = p3 + tempwidth
    pdiff[7] = p2 + nuwidth + tempwidth
    # Sort the difficult points, so they are in order
    pdiff = np.sort(pdiff, axis=0)
    # Linearly interpolate between the difficult point +/- their widths to
    # create a set of integration regions bounded by these points (+/- widths)
    intregions = np.linspace(pdiff[0:7], pdiff[1:8], 100, axis=0)
    # Integrate within each of the regions
    imagintegrateregions = np.trapz(imagint(intregions), intregions, axis=0)
    # Add up the integrations between each region, resulting in an array of
    # shape (M,N)
    imagsolve = np.sum(imagintegrateregions, axis=0)

    ret = 1j * 2 / np.pi / k**3 * imagsolve
    ret += 1 + 2 / np.pi / k**3 * realsolve

    if scalar_input:
        return np.squeeze(ret)
    return ret


def generalMermin(epsilon, k, omega, nu, *args):
    """
    Numerically calculates the Mermin dielectric function. This adds some ionic
    structure to the dielectric function passed through epsilon. Typically this
    will be the RPA dielectric function, but we also want to allow for a
    general dielectric functions.

    Parameters:
    ___________
    epsilon: function
        dielectric function that we want to add ionic information to. The
        argument structure must be epsilon(k, omega, nu, args) and args must
        be ordered properly.
    k: array_like of real values
        The spatial frequency of the perturbation acting on the material.
        Units are a.u. or units of 1/a0, where
        a0 = Bohr radius = 0.529 Angstrom.
    omega: array_like of real values
        Temporal frequency of the perturbation acting on the electron
        gas. Units are a.u. or units of Ha, where
        Ha = Hartree energy = 27.2114 eV.
    nu: array_like of real values
        Collision frequency in a.u. If a 1D array, must has same size as omega.
    args: tuple
        Additional arguments (temperature, chemical potential, ...). Must be
        same order as in the epsilon() function.

    Returns:
    ________
    ret: ndarray of complex values
        If k and omega are both 1D arrays, shape will be (size(k), size(omega)).
        Otherwise, if only one of these arguments is a 1D array of size n and
        the other is a scalar, the shape is (size(n),). If both arguments are
        scalars, the result is a complex scalar as well.
    """

    omega = np.asarray(omega)
    N = omega.shape

    epsnonzerofreq = epsilon(k, omega, nu, *args)
    epszerofreq = epsilon(k, np.zeros(N), np.zeros(N), *args)

    # If nu is zero, we expect to get back epsnonzerofreq. But if omega also
    # equals zero, this code fails. Add a little delta to omega to avoid this.
    delta = 1e-10
    numerator = ((omega + delta) + 1j * nu) * (epsnonzerofreq - 1)
    denominator = (omega + delta) + 1j * nu * (epsnonzerofreq - 1) / (epszerofreq - 1)

    return 1 + numerator / denominator


def MerminDielectric(k, omega, nu, kBT, mu, dosratio=None):
    """
    Numerically calculates the Mermin dielectric, which builds upon the RPA
    dielectric function by taking into account electron collisions with ions.

    Parameters:
    ___________
    k: array_like of real values
        The spatial frequency of the perturbation acting on the material.
        Units are a.u. or units of 1/a0, where
        a0 = Bohr radius = 0.529 Angstrom.
    omega: array_like of real values
        Temporal frequency of the perturbation acting on the electron
        gas. Units are a.u. or units of Ha, where
        Ha = Hartree energy = 27.2114 eV.
    nu: array_like of real values
        Collision frequency in a.u. If a 1D array, must has same size as omega.
    kBT: real
        Thermal energy (kb - Boltzmann constant, T is temperature) in a.u.
    mu: real
        Chemical potential in a.u.
    dosratio: function, optional
        Ratio of the nonideal density of states (DOS) and the ideal DOS. None
        implies that the ideal density of states is correct, so dosratio(x) = 1
        for any x >= 0.

    Returns:
    ________
    ret: ndarray of complex values
        If k and omega are both 1D arrays, shape will be (size(k), size(omega)).
        Otherwise, if only one of these arguments is a 1D array of size n and
        the other is a scalar, the shape is (size(n),). If both arguments are
        scalars, the result is a complex scalar as well.
    """

    return generalMermin(generalRPAdielectric, k, omega, nu, kBT, mu, dosratio)
