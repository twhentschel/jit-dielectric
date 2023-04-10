# UEG dielectric
Numerical methods to compute the finite-temperature random phase approximation and Mermin dielectric functions of a uniform electron gas (UEG).

# Introduction
The dielectric function is a wave vector and frequency dependent generalization of the macroscopic dielectric constant $\epsilon$ used in theory of linear dielectric media that relates the electric displacement field $\mathbf{D}$ and the electric field $\mathbf{E}$: $\mathbf{D} = \epsilon \mathbf{E}$. It ultimately describes how the electrons in a material respond to some external electromagnetic perturbation or force. The dielectric function is important because it is related to many observable quantities in the linear response regime, like the dynamic structure factor (which is measured in scattering experiments) and the stopping power (which is important in inertial confinement fusion experiments). 

This module computes the complex, quantum mechanical dielectric function for a uniform electron gas using two popular approaches:
1. The __random phase approximation__ (e.g. Johnson, Nilsen, and Cheng, [2012](https://link.aps.org/doi/10.1103/PhysRevE.86.036410)) and 
2. The __Mermin ansatz__ (Mermin, [1970](https://link.aps.org/doi/10.1103/PhysRevB.1.2362)), which modifies the electron gas dielectric function to more accurately represent electrons in the real material by incorporating an electron-ion collision rate.

# Getting Started
> Note: all quantities are in atomic units[^1]

We use `ElectronGas` class objects to hold the physical information about the electrons (e.g. temperature/thermal energy, electron density)
```
from uegdielectric import ElectronGas

# thermal energy of electrons
t = 3.17E-2 # atomic units - approximately 10,000 kelvin
# electron density
d = 2.68e-02 # atomic units - approximately 1.53e16 electrons per centimeters cubed
electrons = ElectronGas(t, d)
```

We can use either the `RPA` or `Mermin` classes to compute the complex dielectric function as a function of wave number `q` and frequency `omega`. For the `Mermin` model, we also need to provide an electron-ion collision rate `nu`, which can be frequency dependent.

```
from uegdielectric import RPA
dielectric = RPA(electrons)
# wave number
q = 1.0 # atomic units
# frequency
omega = 0.5 # atomic units
# dielectric object is callable!
dielectric(q, omega)
>>> (1.452804827343392+0.8463619935408268j)
```
More examples are being added to the `docs/notebooks` directory.

# Requirements
* numpy

To run jupyter notebooks in `docs/notebooks`, will also need
* matplotlib

# Installation
Currently, installation is done by cloning this repository from GitHub. To import this project in Python, it needs to be added to your PYTHONPATH.
Personally, I like to import this directory into my jupyter notebooks by doing:
```
import os
from contextlib import contextmanager

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

with cd("path/to/ueg-dielectric directory"):
    from uegdielectric import Mermin, RPA, ElectronGas
```
where you should replace `"path/to/ueg-dielectric directory"` by the appropriate path on your file system.

In the future, this project should be installable via `pip`.

[^1]: A guide on converting between atomic units and SI units is a work in progress.
