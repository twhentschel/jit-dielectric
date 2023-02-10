# electron-dielectric
A simple description of an electron gas at finite temperatures
using the dielectric formalism.

The main feature of this code is the calculation of the dielectric function
for an electron gas.

The dielectric function describe how a material responds to some external
electromagnetic perturbation or force. This is a function of the material
properties (like temperature and density) and also of the frequency modes of the
perturbing force. The spatial frequency is denoted by k, and is also the
momentum transferred to the material. The temporal frequency is denoted by
omega, and is also the energy transferred to the material. Here, we assume that
the material is isotropic so that we only care about the magnitude of the
spatial frequency. 

The dielectric function also describes the screening of ions by electrons in
the system, resulting in a shielding of the original charge of the ions. This
process creates 'qausi-particle' like objects (a positively charged
particle surrounded by negatively charged electrons) that weak interaction due
to a reduced and modified Coulomb force.

To compute the dielectric function, we rely on two main approximations:

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


In addition, we also provide functions to compute two quantities that are related
to the dielectric function:
1. The dynamic structure factor : structurefactor()
   This quantity is closely related to the intensity of scattered photons
   from a material. By comparing the results from this model to experimental
   scattering spectra in the scattering regime where the signal from the
   free-electrons dominate (also known as the collective regime),
   quantities like the temperature and density can be inferred. Here, the
   assumption is that the free-electrons can be treated approximately as a
   uniform electron gas.
2. The charged particle stopping power : stoppingpower()
    This quantity describes the retarding force a charged particle feels as
    is travels through a material. It depends on the velocity and charge
    of the incoming particle, and also the temperature and density of the
    target material. This model returns the electronic stopping power,
    assuming that the electrons behave primarily like a uniform electron
    gas.
