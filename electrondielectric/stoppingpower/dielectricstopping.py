#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:29:10 2020

@author: tommy

Calculates the stopping power based on the dielectric formalism, 
see: M. D. Barriga-Carrasco, PRE, 79, 027401 (2009)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad_vec
import scipy.optimize as opt

def plasmafreq(density):
    return np.sqrt(4*np.pi*density)

def sumrule(density):
    return np.pi/2 * plasmafreq(density)**2

def ELF(dielfunc, k, w):
    eps = dielfunc(k, w)
    if (eps.imag**2 + eps.real**2) == 0:
        raise ValueError
    return eps.imag / (eps.imag**2 + eps.real**2)

def modBG_wp(den, k, temp):
    """
    Modified Bohm-Gross dispersion relation, given in Glezner & Redmer, Rev.
    Mod. Phys., 2009, Eqn (16).
    
    den - electron density (au)
    k - wavenumber (au)
    temp - thermal energy (au)
    """  
    wp    = np.sqrt(4*np.pi*den)
    BG_wp = np.sqrt(wp**2 + 3*temp*k**2)
    thermal_deBroglie = np.sqrt(2*np.pi/temp)
    
    return np.sqrt(BG_wp**2 + 3*temp*k**2 \
                    * 0.088 * den * thermal_deBroglie**3 \
                    + (k**2/2)**2)

def ELFmax(dielfunc, k, prevroot, prevfun, directopt=True):
    """
    Finds the maximum position of the electron loss function (ELF).
    
    Note: this will stop working for small values of k, so be wary!
    One way to make sure that this function is still working is to check that
    the ELF maximum for a larger k is less than the ELF maximum for a smaller
    k (for the RPA dielectric function).
    
    """
    f = lambda x: -genELF(dielfunc, k, x)
    
    root = 0
    feval = 0
    bounds = (0, prevroot)
    if directopt:
        # Look for minimum of ELF by optimizing the ELF directly
        boundsroot = opt.minimize_scalar(f, bounds=bounds, method='bounded')
        if -boundsroot.fun <= prevfun:
            directopt = False
        else:
            root = boundsroot.x
            feval = -boundsroot.fun
    if not directopt:
        # Look for the minimum of the ELF by finding the second zero of the
        # real part of the dielectric function.
        reeps = lambda x : dielfunc(k, x).real
        # Find the zero
        try:
            root = opt.newton(reeps, prevroot)
        except RuntimeError:
            root = prevroot
            print("Newton failed @ k = {}".format(k))
        try:
            # We need to watch out for actual delta-functions
            feval = -f(root)
        except ValueError:
            feval = prevfun

    return root, feval, directopt

def ELFdispersion(dielfunc, temp, chempot, density, kgrid, RPA=True,
                  status=False):
    """
    Calculates the dispersion relation of the electron loss function (ELF) of a
    material. The ELF is related to the dielectric function, dielfunc, by
    ELF = -dielfunc.imag / (dielfunc.imag**2 + dielfunc.real**2)
    
    Parameters:
    ___________
    dielfunc: (complex) function of two arguments
        The dielectric function of the target. This function depends on the
        momentum (spatial frequency) and energy (temporal frequency) of a
        perturbatuion (these are typically denoted by :math:`k, \omega`). This
        function returns a complex quantity.
    temp: float
        Thermal energy of target in a.u. or units of Hartee = 27.2114 eV.
    chempot: float
        Chemical potential or Fermi level of the target in units of a.u.
    density: float
        Electronic density of the target in units of a.u. or units of 1/a_0^3,
        where a_0 = 0.529 Angstrom.
    kgrid: array-like
        Grid used in trapezoidal integration over the spatial frequency
        argument.
    RPA: boolean
        The RPA or random phase approximation dielectric function is a theory
        for describing the response of a collisionless electron gas. In this
        approach, for small k (first argument in the dielectric
        function) the ELF can develop a very sharp peak that makes finding the
        dispersion difficult. A different approach must be used for these cases,
        but there is no guarantee that this will work for arbitrarily small
        k. Tested for k as low as 1e-3 a.u. for specific temp, chempot, and
        density. Default is True.
    status: boolean
        Print a message if first attempt at finding the peak of the ELF function
        failed. Default is False.
        
    Returns:
    roots: array-like
        Position of the ELF peak (in energy space) as a function of kgrid. Same
        length as kgrid.
    froots: array-like
        The value of the the ELF at its peak. Same length as kgrid.
    ________
    
    """

    # We work in descending order, from large k to small k
    flip = False
    if kgrid[0] < kgrid[-1]:
        kgrid = np.flip(kgrid)
        flip=True
    
    # Function to optimize (want to maximize the ELF)
    f = lambda k, w : -ELF(k, w)
    
    # location of peaks
    roots = np.zeros(len(kgrid))
    # value of f at peaks
    froots = np.zeros(len(kgrid))
    
    # Try to optimize the ELF directly first
    directopt = True
    
    # approximate width of ELF peak, better for larger k
    tempwidth = np.sqrt(2*(10*temp + abs(chempot)))
    # For collisional dielectric function, need a larger width (empirical guess)
    collisionalwidth = 1e3 if RPA else 0
    
    # These two parameters important for checking progression of RPA dispersion
    prevELFmax = 0
    prevroot = modBG_wp(density, kgrid[0], temp)
    for i in range(0, len(kgrid)):
        k = klist[i]
        initguess = modBG_wp(density, kgrid[0], temp)
        if RPA:
            initguess = prevroot
        # empirically widening the bounds
        bounds = (0, initguess + tempwidth * kgrid[0] + collisionalwidth)
        
        if directopt or not RPA:
            # directly optimize ELF using bounded minimization
            root = opt.minimize_scalar(f, bounds=bounds, args=(k),
                                       method='bounded')
            if root.fun >= prevELFmax:
                # This method failed, try the second method from now on if using
                # RPA
                directopt = False
            
            roots[i] = root.x
            froots[i] = -root.fun
            
            # Try to find the other peak at higher w, if it exists
            try:
                bounds = (roots[i], initguess + tempwidth * kgrid[0] + \
                    collisionalwidth)
                root = opt.minimize_scalar(f, bounds=bounds, args=(k),
                                           method='bounded')
                roots[i] = root.x
                froots[i] = -root.fun
            except opt.OptimizeWarning:
                continue
            
            if status:
                print("k = {}: first attempt success, root = {}, value = {}".\
                      format(k, root.x, -root.fun))
        
        if not directopt and RPA:
            # Look for the minimum by finding the second zero of the real part 
            # of the dielectric function.
            reeps = lambda w : dielfunc(k, w).real
            
            # Find the zero
            roots[i], res = opt.newton(reeps, prevroot, full_output=True)
            froots[i] = -f(roots[i], k)
            if status:
                print("k = {}: second attempt, root = {}, value = {}".\
                      format(k, roots[i], -f(roots[i], k)))
                print("Real part at root = ", reeps(roots[i]))
            
            
        prevroot = roots[i]              
        prevELFmax = f(prevroot, k)
    if flip:
        roots = np.flip(roots)
        froots = np.flip(froots)
        
    return roots, froots

def omegaintegral(dielfunc, v, k, collfreq, temp, chempot, ELFmaxpos,
                  ELFmaxval, density, vlow=0):
    """
    Calculates the inner, omega integral for the dielectric stopping power

    Parameters
    ----------
    dielfunc : function, of the form f(x, y)
        Dielectric function.
    v : scalar
        Initial charged particle velocity.
    k : scalar
        Wavenumber.
    temp : scalar
        temperature.
    chempot : scalar
        Chemical potential.
    ELFmaxpos: scalar
        Position in omega-space for a given k of the electron loss function
        (ELF).
    ELFmaxval: scalar
        Value of ELF at ELFmaxpos.
    sumrule: scalar
        Sum rule value.
    vlow : scalar, optional
        This parameter is useful for the sequential stopping function. It
        represents the lower integration limit for the omega integral. 
        The default is 0.

    Returns
    -------
    omegaint : float
        Value for the omega integral for a given v, k

    """
    sr = sumrule(density)
    # plasma frequency
    wp = plasmafreq(density)
    
    # A rough lower bound approximate width of peak, most meaningful for small
    # values of k when the ELF is very sharp.
    srwidth = sr / ELFmaxval
       
    # A width associated with the wavenumber, k, and the temperature, temp.
    # This is a conservative upper bound made to capture all of the integrand
    # (with some heuristically motivated approximations).
    tempwidth = np.sqrt(2*(10*temp + abs(chempot)))*k + collfreq(0).real*1e3
    
    width = np.minimum(srwidth, tempwidth)
      
    # Define our integration regions
    regions = np.zeros(4)
    
    regions[1] =  np.maximum(0., ELFmaxpos - width)
    regions[2] = ELFmaxpos + width
    # This region is important for small k, when srwidth becomes really small.
    regions[3] = np.maximum(ELFmaxpos + tempwidth, k*v)
    
    
    
    # Where to place k*v with respect to the regions above
    kvi = np.searchsorted(regions, k*v)
    regions = np.insert(regions, kvi, k*v)

    # integrand
    f = lambda x, y : x * genELF(dielfunc, k, x)
    
    # integral from [0, kv] and from [0, \infty)
    # If kvi == 0, then k*v = 0, don't need to do this integral
    omegaint = 0.
    omegaint_allspace = 0.
    
    for i in range(1, len(regions)):
        I = solve_ivp(f, (regions[i-1], regions[i]), [0],
                      rtol=1e-4, vectorized=True)
        # print("nevals{} = {}".format(i, I.nfev))
        omegaint_allspace += I.y[0][-1]
        # w = np.linspace(regions[i-1], regions[i], 500)
        # omegaint_allspace += np.trapz(f(w), w)
        
        if kvi == i:
            omegaint = omegaint_allspace
        

    # Check the sum rule
    error = (sr - omegaint_allspace)/sr
        
    
    return omegaint, width, error, regions
        

def omegaintegral_check(dielfunc, v, collfreq, temp, chempot, density,
                        kgrid=None):
    """
    Function that makes it easy to check the omega integral by eye (not cool!)

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    dielfunc : TYPE
        DESCRIPTION.
    temp : TYPE
        DESCRIPTION.
    chempot : TYPE
        DESCRIPTION.
    density : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sr = sumrule(density)
    wp = plasmafreq(density)
    
    # String if we don't satisfy the sum rule
    errmssg = ""
    
    ## Upper limit for k integral
    # tempwidth is a temperature-based correction to the typically upper bound
    # seen in the literature.
    tempwidth = np.sqrt(2*(10*temp + abs(chempot)))                                
    kupperbound= 2*(v +  tempwidth)
    # Define grid in k-space
    if kgrid is None:
        kgrid = np.geomspace(5e-2, kupperbound, 100)
    
    omegaint = np.zeros(len(kgrid))
    error    = np.zeros(len(kgrid))
    directopt = True
    
    # inital guess
    ELFmaxpos = modBG_wp(density, kgrid[-1], temp)
    ELFmaxval = -1
    
    # Find maximum positions of ELF, working backwards starting from larger
    # values of k.
    deltafunc = False
    for i, k  in reversed(list(enumerate(kgrid))):
        prevpos =  ELFmaxpos
        ELFmaxpos, ELFmaxval, directopt = ELFmax(dielfunc, k, prevpos,
                                                 ELFmaxval, directopt)
        
        # For small values of k (depends on the plasma frequency), the ELF has
        # a very sharp, delta function like peak that I could never hope to 
        # integrate numerically. Instead, we know that this peak contains a 
        # majority of the area under the curve (the sum rule value), with only
        # a little area existing before the position of this peak. Beyond the 
        # location of the peak, the ELF drops off extremely rapidly with omega.
        srwidth = sr / ELFmaxval
        if srwidth <= 1e-4 or deltafunc:
            deltafunc = True
            if k*v > ELFmaxpos:
                omegaint[i] = sr
            error[i] = 0.
        else:
            # if prevval > ELFmaxval:
            #     break
            omegaintres = omegaintegral(dielfunc, v, k, collfreq, temp, 
                                        chempot, ELFmaxpos, ELFmaxval, density)
            omegaint[i], delta, error[i], reg = omegaintres
            
        SRsatisfied = abs(error[i]) < 5e-2
        
        if (not SRsatisfied):
            omegaint[i] = -omegaint[i]
            errmssg = "########## SUMRULE NOT SATISFIED ############\n"\
                    + "k = {:.15f}\n".format(k)\
                    + "ELF max pos = {}\n".format( ELFmaxpos)\
                    +"regions = {}\n".format( reg)\
                    + "ELF max val = {}\n".format( ELFmaxval)\
                    + "error = {:.3f}\n".format(error[i])
            print(errmssg)                
    
    return omegaint, kgrid, error
    # kintegrand = 1/kgrid * omegaint
    # kintegral = np.trapz(kintegrand, kgrid)

def dielectric_stopping_power(v, Zp, dielfunc, targettemp, targetchempot,
                              targetdensity, RPA=True, kgrid=None):
    """
    Computes the velocity-dependent stopping power in the dielectric formalism.
    For an example of the formula used, see:
    
    Parameters:
    ___________
    v: float or array-like
        Velocity of incoming projectile, in atomic units (a.u) or units of c/137
        where c is the speed of light and 1/137 is the fine-structure constant.
    Zp: float
        Charge of incoming projectile.
    dielfunc: (complex) function of two arguments
        The dielectric function of the target. This function depends on the
        momentum (spatial frequency) and energy (temporal frequency) of a
        perturbatuion (these are typically denoted by :math:`k, \omega`). This
        function returns a complex quantity.
    targetemp: float
        Thermal energy of target in a.u. or units of Hartee = 27.2114 eV.
    targetchempot: float
        Chemical potential or Fermi level of the target in units of a.u.
    targetdensity: float
        Electronic density of the target in units of a.u. or units of 1/a_0^3,
        where a_0 = 0.529 Angstrom.
    RPA: boolean
        The integrand in this numerical integral calculation can be
    kgrid: array-like
        Grid used in trapezoidal integration over the spatial frequency
        argument. Default is None for an automatically chosen grid.
        
    Returns:
    ________
    
    """
    
    # Create kgrid for spatial frequency (denoted by k) integration if not
    # provided
    # first, define the upper limit for k integral:
    # tempwidth is a temperature-based correction to the typically upper bound
    # seen in the literature.
    tempwidth = np.sqrt(2*(10*temp + abs(chempot)))                                
    kupperbound= 2*(v +  tempwidth)
    # Define grid in k-space
    if kgrid is None:
        kgrid = np.geomspace(5e-2, kupperbound, 100)
        
    # Get dispersion of electron loss function (ELF)
    # Essentially finding the (plasmon) peak of the ELF. This makes integrating
    # over the ELF numerically less challenging.
    ELFmaxpos, ELFmax = ELFdispersion(dielfunc, targettemp, targetchempot,
                                      targetdensity, kgrid, RPA):
        
    # Compute the inner, temporal frequency (omega) integral
    
    # Compute the outer, spatial frequency (k) integral
    
    # Handle cases where the ELF's are "delta-function"-like, which occurs for
    # the RPA for small k


