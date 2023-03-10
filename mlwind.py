import numpy as np
from const import neta, bwind, hewind
from numba import njit


# 计算星风质量损失
@njit
def mlwind(kw, lum, r, mt, mc, rl, z):
    lum0 = 7e4     # a constant in formula of mass-loss of Wolf-Rayet-like
    kap = -0.5     # a constant in formula of mass-loss of Wolf-Rayet-like
    dms = 0.0

    # Apply mass loss of Nieuwenhuijzen & de Jager, A&A, 1990, 231, 134, for massive stars over the entire HRD.
    if lum > 4000.0:
        x = min(1.0, (lum-4000.0)/500.0)
        dms = 9.631e-15*x*(r**0.81)*(lum**1.24)*(mt**0.16)
        dms = dms*(z/0.02)**(1.0/2.0)

    teff11 = 1000.0*((1130.0*lum/r**2.0)**(1.0/4.0))
    if 1.25e4 < teff11 <= 2.5e4:
        dms = 10**(-6.688+2.21*np.log10(lum/1.0e5)-1.339*np.log10(mt/30.0)-1.601*np.log10(1.3/2.0)+1.07*np.log10(teff11/2.0e4)+0.85*np.log10(z/0.02))
    elif 2.5e4 < teff11 <= 5.0e4:
        dms = 10**(-6.697+2.194*np.log10(lum/1.0e5)-1.313*np.log10(mt/30.0)-1.226*np.log10(2.6/2.0)+0.933*np.log10(teff11/4.0e4)-10.92*(np.log10(teff11/4.0e4))**2+0.85*np.log10(z/0.02))

    if 2 <= kw <= 9:
        # 'Reimers' mass loss on the GB and beyond,
        dml = neta*4.0e-13*r*lum/mt
        if rl > 0.0:
            dml = dml*(1.0 + bwind*(min(0.50, (r/rl)))**6)
        # Apply mass loss of Vassiliadis & Wood, ApJ, 1993, 413, 641, for high pulsation periods on AGB.
        if kw == 5 or kw == 6:
            log_p0 = min(3.3, -2.07 - 0.9*np.log10(mt) + 1.94*np.log10(r))
            p0 = 10.0**log_p0
            log_dmt = -11.4+0.0125*(p0-100.0*max(mt-2.5, 0))
            dmt = 10.0**log_dmt
            dmt = 1.0*min(dmt, 1.36e-9*lum)
            dml = max(dml, dmt)
        if kw > 6:
            dms = max(dml, 0.5*1.0e-13*hewind*lum**(3.0/2.0)*(z/0.02)**0.86)
        else:
            dms = max(dml, dms)
            mew = ((mt-mc)/mt)*min(5.0, max(1.2, (lum/lum0)**kap))
            # reduced WR-like mass loss for small H-envelope mass
            if mew < 1.0:
                dml = 1.0e-13*lum**(3.0/2.0)*(1.0 - mew)
                dms = max(dml, dms)
            # LBV-like mass loss beyond the Humphreys-Davidson limit.
            x = 1.0e-5*r*lum**(1.0/2.0)
            if lum > 6.0e5 and x > 1.0:
                # dml = 0.1*(x-1.0)**3*(lum/6.0e5-1.0)
                dml = 1.5e-4
                dms = dms + dml

    return dms
