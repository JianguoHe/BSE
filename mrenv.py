import numpy as np
from utils import conditional_njit


# 估算对流包层的质量、半径, 以及包层的 gyration radius
# N.B. Valid only for Z=0.02!
# The following input is needed from HRDIAG:
# kw = stellar type
# mass = zero-age stellar mass
# mt = actual mass
# mc = core mass (not really needed, can also be done outside subroutine)
# lum = luminosity
# rad = radius
# rc = core radius (not really needed...)
# aj = age
# tm = main-sequence lifetime
# ltms = self.lums[2] = luminosity at TMS, lums[2]
# lbgb = luminosity at BGB, lums[3]
# lhei = luminosity at He ignition, lums[4]
# rzams = radius at ZAMS
# rtms = radius at TMS
# rg = giant branch or Hayashi track radius, approporaite for the type. For kw = 1 or 2 this is radius at BGB,
# and for kw = 4 either GB or AGB radius at present luminosity.


@conditional_njit()
def mrenv(self, rzams, rtms, rg):
    logm = np.log10(self.mass0)
    A = min(0.81, max(0.68, 0.68 + 0.4 * logm))
    C = max(-2.5, min(-1.5, -2.5 + 5.0 * logm))
    D = -0.1
    E = 0.025

    # 自己加的变量初始化
    tebgb = 0

    # Zero-age and BGB values of k^2.
    k2z = min(0.21, max(0.09 - 0.27 * logm, 0.037 + 0.033 * logm))
    if logm > 1.3:
        k2z = k2z - 0.055 * (logm - 1.3) ** 2
    k2bgb = min(0.15, min(0.147 + 0.03 * logm, 0.162 - 0.04 * logm))

    # Envelope k^2 for giant-like stars; this will be modified for non-giant CHeB stars or small envelope mass below.
    # Formula is fairly accurate for both FGB and AGB stars if M <= 10, and gives reasonable values for higher masses.
    # Mass dependence is on actual rather than ZA mass, expected to work for mass-losing stars (but not tested!).
    # The slightly complex appearance is to insure continuity at the BGB, which depends on the ZA mass.
    if 3 <= self.type <= 6:
        logmt = np.log10(self.mass)
        F = 0.208 + 0.125 * logmt - 0.035 * logmt ** 2
        B = 1e4 * self.mass ** (3.0 / 2.0) / (1.0 + 0.1 * self.mass ** (3.0 / 2.0))
        x = ((self.L - self.lums[3]) / B) ** 2
        y = (F - 0.033 * np.log10(self.lums[3])) / k2bgb - 1.0
        k2g = (F - 0.033 * np.log10(self.L) + 0.4 * x) / (1.0 + y * (self.lums[3] / self.L) + x)
    # Rough fit for HeGB stars...
    elif self.type == 9:
        B = 3e4 * self.mass ** (3.0 / 2.0)
        x = (max(0.0, self.L / B - 0.5)) ** 2
        k2g = (k2bgb + 0.4 * x) / (1.0 + 0.4 * x)
    else:
        k2g = k2bgb

    if self.type <= 2:
        menvg = 0.5
        renvg = 0.65
    # FGB stars still close to the BGB do not yet have a fully developed CE.
    elif self.type == 3 and self.L < 3.0 * self.lums[3]:
        x = min(3.0, self.lums[4] / self.lums[3])
        tau = max(0.0, min(1.0, (x - self.L / self.lums[3]) / (x - 1.0)))
        menvg = 1.0 - 0.5 * tau ** 2
        renvg = 1.0 - 0.35 * tau ** 2
    else:
        menvg = 1.0
        renvg = 1.0

    # Stars not on the Hayashi track: MS and HG stars, non-giant CHeB stars,
    # HeMS and HeHG stars, as well as giants with very small envelope mass.
    if self.R < rg:
        # Envelope k^2 fitted for MS and HG stars.
        # Again, pretty accurate for M <= 10 but less so for larger masses.
        # Note that this represents the whole star on the MS, so there is a discontinuity in stellar k^2
        # between MS and HG - okay for stars with a MS hook but low-mass stars should preferably be continous...
        #
        # For other types of star not on the Hayashi track we use the same fit as for HG stars,
        # this is not very accurate but has the correct qualitative behaviour. For CheB stars
        # this is an overestimate because they appear to have a more centrally concentrated envelope than HG stars.
        if self.type <= 6:
            k2e = (k2z - E) * (self.R / rzams) ** C + E * (self.R / rzams) ** D
        # Rough fit for naked He MS stars.
        elif self.type == 7:
            tau = self.age / self.tm
            k2e = 0.080 - 0.030 * tau
        # Rough fit for HeHG stars.
        # 致密星不会进入当前程序, 之前为elif type<=9
        else:
            k2e = 0.08 * rzams / self.R

        # tauenv measures proximity to the Hayashi track in terms of Teff.
        # If tauenv > 0 then an appreciable convective envelope is present, and k^2 needs to be modified.
        if self.type <= 2:
            teff = np.sqrt(np.sqrt(self.L) / self.R)
            tebgb = np.sqrt(np.sqrt(self.lums[3]) / rg)
            tauenv = max(0.0, min(1.0, (tebgb / teff - A) / (1.0 - A)))
        else:
            tauenv = max(0.0, min(1.0, (np.sqrt(self.R / rg) - A) / (1.0 - A)))

        if tauenv > 0.0:
            menv = menvg * tauenv ** 5
            renv = renvg * tauenv ** (5.0 / 4.0)
            # Zero-age values for CE mass and radius.
            if self.type <= 1:
                x = max(0.0, min(1.0, (0.1 - logm) / 0.55))
                menvz = 0.18 * x + 0.82 * x ** 5
                renvz = 0.4 * x ** (1.0 / 4.0) + 0.6 * x ** 10
                y = 2.0 + 8.0 * x
                # Values for CE mass and radius at start of the HG.
                tetms = np.sqrt(np.sqrt(self.lums[2]) / rtms)
                tautms = max(0.0, min(1.0, (tebgb / tetms - A) / (1.0 - A)))
                menvt = menvg * tautms ** 5
                renvt = renvg * tautms ** (5.0 / 4.0)
                # Modified expressions during MS evolution.
                tau = self.age / self.tm
                if tautms > 0.0:
                    menv = menvz + tau ** y * menv * (menvt - menvz) / menvt
                    renv = renvz + tau ** y * renv * (renvt - renvz) / renvt
                else:
                    menv = 0.0
                    renv = 0.0
                k2e = k2e + tau ** y * tauenv ** 3 * (k2g - k2e)
            else:
                k2e = k2e + tauenv ** 3 * (k2g - k2e)
        else:
            menv = 0.0
            renv = 0.0
    # All other stars should be true giants.
    else:
        menv = menvg
        renv = renvg
        k2e = k2g

    menv = menv * (self.mass - self.mass_core)
    renv = renv * (self.R - self.radius_core)
    self.mass_envelop = max(menv, 1e-10)
    self.radius_envelop = max(renv, 1e-10)

    # return k2e
    self.k2 = k2e
    return 0