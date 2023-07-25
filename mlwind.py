import numpy as np
from numba import njit
from const import acc1, alpha_wind, kick, output, find, SNtype, G, Msun, Rsun, beta_wind, tiny
from const import eta, bwind, hewind, Zsun, mass_loss_wind, wind_model
from numba import float64, int64, types
from numba.experimental import jitclass


@njit
def mdot_wind(star1, star2, sep, ecc, Z):
    # q =
    # total_mass = star1.mass + star2.mass

    # 计算 star1 的星风质量损失率，用 mdot_wind_loss 表示
    if wind_model == 'Hurley':
        star1.mdot_wind_loss = -star1.cal_mdot_wind_Hurley
    else:
        star1.mdot_wind_loss = -star1.cal_mdot_wind_Belczynski
    # 计算 star2 从 star1 星风中质量吸积率, 用 dma_wind_accrete 表示(Boffin & Jorissen, A&A 1988, 205, 155).
    vorb2 = acc1 * (star1.mass + star2.mass) / sep
    vwind2 = 2.0 * beta_wind * acc1 * star1.mass / star1.R
    term1 = 1.0 / np.sqrt(1.0 - ecc ** 2)
    term2 = (acc1 * star2.mass / vwind2) ** 2
    term3 = 1 / (1.0 + vorb2 / vwind2) ** 1.5
    term4 = alpha_wind * abs(star1.mdot_wind_loss) / (2.0 * sep ** 2)
    star2.mdot_wind_accrete = term1 * term2 * term3 * term4
    star2.mdot_wind_accrete = min(star2.mdot_wind_accrete, 0.8 * abs(star1.mdot_wind_loss))

    # 由于星风损失的轨道角动量
    # ecc2 = ecc * ecc
    # ecc3 = 1.0 - ecc2
    # ecc4 = np.sqrt(1.0 - ecc2)
    # ecc5 = 1.0 / np.sqrt(1.0 - ecc2)
    # omecc2 = 1.0 - ecc2
    # sqome2 = np.sqrt(omecc2)


# # 计算星风质量损失(Hurley Model)
# @njit
# def mlwind_Hurley(kw, lum, r, mt, mc, rl, z):
#     # 初始值
#     mdot_NJ = 0
#     mdot_KR = 0
#     mdot_WR = 0
#     mdot_VW = 0
#     mdot_LBV = 0
#
#     # 计算 mdot_NJ (mass loss rate for massive stars (L > 4000Lsun) over the entire HRD)
#     # Nieuwenhuijzen & de Jager 1990, A&A, 231, 134
#     if lum > 4000.0:
#         term_NJ = min(1.0, (lum - 4000.0) / 500.0)
#         mdot_NJ = 9.631e-15 * term_NJ * r ** 0.81 * lum ** 1.24 * mt ** 0.16 * (z / 0.02) ** 0.5
#
#     Teff = 1000.0 * ((1130.0 * lum / r ** 2.0) ** (1.0 / 4.0))
#     if 1.25e4 < Teff <= 2.5e4:
#         term1 = -6.688 + 2.21 * np.log10(lum / 1.0e5) - 1.339 * np.log10(mt / 30.0) - 1.601 * np.log10(1.3 / 2.0)
#         term2 = 1.07 * np.log10(Teff / 2.0e4) + 0.85 * np.log10(z / 0.02)
#         mdot_NJ = 10 ** (term1 + term2)
#     elif 2.5e4 < Teff <= 5.0e4:
#         term1 = -6.697 + 2.194 * np.log10(lum / 1.0e5) - 1.313 * np.log10(mt / 30.0) - 1.226 * np.log10(2.6 / 2.0)
#         term2 = 0.933 * np.log10(Teff / 4.0e4) - 10.92 * np.log10(Teff / 4.0e4) ** 2 + 0.85 * np.log10(z / 0.02)
#         mdot_NJ = 10 ** (term1 + term2)
#
#     # 计算 mdot_KR (mass loss rate on the GB and beyond)
#     # Hurley et al. 2000, eq 106 (based on a prescription taken from Kudritzki & Reimers, 1978, A&A, 70, 227)
#     if 2 <= kw <= 9:
#         mdot_KR = eta * 4.0e-13 * r * lum / mt
#         # 考虑 mdot_KR 受潮汐增强
#         if rl > 0.0:
#             mdot_KR = mdot_KR * (1.0 + bwind * (min(0.50, (r / rl))) ** 6)
#
#     # 计算 mdot_VW (mass loss rate on the AGB based on the Mira pulsation period)
#     # Hurley et al. 2000, just after eq 106 (from Vassiliadis & Wood, 1993, ApJ, 413, 641)
#     if kw == 5 or kw == 6:
#         p0 = min(1995, 8.51e-3 * r ** 1.94 / mt ** 0.9)
#         p1 = 100.0 * max(mt - 2.5, 0)
#         mdot_VW = min(10.0 ** (-11.4 + 0.0125 * (p0 - p1)), 1.36e-9 * lum)
#
#     # 计算 mdot_WR (mass loss of Wolf–Rayet like, for star with small hydrogen-envelope mass)
#     if 7 <= kw <= 9:
#         mdot_WR = 0.5 * 1.0e-13 * hewind * lum ** 1.5 * (z / 0.02) ** 0.86
#     elif 2 <= kw <= 6:
#         # reduced WR-like mass loss for small H-envelope mass
#         lum0 = 7e4
#         kap = -0.5
#         mew = (mt - mc) / mt * min(5.0, max(1.2, (lum / lum0) ** kap))
#         if mew < 1.0:
#             mdot_WR = 1.0e-13 * lum ** 1.5 * (1.0 - mew)
#
#     # 计算 mdot_LBV (mass loss of Humphreys & Davidson 1994, PASP, 106, 1025, for LBV-like star beyond HD limit)
#     HD = 1.0e-5 * r * lum ** 0.5
#     if lum > 6.0e5 and HD > 1.0:
#         # mdot_LBV = 0.1 * (HD - 1.0) ** 3 * (lum / 6.0e5 - 1.0)
#         mdot_LBV = 1.5e-4
#
#     # 整体星风质量损失
#     if 0 <= kw <= 6:
#         return max(mdot_NJ, mdot_KR, mdot_VW, mdot_WR) + mdot_LBV
#     elif 7<= kw <= 9:
#         return max(mdot_KR, mdot_WR)
#     else:
#         return 0
#
#
# # 计算星风质量损失(Belczynski Model)
# def cal_mdot_wind(kw, lum, r, mt, mc, rl, z):
#
#     if 7 <= kw <= 9:
#         if mass_loss_wind == 'Vink':
#             mdot_wind =
#         elif mass_loss_wind == 'Hurley':
#             mdot_wind =
#         else:
#             mdot_wind = 0
#         return mdot_wind



