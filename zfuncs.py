import numpy as np
from utils import conditional_njit
from const import yearsc, Msun, G, c, pc, num_evolve
import legwork as lw
import astropy.units as u
import astropy.coordinates.sky_coordinate as skycoord

# 集合了所有的独立函数（公式）




# A function to evaluate radius derivitive on the GB (as f(L)).
@conditional_njit()
def rgbdf(m, lum, x):
    a1 = min(x.gbp[20] / m ** x.gbp[21], x.gbp[22] / m ** x.gbp[23])
    rgbd = a1 * (x.gbp[18] * lum ** (x.gbp[18] - 1) + x.gbp[17] * x.gbp[19] * lum ** (x.gbp[19] - 1))
    return rgbd



# A function to evaluate radius derivitive on the AGB (as f(L)).
@conditional_njit()
def ragbdf(m, lum, mhelf, x):
    m1 = mhelf - 0.2
    if m >= mhelf:
        xx = x.gbp[24]
    elif m >= m1:
        xx = 1 + 5 * (x.gbp[24] - 1) * (m - m1)
    else:
        xx = 1
    a4 = xx * x.gbp[19]
    if m <= m1:
        a1 = x.gbp[29] + x.gbp[30] * m
    elif m >= mhelf:
        a1 = min(x.gbp[25] / m ** x.gbp[26], x.gbp[27] / m ** x.gbp[28])
    else:
        a1 = x.gbp[31] + 5 * (x.gbp[32] - x.gbp[31]) * (m - m1)
    ragbd = a1 * (x.gbp[18] * lum ** (x.gbp[18] - 1) + x.gbp[17] * a4 * lum ** (a4 - 1))
    return ragbd


# A function to evaluate core mass at the end of the MS as a fraction of the BGB value,
# i.e. this must be multiplied by the BGB value (see below) to give the actual core mass.
# [已校验] Hurley_2000: equation 5.1.2(29)
@conditional_njit()
def mctmsf(m):
    mctms = (1.586 + m ** 5.25) / (2.434 + 1.02 * m ** 5.25)
    return mctms

# A function to evaluate mass at BGB or He ignition (depending on mchefl) for IM & HM stars by inverting mcheif
@conditional_njit()
def mheif(mc, mhefl, mchefl, x):
    m1 = mbagbf(mc / 0.95, x)
    a3 = mchefl ** 4 - x.gbp[33] * mhefl ** x.gbp[34]
    m2 = ((mc ** 4 - a3) / x.gbp[33]) ** (1 / x.gbp[34])
    mhei = max(m1, m2)
    return mhei


# A function to evaluate mass at the BAGB by inverting mcagbf.
@conditional_njit()
def mbagbf(mc, x):
    mc4 = mc ** 4
    if mc4 > x.gbp[37]:
        mbagb = ((mc4 - x.gbp[37]) / x.gbp[35]) ** (1 / x.gbp[36])
    else:
        mbagb = 0
    return mbagb

# A function to evaluate L given t for GB, AGB and NHe stars
# [已校验] Hurley_2000: equation 5.2(35)
@conditional_njit()
def lgbtf(t, A , GB, tinf1, tinf2, tx):
    if t <= tx:
        lgbt = GB[4] * (((GB[5] - 1) * A * GB[4] * (tinf1 - t)) **(GB[5]/(1-GB[5])))
    else:
        lgbt = GB[3] * (((GB[6] - 1) * A * GB[3] * (tinf2 - t)) **(GB[6]/(1-GB[6])))
    return lgbt






# A function to evaluate the blue-loop fraction of the He-burning lifetime for IM & HM stars  (OP 28/01/98)
# [已校验] Hurley_2000: equation 5.3(58) 有些不太一样
@conditional_njit()
def tblf(m, mhefl, mfgb,x):
    mr = mhefl / mfgb
    if m <= mfgb:
        m1 = m / mfgb
        m2 = np.log10(m1) / np.log10(mr)
        m2 = max(m2, 1e-12)
        tbl = x.gbp[64] * m1 ** x.gbp[63] + x.gbp[65] * m2 ** x.gbp[62]
    else:
        r1 = 1 - rminf(m, x) / ragbf(m, lHeIf(m, mhefl, x), mhefl, x)
        r1 = max(r1, 1e-12)
        tbl = x.gbp[66] * m ** x.gbp[67] * r1 ** x.gbp[68]
    tbl = min(1, max(0, tbl))
    if tbl < 1e-10:
        tbl = 0
    return tbl








# 估算 He 星主序上的光度
# [已校验] Hurley_2000: equation 6.1(78)
@conditional_njit()
def l_He_MS(m):
    lzhe = 15262 * m ** 10.25 / (m ** 9 + 29.54 * m ** 7.5 + 31.18 * m ** 6 + 0.0469)
    return lzhe

# 根据质量、光度估算赫氏空隙中 He 星的半径
@conditional_njit()
def rhehgf(m, lum, rzhe, lthe):
    Lambda = 500 * (2 + m ** 5) / m ** 2.5
    rhehg = rzhe * (lum / lthe) ** 0.2 + 0.02 * (np.exp(lum / Lambda) - np.exp(lthe / Lambda))
    return rhehg


# 估算 He 巨星的半径
@conditional_njit()
def rhegbf(lum):
    rhegb = 0.08 * lum ** (3 / 4)
    return rhegb


# 估算光度扰动的指数(适用于非主序星)
@conditional_njit()
def lpert1f(m, mu):
    b = 0.002 * max(1, 2.5 / m)
    lpert = (1 + b ** 3) * ((mu / b) ** 3) / (1 + (mu / b) ** 3)
    return lpert


# 估算半径扰动的指数(适用于非主序星)
@conditional_njit()
def rpert1f(m, mu, r, rc):
    if mu <= 0:
        rpert = 0
    else:
        c = 0.006 * max(1, 2.5 / m)
        q = np.log(r / rc)
        fac = 0.1 / q
        facmax = -14 / np.log10(mu)
        fac = min(fac, facmax)
        rpert = ((1 + c ** 3) * ((mu / c) ** 3) * (mu ** fac)) / (1 + (mu / c) ** 3)
    return rpert


@conditional_njit()
def vrotf(m):
    vrot = 330 * m ** 3.3 / (15 + m ** 3.45)
    return vrot




