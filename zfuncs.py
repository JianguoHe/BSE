import numpy as np
from utils import conditional_njit
from const import yearsc, Msun, G, c, pc, num_evolve
import legwork as lw
import astropy.units as u
import astropy.coordinates.sky_coordinate as skycoord

# 集合了所有的独立函数（公式）






# 估算巨星分支上的半径
# [已校验] Hurley_2000: equation 5.2(46)
@conditional_njit()
def rgbf(self, lum):
    a = min(x.gbp[20] / m ** x.gbp[21], x.gbp[22] / m ** x.gbp[23])
    rgb = a * (lum ** x.gbp[18] + x.gbp[17] * lum ** x.gbp[19])
    return rgb


# A function to evaluate radius derivitive on the GB (as f(L)).
@conditional_njit()
def rgbdf(m, lum, x):
    a1 = min(x.gbp[20] / m ** x.gbp[21], x.gbp[22] / m ** x.gbp[23])
    rgbd = a1 * (x.gbp[18] * lum ** (x.gbp[18] - 1) + x.gbp[17] * x.gbp[19] * lum ** (x.gbp[19] - 1))
    return rgbd


# 估算渐近巨星分支上的半径
# [已校验] Hurley_2000: equation 5.4(74)
@conditional_njit()
def ragbf(m, lum, mhef, x):
    m1 = mhef - 0.2
    if m <= m1:
        b50 = x.gbp[19]
        A = x.gbp[29] + x.gbp[30] * m
    elif m >= mhef:
        b50 = x.gbp[19] * x.gbp[24]
        A = min(x.gbp[25] / m ** x.gbp[26], x.gbp[27] / m ** x.gbp[28])
    else:
        b50 = x.gbp[19] * (1 + (x.gbp[24] - 1) * (m - m1) / 0.2)
        A = x.gbp[31] + (x.gbp[32] - x.gbp[31]) * (m - m1) / 0.2
    ragb = A * (lum ** x.gbp[18] + x.gbp[17] * lum ** b50)
    return ragb


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










# A function to evaluate the minimum radius during blue loop(He-burning) for IM & HM stars
# [已校验] Hurley_2000: equation 5.3(55)
@conditional_njit()
def rminf(m, x):
    rmin = (x.gbp[49] * m + (x.gbp[50] * m) ** x.gbp[52] * m ** x.gbp[53]) / (x.gbp[51] + m ** x.gbp[53])
    return rmin



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

# 估算低质量恒星的零龄水平分支(ZAHB)半径
# Continuity with R(LHe,min) for IM stars is ensured by setting lx = lHeif(mhefl,z,0.0,1.0)*lHef(mhefl,z,mfgb),
# and the call to rzhef ensures continuity between the ZAHB and the NHe-ZAMS as Menv -> 0.
# [已校验] Hurley_2000: equation 5.3(54)
@conditional_njit()
def rzahbf(m, mc, mhefl, x):
    rx = rzhef(mc)
    ry = rgbf(m, lzahbf(m, mc, mhefl, x), x)
    mm = max((m - mc) / (mhefl - mc), 1e-12)
    f = (1 + x.gbp[76]) * mm ** x.gbp[75] / (1 + x.gbp[76] * mm ** x.gbp[77])
    rzahb = (1 - f) * rx + f * ry
    return rzahb




# 估算 He 星零龄主序的半径
# [已校验] Hurley_2000: equation 6.1(78)
@conditional_njit()
def rzhef(m):
    rzhe = 0.2391 * m ** 4.6 / (m ** 4 + 0.162 * m ** 3 + 0.0065)
    return rzhe





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




