import numpy as np
from utils import conditional_njit
from const import yearsc, Msun, G, c, pc, num_evolve
import legwork as lw
import astropy.units as u
import astropy.coordinates.sky_coordinate as skycoord

# 集合了所有的独立函数（公式）


# 估算零龄主序光度 Lzams （from Tout et al., 1996, MNRAS, 281, 257）【已转移】
@conditional_njit()
def lzamsf(m, x):
    mx = np.sqrt(m)
    lzams = (x.msp[1] * m ** 5 * mx + x.msp[2] * m ** 11) / (
                x.msp[3] + m ** 3 + x.msp[4] * m ** 5 + x.msp[5] * m ** 7 + x.msp[6] * m ** 8 + x.msp[7] * m ** 9 * mx)
    return lzams


# 估算零龄主序半径 Rzams【已转移】
@conditional_njit()
def rzamsf(m, x):
    mx = np.sqrt(m)
    rzams = ((x.msp[8] * m ** 2 + x.msp[9] * m ** 6) * mx + x.msp[10] * m ** 11 + (
            x.msp[11] + x.msp[12] * mx) * m ** 19) / (x.msp[13] + x.msp[14] * m ** 2 + (
            x.msp[15] * m ** 8 + m ** 18 + x.msp[16] * m ** 19) * mx)
    return rzams


# A function to evaluate the lifetime to the BGB or to Helium ignition if no FGB exists. (JH 24/11/97)
# [已校验] Hurley_2000: equation 5.1(4)
@conditional_njit()
def tbgbf(m, x):
    tbgb = (x.msp[17] + x.msp[18] * m ** 4 + x.msp[19] * m ** (11 / 2) + m ** 7) / (
            x.msp[20] * m ** 2 + x.msp[21] * m ** 7)
    return tbgb


# A function to evaluate the derivitive of the lifetime to the BGB
# (or to Helium ignition if no FGB exists) wrt mass. (JH 24/11/97)
@conditional_njit()
def tbgbdf(m, x):
    mx = np.sqrt(m)
    f = x.msp[17] + x.msp[18] * m ** 4 + x.msp[19] * m ** 5 * mx + m ** 7
    df = 4 * x.msp[18] * m ** 3 + 5.5 * x.msp[19] * m ** 4 * mx + 7 * m ** 6
    g = x.msp[20] * m ** 2 + x.msp[21] * m ** 7
    dg = 2 * x.msp[20] * m + 7 * x.msp[21] * m ** 6
    tbgbd = (df * g - f * dg) / (g * g)
    return tbgbd


# A function to evaluate the derivitive of the lifetime to the BGB
# (or to Helium ignition if no FGB exists) wrt Z. (JH 14/12/98)
@conditional_njit()
def tbgdzf(m, x):
    mx = m ** 5 * np.sqrt(m)
    f = x.msp[17] + x.msp[18] * m ** 4 + x.msp[19] * mx + m ** 7
    df = x.msp[117] + x.msp[118] * m ** 4 + x.msp[119] * mx
    g = x.msp[20] * m ** 2 + x.msp[21] * m ** 7
    dg = x.msp[120] * m ** 2
    tbgdz = (df * g - f * dg) / (g * g)
    return tbgdz


# A function to evaluate the lifetime to the end of the MS hook as a fraction of the lifetime to the BGB
# (for those models that have one). Note that this function is only valid for M > Mhook.
# [已校验] Hurley_2000: equation 5.1(7)
@conditional_njit()
def thook_div_tBGB(m, x):
    term = 1 - 0.01 * max(x.msp[22] / m ** x.msp[23], x.msp[24] + x.msp[25] / m ** x.msp[26])
    value = max(0.5, term)
    return value


# 估算主序末尾的光度
# [已校验] Hurley_2000: equation 5.1(8)
@conditional_njit()
def ltmsf(m, x):
    ltms = (x.msp[27] * m ** 3 + x.msp[28] * m ** 4 + x.msp[29] * m ** (x.msp[32] + 1.8)) / (
            x.msp[30] + x.msp[31] * m ** 5 + m ** x.msp[32])
    return ltms


# 估算光度 alpha 系数
# [已校验] Hurley_2000: equation 5.1.1(19)
@conditional_njit()
def lalphaf(m, x):
    mcut = 2.0
    if m < 0.5:
        lalpha = x.msp[39]
    elif m < 0.7:
        lalpha = x.msp[39] + ((0.3 - x.msp[39]) / 0.2) * (m - 0.5)
    elif m < x.msp[37]:
        lalpha = 0.3 + ((x.msp[40] - 0.3) / (x.msp[37] - 0.7)) * (m - 0.7)
    elif m < x.msp[38]:
        lalpha = x.msp[40] + ((x.msp[41] - x.msp[40]) / (x.msp[38] - x.msp[37])) * (m - x.msp[37])
    elif m < mcut:
        lalpha = x.msp[41] + ((x.msp[42] - x.msp[41]) / (mcut - x.msp[38])) * (m - x.msp[38])
    else:
        lalpha = (x.msp[33] + x.msp[34] * m ** x.msp[36]) / (m ** 0.4 + x.msp[35] * m ** 1.9)
    return lalpha


# 估算光度 beta 系数
# [已校验] Hurley_2000: equation 5.1.1(20)
@conditional_njit()
def lbetaf(m, x):
    lbeta = max(0, x.msp[43] - x.msp[44] * m ** x.msp[45])
    if m > x.msp[46] and lbeta > 0:
        B = x.msp[43] - x.msp[44] * x.msp[46] ** x.msp[45]
        lbeta = max(0, B - 10 * B * (m - x.msp[46]))
    return lbeta


# 估算光度 neta 系数
# [已校验] Hurley_2000: equation 5.1.1(18)
@conditional_njit()
def lnetaf(m, x):
    if m <= 1:
        lneta = 10
    elif m >= 1.1:
        lneta = 20
    else:
        lneta = 10 + 100 * (m - 1)
    lneta = min(lneta, x.msp[97])
    return lneta


# A function to evalute the luminosity pertubation on the MS phase for M > Mhook. (JH 24/11/97)【我对这个函数的定义有改动】
# [已校验] Hurley_2000: equation 5.1.1(16)
@conditional_njit()
def lpertf(m, mhook, x):
    if m <= mhook:
        lhook = 0
    elif m >= x.msp[51]:
        lhook = min(x.msp[47] / m ** x.msp[48], x.msp[49] / m ** x.msp[50])
    else:
        B = min(x.msp[47] / x.msp[51] ** x.msp[48], x.msp[49] / x.msp[51] ** x.msp[50])
        lhook = B * ((m - mhook) / (x.msp[51] - mhook)) ** 0.4
    return lhook


# A function to evaluate the radius at the end of the MS
# Note that a safety check is added to ensure Rtms > Rzams when extrapolating the function to low masses. (JH 24/11/97)
# [已校验] Hurley_2000: equation 5.1(9)
@conditional_njit()
def rtmsf(m, x):
    if m <= x.msp[62]:
        rtms = (x.msp[52] + x.msp[53] * m ** x.msp[55]) / (x.msp[54] + m ** x.msp[56])
        # extrapolated to low mass(M < 0.5)
        rtms = max(rtms, 1.5 * rzamsf(m, x))
    elif m >= x.msp[62] + 0.1:
        rtms = (x.msp[57] * m ** 3 + x.msp[58] * m ** x.msp[61] + x.msp[59] * m ** (x.msp[61] + 1.5)) / (
                    x.msp[60] + m ** 5)
    else:
        rtms = x.msp[63] + ((m - x.msp[62]) / 0.1) * (x.msp[64] - x.msp[63])
    return rtms


# 估算半径 alpha 系数
# [已校验] Hurley_2000: equation 5.1.1(21)
@conditional_njit()
def ralphaf(m,x):
    if m <= 0.5:
        ralpha = x.msp[73]
    elif m <= 0.65:
        ralpha = x.msp[73] + ((x.msp[74] - x.msp[73]) / 0.15) * (m - 0.5)
    elif m <= x.msp[70]:
        ralpha = x.msp[74] + ((x.msp[75]-x.msp[74])/(x.msp[70]-0.65))*(m - 0.65)
    elif m <= x.msp[71]:
        ralpha = x.msp[75] + ((x.msp[76] - x.msp[75])/(x.msp[71] - x.msp[70]))*(m - x.msp[70])
    elif m <= x.msp[72]:
        ralpha = (x.msp[65]*m**x.msp[67])/(x.msp[66] + m**x.msp[68])
    else:
        a5 = (x.msp[65] * x.msp[72] ** x.msp[67]) / (x.msp[66] + x.msp[72] ** x.msp[68])
        ralpha = a5 + x.msp[69] * (m - x.msp[72])
    return ralpha


# 估算半径 beta 系数
# [已校验] Hurley_2000: equation 5.1.1(22)
@conditional_njit()
def rbetaf(m,x):
    m2 = 2
    m3 = 16
    if m <= 1:
        rbeta = 1.06
    elif m <= x.msp[82]:
        rbeta = 1.06 + ((x.msp[81] - 1.06) / (x.msp[82] - 1)) * (m - 1)
    elif m <= m2:
        b2 = (x.msp[77] * m2 ** (7 / 2)) / (x.msp[78] + m2 ** x.msp[79])
        rbeta = x.msp[81] + ((b2 - x.msp[81]) / (m2 - x.msp[82])) * (m - x.msp[82])
    elif m <= m3:
        rbeta = (x.msp[77] * m ** (7 / 2)) / (x.msp[78] + m ** x.msp[79])
    else:
        b3 = (x.msp[77] * m3 ** (7 / 2)) / (x.msp[78] + m3 ** x.msp[79])
        rbeta = b3 + x.msp[80] * (m - m3)
    rbeta = rbeta - 1
    return rbeta


# 估算半径 gamma 系数
# [已校验] Hurley_2000: equation 5.1.1(23)
@conditional_njit()
def rgammaf(m, x):
    m1 = 1
    b1 = max(0, x.msp[83] + x.msp[84] * (m1 - x.msp[85]) ** x.msp[86])
    if m <= m1:
        rgamma = x.msp[83] + x.msp[84] * abs(m - x.msp[85]) ** x.msp[86]
    elif m1 < m <= x.msp[88]:
        rgamma = b1 + (x.msp[89] - b1) * ((m - m1) / (x.msp[88] - m1)) ** x.msp[87]
    elif x.msp[88] < m <= x.msp[88] + 0.1:
        if x.msp[88] > m1:
            b1 = x.msp[89]
        rgamma = b1 - 10 * b1 * (m - x.msp[88])
    else:
        rgamma = 0
    rgamma = max(rgamma, 0)
    return rgamma


# A function to evalute the radius pertubation on the MS phase for M > Mhook. (JH 24/11/97)【我对这个函数的定义有改动】
# [已校验] Hurley_2000: equation 5.1.1(17)
@conditional_njit()
def rpertf(m, mhook, x):
    if m <= mhook:
        rhook = 0
    elif m <= x.msp[94]:
        rhook = x.msp[95] * np.sqrt((m - mhook) / (x.msp[94] - mhook))
    elif m <= 2:
        m1 = 2
        B = (x.msp[90] + x.msp[91] * m1 ** (7 / 2)) / (x.msp[92] * m1 ** 3 + m1 ** x.msp[93]) - 1
        rhook = x.msp[95] + (B - x.msp[95]) * ((m - x.msp[94]) / (m1 - x.msp[94])) ** x.msp[96]
    else:
        rhook = (x.msp[90] + x.msp[91] * m ** (7 / 2)) / (x.msp[92] * m ** 3 + m ** x.msp[93]) - 1
    return rhook


# A function to evaluate the luminosity at the base of Giant Branch (for those models that have one)
# Note that this function is only valid for LM & IM stars
# [已校验] Hurley_2000: equation 5.1(10)
@conditional_njit()
def lbgbf(m, x):
    lbgb = (x.gbp[1] * m ** x.gbp[5] + x.gbp[2] * m ** x.gbp[8]) / (x.gbp[3] + x.gbp[4]*m**x.gbp[7] + m**x.gbp[6])
    return lbgb


# A function to evaluate the derivitive of the Lbgb function.
# Note that this function is only valid for LM & IM stars
@conditional_njit()
def lbgbdf(m, x):
    f = x.gbp[1] * m ** x.gbp[5] + x.gbp[2] * m ** x.gbp[8]
    df = x.gbp[5] * x.gbp[1] * m ** (x.gbp[5] - 1) + x.gbp[8] * x.gbp[2] * m ** (x.gbp[8] - 1)
    g = x.gbp[3] + x.gbp[4] * m ** x.gbp[7] + m ** x.gbp[6]
    dg = x.gbp[7] * x.gbp[4] * m ** (x.gbp[7] - 1) + x.gbp[6] * m ** (x.gbp[6] - 1)
    lbgbd = (df * g - f * dg) / (g * g)
    return lbgbd


# A function to evaluate the BAGB luminosity. (OP 21/04/98)
# Continuity between LM and IM functions is ensured by setting gbp(16) = lbagbf(mhefl,0.0) with gbp(16) = 1.0.
# [已校验] Hurley_2000: equation 5.3(56) 第三行有出入
@conditional_njit()
def lbagbf(m, mhefl, x):
    a4 = (x.gbp[9] * mhefl ** x.gbp[10] - x.gbp[16]) / (np.exp(mhefl * x.gbp[11]) * x.gbp[16])
    if m < mhefl:
        lbagb = x.gbp[9] * m ** x.gbp[10] / (1 + a4 * np.exp(m * x.gbp[11]))
    else:
        lbagb = (x.gbp[12] + x.gbp[13] * m ** (x.gbp[15] + 1.8)) / (x.gbp[14] + m ** x.gbp[15])
    return lbagb


# 估算巨星分支上的半径
# [已校验] Hurley_2000: equation 5.2(46)
@conditional_njit()
def rgbf(m, lum, x):
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


# A function to evaluate core mass at BGB or He ignition (depending on mchefl) for IM & HM stars
# [已校验] Hurley_2000: equation 5.2(44)
@conditional_njit()
def mcheif(m, mhefl, mchefl, x):
    mcbagb = mcagbf(m, x)
    a3 = mchefl ** 4 - x.gbp[33] * mhefl ** x.gbp[34]
    mchei = min(0.95 * mcbagb, (a3 + x.gbp[33] * m ** x.gbp[34]) ** (1 / 4))
    return mchei


# A function to evaluate mass at BGB or He ignition (depending on mchefl) for IM & HM stars by inverting mcheif
@conditional_njit()
def mheif(mc, mhefl, mchefl, x):
    m1 = mbagbf(mc / 0.95, x)
    a3 = mchefl ** 4 - x.gbp[33] * mhefl ** x.gbp[34]
    m2 = ((mc ** 4 - a3) / x.gbp[33]) ** (1 / x.gbp[34])
    mhei = max(m1, m2)
    return mhei


# A function to evaluate core mass at the BAGB  (OP 25/11/97)
# [已校验] Hurley_2000: equation 5.3(66)
@conditional_njit()
def mcagbf(m, x):
    mcagb = (x.gbp[37] + x.gbp[35] * m ** x.gbp[36]) ** (1 / 4)
    return mcagb


# A function to evaluate mass at the BAGB by inverting mcagbf.
@conditional_njit()
def mbagbf(mc, x):
    mc4 = mc ** 4
    if mc4 > x.gbp[37]:
        mbagb = ((mc4 - x.gbp[37]) / x.gbp[35]) ** (1 / x.gbp[36])
    else:
        mbagb = 0
    return mbagb


# A function to evaluate Mc given t for GB, AGB and NHe stars
# [已校验] Hurley_2000: equation 5.2(34、39)
@conditional_njit()
def mcgbtf(t, A, GB, tinf1, tinf2, tx):
    if t <= tx:
        mcgbt = ((GB[5] - 1) * A * GB[4] * (tinf1 - t)) ** (1/(1-GB[5]))
    else:
        mcgbt = ((GB[6] - 1) * A * GB[3] * (tinf2 - t)) ** (1/(1-GB[6]))
    return mcgbt


# A function to evaluate L given t for GB, AGB and NHe stars
# [已校验] Hurley_2000: equation 5.2(35)
@conditional_njit()
def lgbtf(t, A , GB, tinf1, tinf2, tx):
    if t <= tx:
        lgbt = GB[4] * (((GB[5] - 1) * A * GB[4] * (tinf1 - t)) **(GB[5]/(1-GB[5])))
    else:
        lgbt = GB[3] * (((GB[6] - 1) * A * GB[3] * (tinf2 - t)) **(GB[6]/(1-GB[6])))
    return lgbt


# 通过光度估算 GB, AGB and NHe stars 的 Mc
# [已校验] Hurley_2000: equation 5.2(37)等效
@conditional_njit()
def lum_to_mc_gb(lum, GB, lx):
    if lum <= lx:
        mc = (lum / GB[4]) ** (1 / GB[5])
    else:
        mc = (lum / GB[3]) ** (1 / GB[6])
    return mc


# 通过 Mc 估算 GB, AGB and Naked He stars 的光度
# [已校验] Hurley_2000: equation 5.2(37)
@conditional_njit()
def mc_to_lum_gb(mc, GB):
    if mc <= GB[7]:
        lum = GB[4] * (mc ** GB[5])
    else:
        lum = GB[3] * (mc ** GB[6])
    return lum


# A function to evaluate He-ignition luminosity  (OP 24/11/97)
# Continuity between the LM and IM functions is ensured with a first call setting lhefl = lHeIf(mhefl,0.0)
# [已校验] Hurley_2000: equation 5.3(49) 第二行有出入
@conditional_njit()
def lHeIf(m, mhefl, x):
    if m < mhefl:
        lHeI = x.gbp[38] * m ** x.gbp[39] / (1 + x.gbp[41] * np.exp(m * x.gbp[40]))
    else:
        lHeI = (x.gbp[42] + x.gbp[43] * m ** 3.8) / (x.gbp[44] + m ** 2)
    return lHeI


# A function to evaluate the ratio LHe,min/LHeI  (OP 20/11/97)
# Note that this function is everywhere <= 1, and is only valid for IM stars
# [已校验] Hurley_2000: equation 5.3(51)
@conditional_njit()
def lHef(m, x):
    lHe = (x.gbp[45] + x.gbp[46] * m ** (x.gbp[48] + 0.1)) / (x.gbp[47] + m ** x.gbp[48])
    return lHe


# A function to evaluate the minimum radius during blue loop(He-burning) for IM & HM stars
# [已校验] Hurley_2000: equation 5.3(55)
@conditional_njit()
def rminf(m, x):
    rmin = (x.gbp[49] * m + (x.gbp[50] * m) ** x.gbp[52] * m ** x.gbp[53]) / (x.gbp[51] + m ** x.gbp[53])
    return rmin


# A function to evaluate the He-burning lifetime.
# For IM & HM stars, tHef is relative to tBGB.
# Continuity between LM and IM stars is ensured by setting thefl = tHef(mhefl,0.0,0.0)
# the call to themsf ensures continuity between HB and NHe stars as Menv -> 0.
# [已校验] Hurley_2000: equation 5.3(57)
@conditional_njit()
def tHef(m, mc, mhefl, x):
    if m <= mhefl:
        mm = max((mhefl - m) / (mhefl - mc), 1e-12)
        tHe = (x.gbp[54] + (themsf(mc) - x.gbp[54]) * mm ** x.gbp[55]) * (1 + x.gbp[57] * np.exp(m * x.gbp[56]))
    else:
        tHe = (x.gbp[58] * m ** x.gbp[61] + x.gbp[59] * m ** 5) / (x.gbp[60] + m ** 5)
    return tHe


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


# A function to evaluate the ZAHB luminosity for LM stars. (OP 28/01/98)
# Continuity with LHe, min for IM stars is ensured by setting lx = lHeif(mhefl,z,0.0,1.0)*lHef(mhefl,z,mfgb)
# and the call to lzhef ensures continuity between the ZAHB and the NHe-ZAMS as Menv -> 0.
# [已校验] Hurley_2000: equation 5.3(53)
@conditional_njit()
def lzahbf(m, mc, mhefl, x):
    a5 = lzhef(mc)
    a4 = (x.gbp[69] + a5 - x.gbp[74]) / ((x.gbp[74] - a5) * np.exp(x.gbp[71] * mhefl))
    mm = max((m - mc) / (mhefl - mc), 1e-12)
    lzahb = a5 + (1 + x.gbp[72]) * x.gbp[69] * mm ** x.gbp[70] / (
            (1 + x.gbp[72] * mm ** x.gbp[73]) * (1 + a4 * np.exp(m * x.gbp[71])))
    return lzahb


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


# 估算 He星零龄主序的光度
# [已校验] Hurley_2000: equation 6.1(77)
@conditional_njit()
def lzhef(m):
    lzhe = 15262 * m ** 10.25 / (m ** 9 + 29.54 * m ** 7.5 + 31.18 * m ** 6 + 0.0469)
    return lzhe


# 估算 He 星零龄主序的半径
# [已校验] Hurley_2000: equation 6.1(78)
@conditional_njit()
def rzhef(m):
    rzhe = 0.2391 * m ** 4.6 / (m ** 4 + 0.162 * m ** 3 + 0.0065)
    return rzhe


# 估算 He 星的主序时间
# [已校验] Hurley_2000: equation 6.1(79)
@conditional_njit()
def themsf(m):
    thems = (0.4129 + 18.81 * m ** 4 + 1.853 * m ** 6) / m ** 6.5
    return thems


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




