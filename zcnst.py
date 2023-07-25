import numpy as np
from zdata import zdata
from utils import conditional_njit
from star import star
from zfuncs import lbagbf, rminf, lum_to_mc_gb, lHeIf, ragbf, lzahbf, rgbf, lHef, tHef, rtmsf


# 设置所有与金属丰度z有关的公式中的常数（其他地方不再依赖于z）
# zpars:   1; M below which hook doesn't appear on MS, Mhook.
#          2; M above which He ignition occurs non-degenerately, Mhef.
#          3; M above which He ignition occurs on the HG, Mfgb.
#          4; M below which C/O ignition doesn't occur, Mup.  (程序中未用到)
#          5; M above which C ignites in the centre, Mec.     (程序中未用到)
#          6; value of log D for M <= Mhef
#          7; value of x for Rgb propto M^(-x)
#          8; value of x for tMS = MAX(tHOOK,x*tBGB)
#          9; constant for McHeIf when computing Mc on BGB, mchefl.
#          10; constant for McHeIf when computing Mc on HeI, mchefl.
#          11; hydrogen abundance.
#          12; helium abundance.
#          13; constant x in rmin = rgb*x**y used by LM CHeB.
#          14; z**0.4 to be used for WD L formula.

# 读取初始金属丰度
# zpars = np.zeros((1, 20)).flatten()
# a = msp = np.zeros((1, 200)).flatten()
# b = gbp = np.zeros((1, 200)).flatten()

@conditional_njit()
def zcnsts_set(x):
    # 这里的 x 就是 Zcnsts 类的实例 zcnsts
    # from star import star
    z = x.z
    lzs = np.log10(z / 0.02)
    dlzs = 1 / (z * np.log(10))
    lz = np.log10(z)
    lzd = lzs + 1
    c = ([0, 3.040581e-01, 8.049509e-02, 8.967485e-02, 8.780198e-02, 2.219170e-02])
    x.zpars[1] = 1.0185 + lzs * (0.16015 + lzs * 0.0892)
    x.zpars[2] = 1.995 + lzs * (0.25 + lzs * 0.087)
    x.zpars[3] = 13.048 * (z / 0.02) ** 0.06 / (1 + 0.0012 * (0.02 / z) ** 1.27)
    x.zpars[4] = max(6.11044 + 1.02167 * lzs, 5)
    x.zpars[5] = x.zpars[4] + 1.8
    x.zpars[6] = 5.37 + lzs * 0.135
    x.zpars[7] = c[1] + lzs * (c[2] + lzs * (c[3] + lzs * (c[4] + lzs * c[5])))
    x.zpars[8] = max(0.95, max(0.95 - (10 / 3) * (z - 0.01), min(0.99, 0.98 - (100 / 7) * (z - 0.001))))

    zdata_return = zdata()
    xz = zdata_return[0]
    xt = zdata_return[1]
    xl = zdata_return[2]
    xr = zdata_return[3]
    xg = zdata_return[4]
    xh = zdata_return[5]

    # msp(main sequence parameter)
    # Lzams
    x.msp[1] = xz[1] + lzs * (xz[2] + lzs * (xz[3] + lzs * (xz[4] + lzs * xz[5])))
    x.msp[2] = xz[6] + lzs * (xz[7] + lzs * (xz[8] + lzs * (xz[9] + lzs * xz[10])))
    x.msp[3] = xz[11] + lzs * (xz[12] + lzs * (xz[13] + lzs * (xz[14] + lzs * xz[15])))
    x.msp[4] = xz[16] + lzs * (xz[17] + lzs * (xz[18] + lzs * (xz[19] + lzs * xz[20])))
    x.msp[5] = xz[21] + lzs * (xz[22] + lzs * (xz[23] + lzs * (xz[24] + lzs * xz[25])))
    x.msp[6] = xz[26] + lzs * (xz[27] + lzs * (xz[28] + lzs * (xz[29] + lzs * xz[30])))
    x.msp[7] = xz[31] + lzs * (xz[32] + lzs * (xz[33] + lzs * (xz[34] + lzs * xz[35])))

    # Rzams
    x.msp[8] = xz[36] + lzs * (xz[37] + lzs * (xz[38] + lzs * (xz[39] + lzs * xz[40])))
    x.msp[9] = xz[41] + lzs * (xz[42] + lzs * (xz[43] + lzs * (xz[44] + lzs * xz[45])))
    x.msp[10] = xz[46] + lzs * (xz[47] + lzs * (xz[48] + lzs * (xz[49] + lzs * xz[50])))
    x.msp[11] = xz[51] + lzs * (xz[52] + lzs * (xz[53] + lzs * (xz[54] + lzs * xz[55])))
    x.msp[12] = xz[56] + lzs * (xz[57] + lzs * (xz[58] + lzs * (xz[59] + lzs * xz[60])))
    x.msp[13] = xz[61]
    x.msp[14] = xz[62] + lzs * (xz[63] + lzs * (xz[64] + lzs * (xz[65] + lzs * xz[66])))
    x.msp[15] = xz[67] + lzs * (xz[68] + lzs * (xz[69] + lzs * (xz[70] + lzs * xz[71])))
    x.msp[16] = xz[72] + lzs * (xz[73] + lzs * (xz[74] + lzs * (xz[75] + lzs * xz[76])))

    # Tbgb
    x.msp[17] = xt[1] + lzs * (xt[2] + lzs * (xt[3] + lzs * xt[4]))
    x.msp[18] = xt[5] + lzs * (xt[6] + lzs * (xt[7] + lzs * xt[8]))
    x.msp[19] = xt[9] + lzs * (xt[10] + lzs * (xt[11] + lzs * xt[12]))
    x.msp[20] = xt[13] + lzs * (xt[14] + lzs * (xt[15] + lzs * xt[16]))
    x.msp[21] = xt[17]

    # dTbgb/dz
    x.msp[117] = dlzs * (xt[2] + lzs * (2 * xt[3] + 3 * lzs * xt[4]))
    x.msp[118] = dlzs * (xt[6] + lzs * (2 * xt[7] + 3 * lzs * xt[8]))
    x.msp[119] = dlzs * (xt[10] + lzs * (2 * xt[11] + 3 * lzs * xt[12]))
    x.msp[120] = dlzs * (xt[14] + lzs * (2 * xt[15] + 3 * lzs * xt[16]))

    # Thook
    x.msp[22] = xt[18] + lzs * (xt[19] + lzs * (xt[20] + lzs * xt[21]))
    x.msp[23] = xt[22]
    x.msp[24] = xt[23] + lzs * (xt[24] + lzs * (xt[25] + lzs * xt[26]))
    x.msp[25] = xt[27] + lzs * (xt[28] + lzs * (xt[29] + lzs * xt[30]))
    x.msp[26] = xt[31]

    # Ltms
    x.msp[27] = xl[1] + lzs * (xl[2] + lzs * (xl[3] + lzs * (xl[4] + lzs * xl[5])))
    x.msp[28] = xl[6] + lzs * (xl[7] + lzs * (xl[8] + lzs * (xl[9] + lzs * xl[10])))
    x.msp[29] = xl[11] + lzs * (xl[12] + lzs * (xl[13] + lzs * xl[14]))
    x.msp[30] = xl[15] + lzs * (xl[16] + lzs * (xl[17] + lzs * (xl[18] + lzs * xl[19])))
    x.msp[27] = x.msp[27] * x.msp[30]
    x.msp[28] = x.msp[28] * x.msp[30]
    x.msp[31] = xl[20] + lzs * (xl[21] + lzs * (xl[22] + lzs * xl[23]))
    x.msp[32] = xl[24] + lzs * (xl[25] + lzs * (xl[26] + lzs * xl[27]))

    # Lalpha
    m1 = 2
    x.msp[33] = xl[28] + lzs * (xl[29] + lzs * (xl[30] + lzs * xl[31]))
    x.msp[34] = xl[32] + lzs * (xl[33] + lzs * (xl[34] + lzs * xl[35]))
    x.msp[35] = xl[36] + lzs * (xl[37] + lzs * (xl[38] + lzs * xl[39]))
    x.msp[36] = xl[40] + lzs * (xl[41] + lzs * (xl[42] + lzs * xl[43]))
    x.msp[37] = max(0.9, 1.1064 + lzs * (0.415 + 0.18 * lzs))
    x.msp[38] = max(1, 1.19 + lzs * (0.377 + 0.176 * lzs))
    if z > 0.01:
        x.msp[37] = min(x.msp[37], 1)
        x.msp[38] = min(x.msp[38], 1.1)
    x.msp[39] = max(0.145, 0.0977 - lzs * (0.231 + 0.0753 * lzs))
    x.msp[40] = min(0.24 + lzs * (0.18 + 0.595 * lzs), 0.306 + 0.053 * lzs)
    x.msp[41] = min(0.33 + lzs * (0.132 + 0.218 * lzs), 0.3625 + 0.062 * lzs)
    x.msp[42] = (x.msp[33] + x.msp[34] * m1 ** x.msp[36]) / (m1 ** 0.4 + x.msp[35] * m1 ** 1.9)

    # Lbeta
    x.msp[43] = xl[44] + lzs * (xl[45] + lzs * (xl[46] + lzs * (xl[47] + lzs * xl[48])))
    x.msp[44] = xl[49] + lzs * (xl[50] + lzs * (xl[51] + lzs * (xl[52] + lzs * xl[53])))
    x.msp[45] = xl[54] + lzs * (xl[55] + lzs * xl[56])
    x.msp[46] = min(1.4, 1.5135 + 0.3769 * lzs)
    x.msp[46] = max(0.6355 - 0.4192 * lzs, max(1.25, x.msp[46]))

    # Lhook
    x.msp[47] = xl[57] + lzs * (xl[58] + lzs * (xl[59] + lzs * xl[60]))
    x.msp[48] = xl[61] + lzs * (xl[62] + lzs * (xl[63] + lzs * xl[64]))
    x.msp[49] = xl[65] + lzs * (xl[66] + lzs * (xl[67] + lzs * xl[68]))
    x.msp[50] = xl[69] + lzs * (xl[70] + lzs * (xl[71] + lzs * xl[72]))
    x.msp[51] = min(1.4, 1.5135 + 0.3769 * lzs)
    x.msp[51] = max(0.6355 - 0.4192 * lzs, max(1.25, x.msp[51]))

    # Rtms
    x.msp[52] = xr[1] + lzs * (xr[2] + lzs * (xr[3] + lzs * (xr[4] + lzs * xr[5])))
    x.msp[53] = xr[6] + lzs * (xr[7] + lzs * (xr[8] + lzs * (xr[9] + lzs * xr[10])))
    x.msp[54] = xr[11] + lzs * (xr[12] + lzs * (xr[13] + lzs * (xr[14] + lzs * xr[15])))
    x.msp[55] = xr[16] + lzs * (xr[17] + lzs * (xr[18] + lzs * xr[19]))
    x.msp[56] = xr[20] + lzs * (xr[21] + lzs * (xr[22] + lzs * xr[23]))
    x.msp[52] = x.msp[52] * x.msp[54]
    x.msp[53] = x.msp[53] * x.msp[54]
    x.msp[57] = xr[24]
    x.msp[58] = xr[25] + lzs * (xr[26] + lzs * (xr[27] + lzs * xr[28]))
    x.msp[59] = xr[29] + lzs * (xr[30] + lzs * (xr[31] + lzs * xr[32]))
    x.msp[60] = xr[33] + lzs * (xr[34] + lzs * (xr[35] + lzs * xr[36]))
    x.msp[61] = xr[37] + lzs * (xr[38] + lzs * (xr[39] + lzs * xr[40]))
    x.msp[62] = max(0.097 - 0.1072 * (lz + 3), max(0.097, min(0.1461, 0.1461 + 0.1237 * (lz + 2))))
    x.msp[62] = 10 ** x.msp[62]
    x.msp[63] = rtmsf(x.msp[62], x)
    x.msp[64] = rtmsf(x.msp[62] + 0.1, x)

    # Ralpha
    x.msp[65] = xr[41] + lzs * (xr[42] + lzs * (xr[43] + lzs * xr[44]))
    x.msp[66] = xr[45] + lzs * (xr[46] + lzs * (xr[47] + lzs * xr[48]))
    x.msp[67] = xr[49] + lzs * (xr[50] + lzs * (xr[51] + lzs * xr[52]))
    x.msp[68] = xr[53] + lzs * (xr[54] + lzs * (xr[55] + lzs * xr[56]))
    x.msp[69] = xr[57] + lzs * (xr[58] + lzs * (xr[59] + lzs * (xr[60] + lzs * xr[61])))
    x.msp[70] = max(0.9, min(1, 1.116 + 0.166 * lzs))
    x.msp[71] = max(1.477 + 0.296 * lzs, min(1.6, -0.308 - 1.046 * lzs))
    x.msp[71] = max(0.8, min(0.8 - 2 * lzs, x.msp[71]))
    x.msp[72] = xr[62] + lzs * (xr[63] + lzs * xr[64])
    x.msp[73] = max(0.065, 0.0843 - lzs * (0.0475 + 0.0352 * lzs))
    x.msp[74] = 0.0736 + lzs * (0.0749 + 0.04426 * lzs)
    if z < 0.004:
        x.msp[74] = min(0.055, x.msp[74])
    x.msp[75] = max(0.091, min(0.121, 0.136 + 0.0352 * lzs))
    x.msp[76] = (x.msp[65] * x.msp[71] ** x.msp[67]) / (x.msp[66] + x.msp[71] ** x.msp[68])
    if x.msp[70] > x.msp[71]:
        x.msp[70] = x.msp[71]
        x.msp[75] = x.msp[76]

    # Rbeta
    x.msp[77] = xr[65] + lzs * (xr[66] + lzs * (xr[67] + lzs * xr[68]))
    x.msp[78] = xr[69] + lzs * (xr[70] + lzs * (xr[71] + lzs * xr[72]))
    x.msp[79] = xr[73] + lzs * (xr[74] + lzs * (xr[75] + lzs * xr[76]))
    x.msp[80] = xr[77] + lzs * (xr[78] + lzs * (xr[79] + lzs * xr[80]))
    x.msp[81] = xr[81] + lzs * (xr[82] + lzs * lzs * xr[83])
    if z > 0.01:
        x.msp[81] = max(x.msp[81], 0.95)
    x.msp[82] = max(1.4, min(1.6, 1.6 + lzs * (0.764 + 0.3322 * lzs)))

    # Rgamma
    x.msp[83] = max(xr[84] + lzs * (xr[85] + lzs * (xr[86] + lzs * xr[87])), xr[96] + lzs * (xr[97] + lzs * xr[98]))
    x.msp[84] = min(0, xr[88] + lzs * (xr[89] + lzs * (xr[90] + lzs * xr[91])))
    x.msp[84] = max(x.msp[84], xr[99] + lzs * (xr[100] + lzs * xr[101]))
    x.msp[85] = xr[92] + lzs * (xr[93] + lzs * (xr[94] + lzs * xr[95]))
    x.msp[85] = max(0, min(x.msp[85], 7.454 + 9.046 * lzs))
    x.msp[86] = min(xr[102] + lzs * xr[103], max(2, -13.3 - 18.6 * lzs))
    x.msp[87] = min(1.5, max(0.4, 2.493 + 1.1475 * lzs))
    x.msp[88] = max(1.0, 0.6355 - 0.4192 * lzs, min(1.27, 0.8109 - 0.6282 * lzs))
    x.msp[89] = max(5.855420e-02, -0.2711 - lzs * (0.5756 + 0.0838 * lzs))

    # Rhook
    x.msp[90] = xr[104] + lzs * (xr[105] + lzs * (xr[106] + lzs * xr[107]))
    x.msp[91] = xr[108] + lzs * (xr[109] + lzs * (xr[110] + lzs * xr[111]))
    x.msp[92] = xr[112] + lzs * (xr[113] + lzs * (xr[114] + lzs * xr[115]))
    x.msp[93] = xr[116] + lzs * (xr[117] + lzs * (xr[118] + lzs * xr[119]))
    x.msp[94] = min(1.25, max(1.1, 1.9848 + lzs * (1.1386 + 0.3564 * lzs)))
    x.msp[95] = 0.063 + lzs * (0.0481 + 0.00984 * lzs)
    x.msp[96] = min(1.3, max(0.45, 1.2 + 2.45 * lzs))

    # Lneta
    if z > 0.0009:
        x.msp[97] = 10
    else:
        x.msp[97] = 20

    # gbp(giant branch parameter)
    # Lbgb
    x.gbp[1] = xg[1] + lzs * (xg[2] + lzs * (xg[3] + lzs * xg[4]))
    x.gbp[2] = xg[5] + lzs * (xg[6] + lzs * (xg[7] + lzs * xg[8]))
    x.gbp[3] = xg[9] + lzs * (xg[10] + lzs * (xg[11] + lzs * xg[12]))
    x.gbp[4] = xg[13] + lzs * (xg[14] + lzs * (xg[15] + lzs * xg[16]))
    x.gbp[5] = xg[17] + lzs * (xg[18] + lzs * xg[19])
    x.gbp[6] = xg[20] + lzs * (xg[21] + lzs * xg[22])
    x.gbp[3] = x.gbp[3] ** x.gbp[6]
    x.gbp[7] = xg[23]
    x.gbp[8] = xg[24]

    # Lbagb
    # set x.gbp[16] = 1 until it is reset later with an initial call to Lbagbf using mass = zpars(2) and mhefl = 0.0
    x.gbp[9] = xg[25] + lzs * (xg[26] + lzs * xg[27])
    x.gbp[10] = xg[28] + lzs * (xg[29] + lzs * xg[30])
    x.gbp[11] = 15
    x.gbp[12] = xg[31] + lzs * (xg[32] + lzs * (xg[33] + lzs * xg[34]))
    x.gbp[13] = xg[35] + lzs * (xg[36] + lzs * (xg[37] + lzs * xg[38]))
    x.gbp[14] = xg[39] + lzs * (xg[40] + lzs * (xg[41] + lzs * xg[42]))
    x.gbp[15] = xg[43] + lzs * xg[44]
    x.gbp[12] = x.gbp[12] ** x.gbp[15]
    x.gbp[14] = x.gbp[14] ** x.gbp[15]
    x.gbp[16] = 1

    # Rgb(radius on giant branch)
    x.gbp[17] = -4.6739 - 0.9394 * lz
    x.gbp[17] = 10 ** x.gbp[17]
    x.gbp[17] = max(x.gbp[17], -0.04167 + 55.67 * z)
    x.gbp[17] = min(x.gbp[17], 0.4771 - 9329.21 * z ** 2.94)
    x.gbp[18] = min(0.54, 0.397 + lzs * (0.28826 + 0.5293 * lzs))
    x.gbp[19] = max(-0.1451, -2.2794 - lz * (1.5175 + 0.254 * lz))
    x.gbp[19] = 10 ** x.gbp[19]
    if z > 0.004:
        x.gbp[19] = max(x.gbp[19], 0.7307 + 14265.1 * z ** 3.395)
    x.gbp[20] = xg[45] + lzs * (xg[46] + lzs * (xg[47] + lzs * (xg[48] + lzs * (xg[49] + lzs * xg[50]))))
    x.gbp[21] = xg[51] + lzs * (xg[52] + lzs * (xg[53] + lzs * (xg[54] + lzs * xg[55])))
    x.gbp[22] = xg[56] + lzs * (xg[57] + lzs * (xg[58] + lzs * (xg[59] + lzs * (xg[60] + lzs * xg[61]))))
    x.gbp[23] = xg[62] + lzs * (xg[63] + lzs * (xg[64] + lzs * (xg[65] + lzs * xg[66])))

    # Ragb
    x.gbp[24] = min(0.99164 - 743.123 * z ** 2.83, 1.0422 + lzs * (0.13156 + 0.045 * lzs))
    x.gbp[25] = xg[67] + lzs * (xg[68] + lzs * (xg[69] + lzs * (xg[70] + lzs * (xg[71] + lzs * xg[72]))))
    x.gbp[26] = xg[73] + lzs * (xg[74] + lzs * (xg[75] + lzs * (xg[76] + lzs * xg[77])))
    x.gbp[27] = xg[78] + lzs * (xg[79] + lzs * (xg[80] + lzs * (xg[81] + lzs * (xg[82] + lzs * xg[83]))))
    x.gbp[28] = xg[84] + lzs * (xg[85] + lzs * (xg[86] + lzs * (xg[87] + lzs * xg[88])))
    x.gbp[29] = xg[89] + lzs * (xg[90] + lzs * (xg[91] + lzs * (xg[92] + lzs * (xg[93] + lzs * xg[94]))))
    x.gbp[30] = xg[95] + lzs * (xg[96] + lzs * (xg[97] + lzs * (xg[98] + lzs * (xg[99] + lzs * xg[100]))))
    x.gbp[31] = x.gbp[29] + x.gbp[30] * (x.zpars[2] - 0.2)
    x.gbp[32] = min(x.gbp[25] / x.zpars[2] ** x.gbp[26], x.gbp[27] / x.zpars[2] ** x.gbp[28])

    # Mchei
    x.gbp[33] = xg[101] ** 4
    x.gbp[34] = xg[102] * 4

    # Mcagb
    x.gbp[35] = xg[103] + lzs * (xg[104] + lzs * (xg[105] + lzs * xg[106]))
    x.gbp[36] = xg[107] + lzs * (xg[108] + lzs * (xg[109] + lzs * xg[110]))
    x.gbp[37] = xg[111] + lzs * xg[112]
    x.gbp[35] = x.gbp[35] ** 4
    x.gbp[36] = x.gbp[36] * 4
    x.gbp[37] = x.gbp[37] ** 4

    # Lhei
    # set x.gbp[41] = -1 until it is reset later with an initial call to Lheif using mass = zpars(2) and mhefl = 0.0
    x.gbp[38] = xh[1] + lzs * xh[2]
    x.gbp[39] = xh[3] + lzs * xh[4]
    x.gbp[40] = xh[5]
    x.gbp[41] = -1
    x.gbp[42] = xh[6] + lzs * (xh[7] + lzs * xh[8])
    x.gbp[43] = xh[9] + lzs * (xh[10] + lzs * xh[11])
    x.gbp[44] = xh[12] + lzs * (xh[13] + lzs * xh[14])
    x.gbp[42] = x.gbp[42] ** 2
    x.gbp[44] = x.gbp[44] ** 2

    # Lhe
    x.gbp[45] = xh[15] + lzs * (xh[16] + lzs * xh[17])
    if lzs > -1:
        x.gbp[46] = 1 - xh[19] * (lzs + 1) ** xh[18]
    else:
        x.gbp[46] = 1
    x.gbp[47] = xh[20] + lzs * (xh[21] + lzs * xh[22])
    x.gbp[48] = xh[23] + lzs * (xh[24] + lzs * xh[25])
    x.gbp[45] = x.gbp[45] ** x.gbp[48]
    x.gbp[47] = x.gbp[47] ** x.gbp[48]
    x.gbp[46] = x.gbp[46] / x.zpars[3] ** 0.1 + (x.gbp[46] * x.gbp[47] - x.gbp[45]) / x.zpars[3] ** (x.gbp[48] + 0.1)

    # Rmin
    x.gbp[49] = xh[26] + lzs * (xh[27] + lzs * (xh[28] + lzs * xh[29]))
    x.gbp[50] = xh[30] + lzs * (xh[31] + lzs * (xh[32] + lzs * xh[33]))
    x.gbp[51] = xh[34] + lzs * (xh[35] + lzs * (xh[36] + lzs * xh[37]))
    x.gbp[52] = 5 + xh[38] * z ** xh[39]
    x.gbp[53] = xh[40] + lzs * (xh[41] + lzs * (xh[42] + lzs * xh[43]))
    x.gbp[49] = x.gbp[49] ** x.gbp[53]
    x.gbp[51] = x.gbp[51] ** (2 * x.gbp[53])

    # The
    # set x.gbp[57] = -1 until it is reset later with an initial
    # call to Thef using mass = zpars(2), mc = 0.0  and mhefl = 0.0
    x.gbp[54] = xh[44] + lzs * (xh[45] + lzs * (xh[46] + lzs * xh[47]))
    x.gbp[55] = xh[48] + lzs * (xh[49] + lzs * xh[50])
    x.gbp[55] = max(x.gbp[55], 1)
    x.gbp[56] = xh[51]
    x.gbp[57] = -1
    x.gbp[58] = xh[52] + lzs * (xh[53] + lzs * (xh[54] + lzs * xh[55]))
    x.gbp[59] = xh[56] + lzs * (xh[57] + lzs * (xh[58] + lzs * xh[59]))
    x.gbp[60] = xh[60] + lzs * (xh[61] + lzs * (xh[62] + lzs * xh[63]))
    x.gbp[61] = xh[64] + lzs * xh[65]
    x.gbp[58] = x.gbp[58] ** x.gbp[61]
    x.gbp[60] = x.gbp[60] ** 5

    # Tbl
    dum1 = x.zpars[2] / x.zpars[3]
    x.gbp[62] = xh[66] + lzs * xh[67]
    x.gbp[62] = -x.gbp[62] * np.log10(dum1)
    x.gbp[63] = xh[68]
    if lzd > 0:
        x.gbp[64] = 1 - lzd * (xh[69] + lzd * (xh[70] + lzd * xh[71]))
    else:
        x.gbp[64] = 1
    x.gbp[65] = 1 - x.gbp[64] * dum1 ** x.gbp[63]
    x.gbp[66] = 1 - lzd * (xh[77] + lzd * (xh[78] + lzd * xh[79]))
    x.gbp[67] = xh[72] + lzs * (xh[73] + lzs * (xh[74] + lzs * xh[75]))
    x.gbp[68] = xh[76]

    # Lzahb
    x.gbp[69] = xh[80] + lzs * (xh[81] + lzs * xh[82])
    x.gbp[70] = xh[83] + lzs * (xh[84] + lzs * xh[85])
    x.gbp[71] = 15
    x.gbp[72] = xh[86]
    x.gbp[73] = xh[87]

    # Rzahb
    x.gbp[75] = xh[88] + lzs * (xh[89] + lzs * (xh[90] + lzs * xh[91]))
    x.gbp[76] = xh[92] + lzs * (xh[93] + lzs * (xh[94] + lzs * xh[95]))
    x.gbp[77] = xh[96] + lzs * (xh[97] + lzs * (xh[98] + lzs * xh[99]))

    # finish Lbagb
    mhefl = 0
    lx = lbagbf(x.zpars[2], mhefl, x)
    x.gbp[16] = lx

    # finish LHeI
    dum1 = 0
    lhefl = lHeIf(x.zpars[2], mhefl, x)
    x.gbp[41] = (x.gbp[38] * x.zpars[2] ** x.gbp[39] - lhefl) / (np.exp(x.zpars[2] * x.gbp[40]) * lhefl)

    # finish THe
    thefl = tHef(x.zpars[2], dum1, mhefl, x) * (x.zpars[2])
    x.gbp[57] = (thefl - x.gbp[54]) / (x.gbp[54] * np.exp(x.gbp[56] * x.zpars[2]))

    # finish Tblf
    rb = ragbf(x.zpars[3], lHeIf(x.zpars[3], x.zpars[2], x), mhefl, x)
    rr = 1 - rminf(x.zpars[3], x) / rb
    rr = max(rr, 1.0e-12)
    x.gbp[66] = x.gbp[66] / (x.zpars[3] ** x.gbp[67] * rr ** x.gbp[68])

    # finish Lzahb
    x.gbp[74] = lhefl * lHef(x.zpars[2], x)

    kw = 1
    (tm, tn, tscls, lums, GB) = star(kw, x.zpars[2], x.zpars[2], x)

    # 这里zpars[9]和zpars[10]分别算的是质量为M_HeF的恒星在BGB和HeI时的核质量
    x.zpars[9] = lum_to_mc_gb(lums[3], GB, lums[6])
    x.zpars[10] = lum_to_mc_gb(lums[4], GB, lums[6])
    # set the hydrogen and helium abundances
    x.zpars[11] = 0.76 - 3 * z
    x.zpars[12] = 0.24 + 2 * z
    # set constant for low-mass CHeB stars
    x.zpars[13] = rminf(x.zpars[2], x) / rgbf(x.zpars[2], lzahbf(x.zpars[2], x.zpars[9], x.zpars[2], x), x)
    #
    x.zpars[14] = z ** 0.4



