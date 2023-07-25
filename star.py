from zfuncs import lzamsf, lzahbf, lzhef, ltmsf, lbgbf, lHeIf, lHef, lbagbf, mc_to_lum_gb
from zfuncs import tbgbf, thook_div_tBGB, tHef, themsf, lum_to_mc_gb, mcagbf, mcheif, mcgbtf
from const import mch
from utils import conditional_njit
import numpy as np


# 用途: 推导不同演化阶段的时标、标志性光度、巨星分支参数

# Computes the characteristic luminosities at different stages (lums), and various timescales (tscls).
# Ref: P.P. Eggleton, M.J. Fitchett & C.A. Tout (1989) Ap.J. 347, 998.
# Revised 27th March 1995 by C. A. Tout and 24th October 1995 to include metallicity
# and 13th December 1996 to include naked helium stars
# Revised 5th April 1997 by J. R. Hurley to include Z=0.001 as well as Z=0.02, convective overshooting,
# MS hook and more elaborate CHeB. It now also sets the Giant Branch parameters relevant to the mass of the star.

#       ------------------------------------------------------------
#
#       [tscls] 1: BGB               2: He ignition         3: He burning      (BGB is the base of giant branch.)
#               4: Giant t(inf1)     5: Giant t(inf2)       6: Giant t(Mx)
#               7: EAGB t(inf1)      8: EAGB t(inf2)        9: EAGB  t(Mx)
#               10: TPAGB t(inf1)    11: TPAGB t(inf2)      12: TPAGB  t(Mx)
#               13: TP               14: t(Mcmax)                              (TP is thermally-pulsing AGB)
#
#       [lums]  1: ZAMS              2: End MS              3: BGB
#               4: He ignition       5: He burning          6: L(Mx)
#               7: BAGB              8: TP
#
#       [GB]    1: effective A(H)    2: A(H,He)             3: B
#               4: D                 5: p                   6: q
#               7: Mx                8: A(He)               9: Mc,BGB
#
#       ------------------------------------------------------------


@conditional_njit()
def star(kw, mass0, mt, zcnsts):
    # 输入 kw, mass0, mt, zcnsts

    # 输出参数
    tm = 0                                 # 主序时间
    tn = 0                                 # 核燃烧时间
    tscls = np.zeros((1, 21)).flatten()    # 到达不同阶段的时标
    lums = np.zeros((1, 11)).flatten()     # 特征光度
    GB = np.zeros((1, 11)).flatten()       # 巨星分支参数

    # 限定拟合公式的质量在 100Msun 以下
    mass = min(mass0, 100)

    if 7 <= kw <= 9:
        # 估算 He 星的主序时间
        tm = themsf(mass)
        tscls[1] = tm
        # He 星在零龄主序和主序末尾的光度
        lums[1] = lzhef(mass)
        lums[2] = lums[1] * (1 + 0.45 + max(0.0, 0.85 - 0.08 * mass))
        # 设置 He 星 GB 参数
        GB[8] = 8.0e-5
        GB[3] = 4.1e4
        GB[4] = 5.5e4 / (1 + 0.4 * mass ** 4)
        GB[5] = 5
        GB[6] = 3
        GB[7] = (GB[3] / GB[4]) ** (1 / (GB[5] - GB[6]))
        # Change in slope of giant L-Mc relation
        lums[6] = GB[4] * GB[7] ** GB[5]
        # 设置 He 星的 GB 时标(下面的mc1表示HeMS末尾的核质量)
        mc1 = lum_to_mc_gb(lums[2], GB, lums[6])
        tscls[4] = tm + (1 / ((GB[5] - 1) * GB[8] * GB[4])) * mc1 ** (1 - GB[5])
        tscls[6] = tscls[4] - (tscls[4] - tm) * ((GB[7] / mc1) ** (1 - GB[5]))
        tscls[5] = tscls[6] + (1 / ((GB[6] - 1) * GB[8] * GB[3])) * GB[7] ** (1 - GB[6])
        # 确定氦巨星 CO 核质量达到最大值的时标
        mcmax = min(mt, 1.45 * mt - 0.31)
        if mcmax <= 0:
            mcmax = mt
        mcmax = min(mcmax, max(mch, 0.773 * mass - 0.35))
        if mcmax <= GB[7]:
            tscls[14] = tscls[4] - (1 / ((GB[5] - 1) * GB[8] * GB[4])) * (mcmax ** (1 - GB[5]))
        else:
            tscls[14] = tscls[5] - (1 / ((GB[6] - 1) * GB[8] * GB[3])) * (mcmax ** (1 - GB[6]))
        tscls[14] = max(tscls[14], tm)
        tn = tscls[14]
        return tm, tn, tscls, lums, GB    # 结束此函数

    if kw >= 10:
        tm = 1e10
        tscls[1] = tm
        tn = 1e10
        return tm, tn, tscls, lums, GB    # 结束此函数

    # 主序和 BGB 时间
    tscls[1] = tbgbf(mass, zcnsts)
    tm = max(zcnsts.zpars[8], thook_div_tBGB(mass, zcnsts)) * tscls[1]
    # 零龄主序和主序末尾的光度
    lums[1] = lzamsf(mass, zcnsts)
    lums[2] = ltmsf(mass, zcnsts)
    # 设置巨星分支参数 GB
    GB[1] = 10 ** max(-4.8, min(-5.7 + 0.8 * mass, -4.1 + 0.14 * mass))
    GB[2] = 1.27e-5
    GB[8] = 8e-5
    GB[3] = max(3e4, 500 + 1.75e4 * mass ** 0.6)
    if mass <= zcnsts.zpars[2]:
        GB[4] = zcnsts.zpars[6]
        GB[5] = 6
        GB[6] = 3
    elif mass < 2.5:
        # 这里用的是线性插值，很明显在 mass=2.5 处，GB[4] = 0.975 * zcnsts.zpars[6] - 0.18 * mass
        dlogD = (0.975 * zcnsts.zpars[6] - 0.18 * 2.5) - zcnsts.zpars[6]
        GB[4] = zcnsts.zpars[6] + dlogD * (mass - zcnsts.zpars[2]) / (2.5 - zcnsts.zpars[2])
        GB[5] = 6 - (mass - zcnsts.zpars[2]) / (2.5 - zcnsts.zpars[2])
        GB[6] = 3 - (mass - zcnsts.zpars[2]) / (2.5 - zcnsts.zpars[2])
    else:
        GB[4] = max(-1, 0.975 * zcnsts.zpars[6] - 0.18 * mass, 0.5 * zcnsts.zpars[6] - 0.06 * mass)
        GB[5] = 5
        GB[6] = 2
    GB[4] = 10 ** GB[4]
    GB[7] = (GB[3] / GB[4]) ** (1 / (GB[5] - GB[6]))
    # Change in slope of giant L-Mc relation.
    lums[6] = GB[4] * GB[7] ** GB[5]
    # 氦点燃光度
    lums[4] = lHeIf(mass, zcnsts.zpars[2], zcnsts)
    lums[7] = lbagbf(mass, zcnsts.zpars[2], zcnsts)
    if mass < 0.1 and kw <= 1:
        tscls[2] = 1.1 * tscls[1]
        tscls[3] = 0.1 * tscls[1]
        lums[3] = lbgbf(mass, zcnsts)
        tn = 1e10
        return tm, tn, tscls, lums, GB   # 结束此函数

    # 中小质量恒星, 会经历FGB阶段
    if mass <= zcnsts.zpars[3]:
        # 巨星分支底部的光度
        lums[3] = lbgbf(mass, zcnsts)
        # Set GB timescales
        tscls[4] = tscls[1] + (1 / ((GB[5] - 1) * GB[1] * GB[4])) * ((GB[4] / lums[3]) ** ((GB[5] - 1) / GB[5]))
        tscls[6] = tscls[4] - (tscls[4] - tscls[1]) * ((lums[3] / lums[6]) ** ((GB[5] - 1) / GB[5]))
        tscls[5] = tscls[6] + (1 / ((GB[6] - 1) * GB[1] * GB[3])) * ((GB[3] / lums[6]) ** ((GB[6] - 1) / GB[6]))
        # 设置氦点燃时间
        if lums[4] <= lums[6]:
            tscls[2] = tscls[4] - (1 / ((GB[5] - 1) * GB[1] * GB[4])) * ((GB[4] / lums[4]) ** ((GB[5] - 1) / GB[5]))
        else:
            tscls[2] = tscls[5] - (1 / ((GB[6] - 1) * GB[1] * GB[3])) * ((GB[3] / lums[4]) ** ((GB[6] - 1) / GB[6]))
        # 小质量恒星
        if mass <= zcnsts.zpars[2]:
            mc1 = lum_to_mc_gb(lums[4], GB, lums[6])
            lums[5] = lzahbf(mass, mc1, zcnsts.zpars[2], zcnsts)
            tscls[3] = tHef(mass, mc1, zcnsts.zpars[2], zcnsts)
        # 中等质量恒星
        else:
            lums[5] = lHef(mass, zcnsts) * lums[4]
            tscls[3] = tHef(mass, 1, zcnsts.zpars[2], zcnsts) * tscls[1]
    # 大质量恒星
    else:
        # Note that for M > zpars[3] there is no GB as the star goes from HG -> CHeB -> AGB.
        # So in effect tscls[1] refers to the time of Helium ignition and not the BGB.
        tscls[2] = tscls[1]
        # 这里由于是大质量恒星, 因此氦燃烧时间与核质量无关，可为任意值(此处为1)
        tscls[3] = tHef(mass, 1, zcnsts.zpars[2], zcnsts) * tscls[1]
        # This now represents the luminosity at the end of CHeB, ie. BAGB
        lums[5] = lums[7]   # 【疑问】为什么对于大质量恒星, 氦燃烧的光度等于BAGB的光度？
        # We set lums[3] to be the luminosity at the end of the HG
        lums[3] = lums[4]

    # 设置巨星分支底部的核质量
    if mass <= zcnsts.zpars[2]:
        GB[9] = lum_to_mc_gb(lums[3], GB, lums[6])
    elif mass <= zcnsts.zpars[3]:
        GB[9] = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[9], zcnsts)
    else:
        GB[9] = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)

    # EAGB 时标参数
    tbagb = tscls[2] + tscls[3]
    tscls[7] = tbagb + (1 / ((GB[5] - 1) * GB[8] * GB[4])) * ((GB[4] / lums[7]) ** ((GB[5] - 1) / GB[5]))
    tscls[9] = tscls[7] - (tscls[7] - tbagb) * ((lums[7] / lums[6]) ** ((GB[5] - 1) / GB[5]))
    tscls[8] = tscls[9] + (1 / ((GB[6] - 1) * GB[8] * GB[3])) * ((GB[3] / lums[6]) ** ((GB[6] - 1) / GB[6]))

    # Now to find Ltp and ttp using Mc,He,tp
    mcbagb = mcagbf(mass, zcnsts)
    mc1 = mcbagb
    # The star undergoes dredge-up at Ltp causing a decrease in Mc,He
    if 0.8 <= mc1 < 2.25:
        mc1 = 0.44 * mc1 + 0.448
    lums[8] = mc_to_lum_gb(mc1, GB)
    if mc1 <= GB[7]:
        tscls[13] = tscls[7] - (1 / ((GB[5] - 1) * GB[8] * GB[4])) * (mc1 ** (1 - GB[5]))
    else:
        tscls[13] = tscls[8] - (1 / ((GB[6] - 1) * GB[8] * GB[3])) * (mc1 ** (1 - GB[6]))

    # TPAGB 时标参数
    if mc1 <= GB[7]:
        tscls[10] = tscls[13] + (1 / ((GB[5] - 1) * GB[2] * GB[4])) * ((GB[4] / lums[8]) ** ((GB[5] - 1) / GB[5]))
        tscls[12] = tscls[10] - (tscls[10] - tscls[13]) * ((lums[8] / lums[6]) ** ((GB[5] - 1) / GB[5]))
        tscls[11] = tscls[12] + (1 / ((GB[6] - 1) * GB[2] * GB[3])) * ((GB[3] / lums[6]) ** ((GB[6] - 1) / GB[6]))
    else:
        tscls[10] = tscls[7]
        tscls[12] = tscls[9]
        tscls[11] = tscls[13] + (1 / ((GB[6] - 1) * GB[2] * GB[3])) * ((GB[3] / lums[8]) ** ((GB[6] - 1) / GB[6]))

    # Get an idea of when Mc,C = Mc,C,max on the AGB
    tau = tscls[2] + tscls[3]
    mc2 = mcgbtf(tau, GB[8], GB, tscls[7], tscls[8], tscls[9])
    mcmax = max(max(mch, 0.773 * mcbagb - 0.35), 1.05 * mc2)
    if mcmax <= mc1:
        if mcmax <= GB[7]:
            tscls[14] = tscls[7] - (1 / ((GB[5] - 1) * GB[8] * GB[4])) * (mcmax ** (1 - GB[5]))
        else:
            tscls[14] = tscls[8] - (1 / ((GB[6] - 1) * GB[8] * GB[3])) * (mcmax ** (1 - GB[6]))
    # Star is on SAGB and we need to increase mcmax if any 3rd dredge-up has occurred.
    else:
        Lambda = min(0.9, 0.3 + 0.001 * mass ** 5)  # 这里的 Lambda 仅为局部变量
        mcmax = (mcmax - Lambda * mc1) / (1 - Lambda)
        if mcmax <= GB[7]:
            tscls[14] = tscls[10] - (1 / ((GB[5] - 1) * GB[2] * GB[4])) * (mcmax ** (1 - GB[5]))
        else:
            tscls[14] = tscls[11] - (1 / ((GB[6] - 1) * GB[2] * GB[3])) * (mcmax ** (1 - GB[6]))
    tscls[14] = max(tbagb, tscls[14])
    if mass > 100:
        tn = tscls[2]
        return tm, tn, tscls, lums, GB   # 结束此函数

    # 计算核时标: 不考虑进一步的质量损失时, 耗尽核燃料的时间。我们定义 Mc = Mt 的时间为 Tn, 这也会用于确定所需的时间步长
    # 注意, 当某些恒星达到 Mc = Mt 之后还会有一个氦星的演化时间, 后者也是一个核燃烧阶段, 但并不包括在 Tn 内
    if abs(mt - mcbagb) < 1e-14 and kw < 5:
        tn = tbagb
    # Note that the only occurence of Mc being double-valued is for stars that have a dredge-up.
    # If Mt = Mc where Mc could be the value taken from CHeB or from the AGB we need to check the current stellar type.
    else:
        if mt > mcbagb or (mt >= mc1 and kw > 4):
            if kw == 6:
                Lambda = min(0.9, 0.3 + 0.001 * mass ** 5)  # 这里的 Lambda 仅为局部变量
                mc1 = (mt - Lambda * mc1) / (1 - Lambda)
            else:
                mc1 = mt
            if mc1 <= GB[7]:
                tn = tscls[10] - (1 / ((GB[5] - 1) * GB[2] * GB[4])) * (mc1 ** (1 - GB[5]))
            else:
                tn = tscls[11] - (1 / ((GB[6] - 1) * GB[2] * GB[3])) * (mc1 ** (1 - GB[6]))
        else:
            # 大质量恒星
            if mass > zcnsts.zpars[3]:
                mc1 = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
                if mt <= mc1:
                    tn = tscls[2]
                else:
                    tn = tscls[2] + tscls[3] * ((mt - mc1) / (mcbagb - mc1))
            # 小质量恒星
            elif mass <= zcnsts.zpars[2]:
                mc1 = lum_to_mc_gb(lums[3], GB, lums[6])
                mc2 = lum_to_mc_gb(lums[4], GB, lums[6])
                if mt <= mc1:
                    tn = tscls[1]
                elif mt <= mc2:
                    if mt <= GB[7]:
                        tn = tscls[4] - (1 / ((GB[5] - 1) * GB[1] * GB[4])) * (mt ** (1 - GB[5]))
                    else:
                        tn = tscls[5] - (1 / ((GB[6] - 1) * GB[1] * GB[3])) * (mt ** (1 - GB[6]))
                else:
                    tn = tscls[2] + tscls[3] * ((mt - mc2) / (mcbagb - mc2))
            # 中等质量恒星
            else:
                mc1 = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[9], zcnsts)
                mc2 = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
                if mt <= mc1:
                    tn = tscls[1]
                elif mt <= mc2:
                    tgb = tscls[2] - tscls[1]
                    tn = tscls[1] + tgb * ((mt - mc1) / (mc2 - mc1))
                else:
                    tn = tscls[2] + tscls[3] * ((mt - mc2) / (mcbagb - mc2))
    tn = min(tn, tscls[14])
    return tm, tn, tscls, lums, GB    # 结束此函数





