from const import mch
from utils import conditional_njit

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
# 输出 tm, tn, self.tscls, lums, GB
@conditional_njit()
def star(self):
    # 输入 kw, mass, mt, zcnsts

    # 输出参数
    # tm = 0                                 # 主序时间
    # tn = 0                                 # 核燃烧时间
    # self.tscls = np.zeros((1, 21)).flatten()    # 到达不同阶段的时标
    # lums = np.zeros((1, 11)).flatten()     # 特征光度
    # GB = np.zeros((1, 11)).flatten()       # 巨星分支参数

    if self.mass0 > 100:
        raise ValueError('mass exceeded')

    if 7 <= self.type <= 9:
        # 估算 He 星的主序时间
        self.tm = self.themsf()
        self.tscls[1] = self.tm
        # He 星在零龄主序和主序末尾的光度
        self.lums[1] = self.lzhef()
        self.lums[2] = self.lums[1] * (1 + 0.45 + max(0.0, 0.85 - 0.08 * self.mass0))
        # 设置 He 星 GB 参数
        self.GB[8] = 8.0e-5
        self.GB[3] = 4.1e4
        self.GB[4] = 5.5e4 / (1 + 0.4 * self.mass0 ** 4)
        self.GB[5] = 5
        self.GB[6] = 3
        self.GB[7] = (self.GB[3] / self.GB[4]) ** (1 / (self.GB[5] - self.GB[6]))
        # Change in slope of giant L-Mc relation
        self.lums[6] = self.GB[4] * self.GB[7] ** self.GB[5]
        # 设置 He 星的 GB 时标(下面的mc1表示HeMS末尾的核质量)
        mc1 = self.lum_to_mc_gb(self.lums[2])
        self.tscls[4] = self.tm + (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * mc1 ** (1 - self.GB[5])
        self.tscls[6] = self.tscls[4] - (self.tscls[4] - self.tm) * ((self.GB[7] / mc1) ** (1 - self.GB[5]))
        self.tscls[5] = self.tscls[6] + (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * self.GB[7] ** (1 - self.GB[6])
        # 确定氦巨星 CO 核质量达到最大值的时标
        mcmax = min(self.mass, 1.45 * self.mass - 0.31)
        if mcmax <= 0:
            mcmax = self.mass
        mcmax = min(mcmax, max(mch, 0.773 * self.mass0 - 0.35))
        if mcmax <= self.GB[7]:
            self.tscls[14] = self.tscls[4] - (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * (mcmax ** (1 - self.GB[5]))
        else:
            self.tscls[14] = self.tscls[5] - (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * (mcmax ** (1 - self.GB[6]))
        self.tscls[14] = max(self.tscls[14], self.tm)
        self.tn = self.tscls[14]
        return 0

    if self.type >= 10:
        self.tm = 1e10
        self.tscls[1] = self.tm
        self.tn = 1e10
        return 0

    # 主序和 BGB 时间
    self.tscls[1] = self.tbgbf()
    self.tm = max(self.zcnsts.zpars[8], self.thook_div_tBGB()) * self.tscls[1]
    # 零龄主序和主序末尾的光度
    self.lums[1] = self.lzamsf()
    self.lums[2] = self.ltmsf()
    # 设置巨星分支参数 GB
    self.GB[1] = 10 ** max(-4.8, min(-5.7 + 0.8 * self.mass0, -4.1 + 0.14 * self.mass0))
    self.GB[2] = 1.27e-5
    self.GB[8] = 8e-5
    self.GB[3] = max(3e4, 500 + 1.75e4 * self.mass0 ** 0.6)
    if self.mass0 <= self.zcnsts.zpars[2]:
        self.GB[4] = self.zcnsts.zpars[6]
        self.GB[5] = 6
        self.GB[6] = 3
    elif self.mass0 < 2.5:
        # 这里用的是线性插值，很明显在 mass=2.5 处，self.GB[4] = 0.975 * zcnsts.zpars[6] - 0.18 * mass
        dlogD = (0.975 * self.zcnsts.zpars[6] - 0.18 * 2.5) - self.zcnsts.zpars[6]
        self.GB[4] = self.zcnsts.zpars[6] + dlogD * (self.mass0 - self.zcnsts.zpars[2]) / (2.5 - self.zcnsts.zpars[2])
        self.GB[5] = 6 - (self.mass0 - self.zcnsts.zpars[2]) / (2.5 - self.zcnsts.zpars[2])
        self.GB[6] = 3 - (self.mass0 - self.zcnsts.zpars[2]) / (2.5 - self.zcnsts.zpars[2])
    else:
        self.GB[4] = max(-1, 0.975 * self.zcnsts.zpars[6] - 0.18 * self.mass0, 0.5 * self.zcnsts.zpars[6] - 0.06 * self.mass0)
        self.GB[5] = 5
        self.GB[6] = 2
    self.GB[4] = 10 ** self.GB[4]
    self.GB[7] = (self.GB[3] / self.GB[4]) ** (1 / (self.GB[5] - self.GB[6]))
    # Change in slope of giant L-Mc relation.
    self.lums[6] = self.GB[4] * self.GB[7] ** self.GB[5]
    # 氦点燃光度
    self.lums[4] = self.lHeIf()
    self.lums[7] = self.lbagbf()
    if self.mass0 < 0.1 and self.type <= 1:
        self.tscls[2] = 1.1 * self.tscls[1]
        self.tscls[3] = 0.1 * self.tscls[1]
        self.lums[3] = self.lbgbf()
        self.tn = 1e10
        return 0

    # 中小质量恒星, 会经历FGB阶段
    if self.mass0 <= self.zcnsts.zpars[3]:
        # 巨星分支底部的光度
        self.lums[3] = self.lbgbf()
        # Set GB timescales
        self.tscls[4] = self.tscls[1] + (1 / ((self.GB[5] - 1) * self.GB[1] * self.GB[4])) * ((self.GB[4] / self.lums[3]) ** ((self.GB[5] - 1) / self.GB[5]))
        self.tscls[6] = self.tscls[4] - (self.tscls[4] - self.tscls[1]) * ((self.lums[3] / self.lums[6]) ** ((self.GB[5] - 1) / self.GB[5]))
        self.tscls[5] = self.tscls[6] + (1 / ((self.GB[6] - 1) * self.GB[1] * self.GB[3])) * ((self.GB[3] / self.lums[6]) ** ((self.GB[6] - 1) / self.GB[6]))
        # 设置氦点燃时间
        if self.lums[4] <= self.lums[6]:
            self.tscls[2] = self.tscls[4] - (1 / ((self.GB[5] - 1) * self.GB[1] * self.GB[4])) * ((self.GB[4] / self.lums[4]) ** ((self.GB[5] - 1) / self.GB[5]))
        else:
            self.tscls[2] = self.tscls[5] - (1 / ((self.GB[6] - 1) * self.GB[1] * self.GB[3])) * ((self.GB[3] / self.lums[4]) ** ((self.GB[6] - 1) / self.GB[6]))
        # 小质量恒星
        if self.mass0 <= self.zcnsts.zpars[2]:
            mc1 = self.lum_to_mc_gb(self.lums[4])
            self.lums[5] = self.lzahbf(self.mass0, mc1, self.zcnsts.zpars[2])
            self.tscls[3] = self.tHef(self.mass0, mc1, self.zcnsts.zpars[2])
        # 中等质量恒星
        else:
            self.lums[5] = self.lHef() * self.lums[4]
            self.tscls[3] = self.tHef(self.mass0, 1, self.zcnsts.zpars[2]) * self.tscls[1]
    # 大质量恒星
    else:
        # Note that for M > zpars[3] there is no GB as the star goes from HG -> CHeB -> AGB.
        # So in effect self.tscls[1] refers to the time of Helium ignition and not the BGB.
        self.tscls[2] = self.tscls[1]
        # 这里由于是大质量恒星, 因此氦燃烧时间与核质量无关，可为任意值(此处为1)
        self.tscls[3] = self.tHef(self.mass0, 1, self.zcnsts.zpars[2]) * self.tscls[1]
        # This now represents the luminosity at the end of CHeB, ie. BAGB
        self.lums[5] = self.lums[7]   # 【疑问】为什么对于大质量恒星, 氦燃烧的光度等于BAGB的光度？
        # We set lums[3] to be the luminosity at the end of the HG
        self.lums[3] = self.lums[4]

    # 设置巨星分支底部的核质量
    if self.mass0 <= self.zcnsts.zpars[2]:
        self.GB[9] = self.lum_to_mc_gb(self.lums[3])
    elif self.mass0 <= self.zcnsts.zpars[3]:
        self.GB[9] = self.mcheif(self.mass0, self.zcnsts.zpars[2], self.zcnsts.zpars[9])
    else:
        self.GB[9] = self.mcheif(self.mass0, self.zcnsts.zpars[2], self.zcnsts.zpars[10])

    # EAGB 时标参数
    tbagb = self.tscls[2] + self.tscls[3]
    self.tscls[7] = tbagb + (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * ((self.GB[4] / self.lums[7]) ** ((self.GB[5] - 1) / self.GB[5]))
    self.tscls[9] = self.tscls[7] - (self.tscls[7] - tbagb) * ((self.lums[7] / self.lums[6]) ** ((self.GB[5] - 1) / self.GB[5]))
    self.tscls[8] = self.tscls[9] + (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * ((self.GB[3] / self.lums[6]) ** ((self.GB[6] - 1) / self.GB[6]))

    # Now to find Ltp and ttp using Mc,He,tp
    mcbagb = self.mcagbf(self.mass0)
    mc1 = mcbagb
    # The star undergoes dredge-up at Ltp causing a decrease in Mc,He
    if 0.8 <= mc1 < 2.25:
        mc1 = 0.44 * mc1 + 0.448
    self.lums[8] = self.mc_to_lum_gb(mc1, self.GB)
    if mc1 <= self.GB[7]:
        self.tscls[13] = self.tscls[7] - (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * (mc1 ** (1 - self.GB[5]))
    else:
        self.tscls[13] = self.tscls[8] - (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * (mc1 ** (1 - self.GB[6]))

    # TPAGB 时标参数
    if mc1 <= self.GB[7]:
        self.tscls[10] = self.tscls[13] + (1 / ((self.GB[5] - 1) * self.GB[2] * self.GB[4])) * ((self.GB[4] / self.lums[8]) ** ((self.GB[5] - 1) / self.GB[5]))
        self.tscls[12] = self.tscls[10] - (self.tscls[10] - self.tscls[13]) * ((self.lums[8] / self.lums[6]) ** ((self.GB[5] - 1) / self.GB[5]))
        self.tscls[11] = self.tscls[12] + (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * ((self.GB[3] / self.lums[6]) ** ((self.GB[6] - 1) / self.GB[6]))
    else:
        self.tscls[10] = self.tscls[7]
        self.tscls[12] = self.tscls[9]
        self.tscls[11] = self.tscls[13] + (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * ((self.GB[3] / self.lums[8]) ** ((self.GB[6] - 1) / self.GB[6]))

    # Get an idea of when Mc,C = Mc,C,max on the AGB
    tau = self.tscls[2] + self.tscls[3]
    mc2 = self.mcgbtf(tau, self.GB[8], self.GB, self.tscls[7], self.tscls[8], self.tscls[9])
    mcmax = max(max(mch, 0.773 * mcbagb - 0.35), 1.05 * mc2)
    if mcmax <= mc1:
        if mcmax <= self.GB[7]:
            self.tscls[14] = self.tscls[7] - (1 / ((self.GB[5] - 1) * self.GB[8] * self.GB[4])) * (mcmax ** (1 - self.GB[5]))
        else:
            self.tscls[14] = self.tscls[8] - (1 / ((self.GB[6] - 1) * self.GB[8] * self.GB[3])) * (mcmax ** (1 - self.GB[6]))
    # Star is on SAGB and we need to increase mcmax if any 3rd dredge-up has occurred.
    else:
        Lambda = min(0.9, 0.3 + 0.001 * self.mass0 ** 5)  # 这里的 Lambda 仅为局部变量
        mcmax = (mcmax - Lambda * mc1) / (1 - Lambda)
        if mcmax <= self.GB[7]:
            self.tscls[14] = self.tscls[10] - (1 / ((self.GB[5] - 1) * self.GB[2] * self.GB[4])) * (mcmax ** (1 - self.GB[5]))
        else:
            self.tscls[14] = self.tscls[11] - (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * (mcmax ** (1 - self.GB[6]))
    self.tscls[14] = max(tbagb, self.tscls[14])
    if self.mass0 > 100:
        self.tn = self.tscls[2]
        return 0

    # 计算核时标: 不考虑进一步的质量损失时, 耗尽核燃料的时间。我们定义 Mc = self.mass 的时间为 Tn, 这也会用于确定所需的时间步长
    # 注意, 当某些恒星达到 Mc = self.mass 之后还会有一个氦星的演化时间, 后者也是一个核燃烧阶段, 但并不包括在 self.tn 内
    if abs(self.mass - mcbagb) < 1e-14 and self.type < 5:
        self.tn = tbagb
    # Note that the only occurence of Mc being double-valued is for stars that have a dredge-up.
    # If self.mass = Mc where Mc could be the value taken from CHeB or from the AGB we need to check the current stellar type.
    else:
        if self.mass > mcbagb or (self.mass >= mc1 and self.type > 4):
            if self.type == 6:
                Lambda = min(0.9, 0.3 + 0.001 * self.mass0 ** 5)  # 这里的 Lambda 仅为局部变量
                mc1 = (self.mass - Lambda * mc1) / (1 - Lambda)
            else:
                mc1 = self.mass
            if mc1 <= self.GB[7]:
                self.tn = self.tscls[10] - (1 / ((self.GB[5] - 1) * self.GB[2] * self.GB[4])) * (mc1 ** (1 - self.GB[5]))
            else:
                self.tn = self.tscls[11] - (1 / ((self.GB[6] - 1) * self.GB[2] * self.GB[3])) * (mc1 ** (1 - self.GB[6]))
        else:
            # 大质量恒星
            if self.mass0 > self.zcnsts.zpars[3]:
                mc1 = self.mcheif(self.mass0, self.zcnsts.zpars[2], self.zcnsts.zpars[10])
                if self.mass <= mc1:
                    self.tn = self.tscls[2]
                else:
                    self.tn = self.tscls[2] + self.tscls[3] * ((self.mass - mc1) / (mcbagb - mc1))
            # 小质量恒星
            elif self.mass0 <= self.zcnsts.zpars[2]:
                mc1 = self.lum_to_mc_gb(self.lums[3])
                mc2 = self.lum_to_mc_gb(self.lums[4])
                if self.mass <= mc1:
                    self.tn = self.tscls[1]
                elif self.mass <= mc2:
                    if self.mass <= self.GB[7]:
                        self.tn = self.tscls[4] - (1 / ((self.GB[5] - 1) * self.GB[1] * self.GB[4])) * (self.mass ** (1 - self.GB[5]))
                    else:
                        self.tn = self.tscls[5] - (1 / ((self.GB[6] - 1) * self.GB[1] * self.GB[3])) * (self.mass ** (1 - self.GB[6]))
                else:
                    self.tn = self.tscls[2] + self.tscls[3] * ((self.mass - mc2) / (mcbagb - mc2))
            # 中等质量恒星
            else:
                mc1 = self.mcheif(self.mass0, self.zcnsts.zpars[2], self.zcnsts.zpars[9])
                mc2 = self.mcheif(self.mass0, self.zcnsts.zpars[2], self.zcnsts.zpars[10])
                if self.mass <= mc1:
                    self.tn = self.tscls[1]
                elif self.mass <= mc2:
                    tgb = self.tscls[2] - self.tscls[1]
                    self.tn = self.tscls[1] + tgb * ((self.mass - mc1) / (mc2 - mc1))
                else:
                    self.tn = self.tscls[2] + self.tscls[3] * ((self.mass - mc2) / (mcbagb - mc2))
    self.tn = min(self.tn, self.tscls[14])
    return 0
