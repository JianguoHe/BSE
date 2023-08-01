import numpy as np
from utils import conditional_njit
from StellarCal import StellarCal
from mrenv import mrenv
from supernova import SN_remnant
from const import wdflag, mxns, mch, Rsun, tiny, M_ECSN
# from zfuncs import thook_div_tBGB, tblf, lalphaf, lbetaf, lnetaf, lpertf, lgbtf
# from zfuncs import mc_to_lum_gb, lzhef, lpert1f, rzamsf, rtmsf, ralphaf, rbetaf, rgammaf
# from zfuncs import rpertf, rgbf, rminf, ragbf, rzahbf, rzhef, rhehgf, rhegbf
# from zfuncs import rpert1f, mctmsf, mcgbtf, lum_to_mc_gb, mcheif, mcagbf


# 用途: 确定恒星目前处于哪一个演化阶段(kw, age), 然后计算光度、半径、质量、核质量
# 通常当恒星进入一个新的阶段后, 某些变量会重置, 比如 kw, 对于特定的阶段(巨星、氦星、致密星), mass/mt 也会重置

# 恒星光度/演化时间
# kw      恒星类型（ 0 - 15 ）
# aj      当前时间（Myr）
# mass    初始质量 (zero-age stellar mass)
# mt      当前质量 (used for R)
# lums    特征光度
# r       恒星半径（太阳单位）
# mc      核质量
# rc      核半径
# menv    包层质量
# renv    包层半径
# tm      主序时间
# tn      核燃烧时间
# tscls   到达不同阶段的时标
# lums
# GB      巨星分支参数
# zpars   区分各种质量区间的参数
# te：    有效温度（suppressed)

# kw, aj, mass, mt, lum, r, mc, rc, k2, tm, tn, tscls, lums, GB, kick, zcnsts
@conditional_njit()
def StellarProp(self):
    # 设置常数
    mlp = 12.0
    ahe = 4            # 产能效率
    aco = 16

    if self.type <= 6:
        tbagb = self.tscls[2] + self.tscls[3]
        rzams = self.rzamsf()
        rtms = self.rtmsf()

        # 主序和赫氏空隙两个阶段
        if self.age < self.tscls[1]:
            rg = self.rgbf(self.mass, self.lums[3])
            # 主序阶段(通常认为这个阶段核质量为 0)
            if self.age < self.tm:
                self.mass_core = 0.0
                tau = self.age / self.tm
                thook = self.thook_div_tBGB() * self.tscls[1]
                epsilon = 0.01
                tau1 = min(1.0, self.age / thook)
                tau2 = max(0.0, min(1.0, (self.age - (1.0 - epsilon) * thook) / (epsilon * thook)))

                # 计算主序阶段光度
                delta_L = self.lpertf()
                dtau = tau1 ** 2 - tau2 ** 2
                alpha_L = self.lalphaf()
                beta_L = self.lbetaf()
                eta = self.lnetaf()
                lx = np.log10(self.lums[2] / self.lums[1])
                xx = alpha_L * tau + beta_L * tau ** eta + (lx - alpha_L - beta_L) * tau ** 2 - delta_L * dtau
                self.L = self.lums[1] * 10 ** xx

                # 计算主序阶段半径
                delta_R = self.rpertf()
                dtau = tau1 ** 3 - tau2 ** 3
                alpha_R = self.ralphaf()
                beta_R = self.rbetaf()
                gamma = self.rgammaf()
                rx = np.log10(rtms / rzams)
                xx = alpha_R * tau + beta_R * tau ** 10 + gamma * tau ** 40 + (
                        rx - alpha_R - beta_R - gamma) * tau ** 3 - delta_R * dtau
                self.R = rzams * 10.0 ** xx

                # This following is given by Chris for low mass MS stars which will be substantially degenerate.
                # We need the Hydrogen abundance X, which we calculate according to X = 0.76 - 3*Z,
                # the helium abundance Y, is calculated according to Y = 0.24 + 2*Z
                if self.mass0 < self.zpars[1] - 0.3:
                    self.type = 0
                    self.R = max(self.R, 0.0258 * ((1 + self.zpars[11]) ** (5 / 3)) * (self.mass0 ** (-1 / 3)))
                else:
                    self.type = 1

            # 赫氏空隙阶段
            else:
                # 计算核质量
                if self.mass0 <= self.zpars[2]:
                    mcEHG = self.lum_to_mc_gb(self.lums[3])
                elif self.mass0 <= self.zpars[3]:
                    mcEHG = self.mcheif(self.mass0, self.zpars[2], self.zpars[9])
                else:
                    mcEHG = self.mcheif(self.mass0, self.zpars[2], self.zpars[10])
                rho = self.mctmsf()
                tau = (self.age - self.tm) / (self.tscls[1] - self.tm)
                mc_new = ((1.0 - tau) * rho + tau) * mcEHG
                self.mass_core = max(self.mass_core, mc_new)
                # 检验核质量是否达到当前的总质量(如果达到，则说明包层已被剥离，氦核根据是否简并分别演化为氦主序或氦白矮星)
                if self.mass_core >= self.mass:
                    # 非简并氦核, 则演变为零龄 HeMS
                    if self.mass > self.zpars[2]:
                        self.type = 7
                        self.age = 0.0
                        self.mass0 = self.mass
                        StellarCal(self)
                    # 简并氦核, 则演变为零龄 HeWD
                    else:
                        self.type = 10
                        self.age = 0.0
                        self.mass0 = self.mass
                        # mc = mt
                else:
                    self.type = 2
                    # 计算赫氏空隙阶段光度
                    self.L = self.lums[2] * (self.lums[3] / self.lums[2]) ** tau

                    # 计算赫氏空隙阶段半径
                    # 中低质量的 HG 末尾在 BGB 处
                    if self.mass0 <= self.zpars[3]:
                        rx = rg
                    # 大质量的 HG 末尾在 He 点燃时(at Rmin)
                    else:
                        # 首先算一下 blue loop 阶段的最小半径
                        rmin = self.rminf(self.mass0)
                        # 然后算一下 He 点燃时的半径
                        ry = self.ragbf(self.mass, self.lums[4], self.zpars[2])
                        rx = min(rmin, ry)
                        if self.mass0 <= mlp:
                            texp = np.log(self.mass0 / mlp) / np.log(self.zpars[3] / mlp)
                            rx = rg
                            rx = rmin * (rx / rmin) ** texp
                        tau2 = self.tblf()
                        if tau2 < tiny:
                            rx = ry
                    self.R = rtms * (rx / rtms) ** tau

        # 巨星分支
        elif self.age < self.tscls[2]:
            self.type = 3
            # 计算光度和半径
            self.L = self.lgbtf(self.GB[1])
            self.R = self.rgbf(self.mass, self.L)
            rg = self.R
            # 计算核质量(对于核是否简并有不同的核质量公式)
            # 核简并时，核的质量在GB上持续增加
            if self.mass0 <= self.zpars[2]:
                self.mass_core = self.lum_to_mc_gb(self.L)
            # 非简并核的质量在GB阶段只会轻微的增加
            else:
                tau = (self.age - self.tscls[1])/(self.tscls[2] - self.tscls[1])
                mc_bgb = self.mcheif(self.mass0, self.zpars[2], self.zpars[9])
                mc_hei = self.mcheif(self.mass0, self.zpars[2], self.zpars[10])
                self.mass_core = mc_bgb + (mc_hei - mc_bgb) * tau
            # 检验核质量是否达到当前的总质量
            if self.mass_core >= self.mass:
                # 非简并氦核, 则演变为零龄 HeMS
                if self.mass0 > self.zpars[2]:
                    self.type = 7
                    self.age = 0.0
                    self.mass0 = self.mass
                    StellarCal(self)
                # 简并氦核, 则演变为零龄 HeWD
                else:
                    self.type = 10
                    self.age = 0.0
                    self.mass0 = self.mass

        # 水平分支
        elif self.age < tbagb:
            if self.type == 3 and self.mass0 <= self.zpars[2]:
                self.mass0 = self.mass    # 这里为什么改变初始质量？不懂！
                StellarCal(self)
                self.age = self.tscls[2]

            # 计算核质量
            if self.mass0 <= self.zpars[2]:
                mchei = self.lum_to_mc_gb(self.lums[4])
            else:
                mchei = self.mcheif(self.mass0, self.zpars[2], self.zpars[10])
            tau = (self.age - self.tscls[2]) / self.tscls[3]
            self.mass_core = mchei + (self.mcagbf(self.mass0) - mchei) * tau

            # 低质量恒星
            if self.mass0 <= self.zpars[2]:
                lx = self.lums[5]
                ly = self.lums[7]
                rx = self.rzahbf(self.mass, self.mass_core, self.zpars[2])
                rg = self.rgbf(self.mass, lx)
                rmin = rg * self.zpars[13] ** (self.mass0 / self.zpars[2])
                texp = min(max(0.4, rmin / rx), 2.5)
                ry = self.ragbf(self.mass, ly, self.zpars[2])
                if rmin < rx:
                    taul = (np.log(rx / rmin)) ** (1 / 3)
                else:
                    rmin = rx
                    taul = 0.0
                tauh = (np.log(ry / rmin)) ** (1 / 3)
                tau2 = taul * (tau - 1.0) + tauh * tau
                self.R = rmin * np.exp(abs(tau2) ** 3)
                rg = rg + tau * (ry - rg)
                self.L = lx * (ly / lx) ** (tau ** texp)

            # 大质量恒星, 氦点燃发生在 HG 上的最小半径 (Rmin) 处
            # CHeB consists of a blue phase (before tloop) and a RG phase (after tloop).
            elif self.mass0 > self.zpars[3]:
                tau2 = self.tblf()
                tloop = self.tscls[2] + tau2 * self.tscls[3]
                rmin = self.rminf(self.mass0)
                rg = self.rgbf(self.mass, self.lums[4])
                rx = self.ragbf(self.mass, self.lums[4], self.zpars[2])
                rmin = min(rmin, rx)
                if self.mass0 <= mlp:
                    texp = np.log(self.mass0 / mlp) / np.log(self.zpars[3] / mlp)
                    rx = rg
                    rx = rmin * (rx / rmin) ** texp
                else:
                    rx = rmin
                texp = min(max(0.4, rmin / rx), 2.5)
                self.L = self.lums[4] * (self.lums[7] / self.lums[4]) ** (tau ** texp)
                if self.age < tloop:
                    ly = self.lums[4] * (self.lums[7] / self.lums[4]) ** (tau2 ** texp)
                    ry = self.ragbf(self.mass, ly, self.zpars[2])
                    taul = 0.0
                    if abs(rmin - rx) > tiny:
                        taul = (np.log(rx / rmin)) ** (1 / 3)
                    tauh = 0.0
                    if ry > rmin:
                        tauh = (np.log(ry / rmin)) ** (1 / 3)
                    tau = (self.age - self.tscls[2]) / (tau2 * self.tscls[3])
                    tau2 = taul * (tau - 1.0) + tauh * tau
                    self.R = rmin * np.exp(abs(tau2) ** 3)
                    rg = rg + tau * (ry - rg)
                else:
                    self.R = self.ragbf(self.mass, self.L, self.zpars[2])
                    rg = self.R

            # 中等质量恒星, CHeB consists of a RG phase (before tloop) and a blue loop (after tloop).
            else:
                tau2 = 1.0 - self.tblf()
                tloop = self.tscls[2] + tau2 * self.tscls[3]
                if self.age < tloop:
                    tau = (tloop - self.age) / (tau2 * self.tscls[3])
                    self.L = self.lums[5] * (self.lums[4] / self.lums[5]) ** (tau ** 3)
                    self.R = self.rgbf(self.mass, self.L)
                    rg = self.R
                else:
                    lx = self.lums[5]
                    ly = self.lums[7]
                    rx = self.rgbf(self.mass, lx)
                    rmin = self.rminf(self.mass)
                    texp = min(max(0.4, rmin / rx), 2.5)
                    ry = self.ragbf(self.mass, ly, self.zpars[2])
                    if rmin < rx:
                        taul = (np.log(rx / rmin)) ** (1 / 3)
                    else:
                        rmin = rx
                        taul = 0.0
                    tauh = (np.log(ry / rmin)) ** (1 / 3)
                    tau = (self.age - tloop) / (self.tscls[3] - (tloop - self.tscls[2]))
                    tau2 = taul * (tau - 1.0) + tauh * tau
                    self.R = rmin * np.exp(abs(tau2) ** 3)
                    rg = rx + tau * (ry - rx)
                    self.L = lx * (ly / lx) ** (tau ** texp)

            # 检验核质量是否达到当前的总质量
            if self.mass_core >= self.mass:
                self.type = 7
                tau = (self.age - self.tscls[2]) / self.tscls[3]
                # 把氦星的初始质量 mass 近似为当前的核质量 mt, 因为后者的实际值无法计算
                self.mass0 = self.mass
                StellarCal(self)
                self.age = tau * self.tm
            else:
                self.type = 4

        # 渐近巨星分支
        else:
            envelop_lost = False
            SN_explosion = False
            # 以下的 mc_CO 表示CO核的质量, 部分情况也表示ONe核的质量
            mcbagb = self.mcagbf(self.mass0)                                               # BAGB时的核质量(He + CO)
            mc_CO_bagb = self.mcgbtf(tbagb, self.GB[8], self.GB, self.tscls[7], self.tscls[8], self.tscls[9])         # BAGB时的CO核质量
            # 根据mcbagb质量不同, 超新星爆发有不同的临界质量
            # 对于简并碳氧核, 超新星爆发的核质量极限是Mch
            if mcbagb < 1.83:
                mc_max_SN = mch
            # 对于半简并碳氧核, 在 M_CO = 1.08Msun 时会发生非中心点燃生成简并ONeMg核, 而ONe核发生ECSN爆发的质量极限是1.38Msun
            elif mcbagb < 2.25:
                mc_max_SN = M_ECSN
            # 对于非简并碳氧核, 可以一直燃烧到Fe核形成, SN爆炸的质量极限根据mcbagb确定
            else:
                mc_max_SN = 0.773 * mcbagb - 0.35
            # CO核/ONe核的质量有两个上限: SN爆炸极限质量和当前恒星总质量(后者情况, 包层被剥离, 核未达到SN极限, 只能变成CO/ONe WD)
            # CO核/ONe核的质量上限不应该受到mcbagb的限制, 因为只要有包层, H → He → CO就会一直发生, 即CO核质量持续增加
            mcmax = mc_max_SN      # 【改动】
            # mcmax = max(mc_CO_max_SN, 1.05 * mc_CO_bagb)

            # EAGB 阶段, Mc = Mc_He + Mc_CO = Mc_bagb(常数), 而Mc_CO 随时间不断增长, 直到全部的He核转为CO核, EAGB结束
            # 对于0.8 < Mc_bagb < 2.25的恒星, 会有一个second dredge-up阶段, 因此在EAGB末尾的CO核质量到不了Mc_bagb
            if self.age < self.tscls[13]:
                self.type = 5
                self.mass_core = mcbagb
                self.mass_co_core = self.mcgbtf(self.age, self.GB[8], self.GB, self.tscls[7], self.tscls[8], self.tscls[9])
                # 相应光度根据 L-mc_CO 关系变化
                self.L = self.mc_to_lum_gb(self.mass_co_core, self.GB)
                # 如果当前核质量大于恒星总质量, 说明包层已经损失, 但由于氦核没有全部燃烧完, 因此成为post-HeMS 裸氦星
                if self.mass_core >= self.mass:
                    self.type = 9
                    self.mass0 = self.mass_core
                    self.mass = self.mass_core
                    StellarCal(self)
                    if self.mass_co_core <= self.GB[7]:
                        self.age = self.tscls[4] - (1.0 / ((self.GB[5] - 1.0) * self.GB[8] * self.GB[4])) * (self.mass_co_core ** (1.0 - self.GB[5]))
                    else:
                        self.age = self.tscls[5] - (1.0 / ((self.GB[6] - 1.0) * self.GB[8] * self.GB[3])) * (self.mass_co_core ** (1.0 - self.GB[6]))
                    self.age = max(self.age, self.tm)
                    envelop_lost = True

            # TPAGB 阶段, Mc = Mc_CO, 如果能达到 Mcmax, 则根据此时的 Mc 演化成不同的恒星类型
            else:
                self.type = 6
                # TPAGB开始时的CO核质量
                mc_CO_1 = self.mcgbtf(self.tscls[13], self.GB[2], self.GB, self.tscls[10], self.tscls[11], self.tscls[12])
                # TPAGB开始后没有三次挖掘时的CO核质量
                self.mass_co_core = self.mcgbtf(self.age, self.GB[2], self.GB, self.tscls[10], self.tscls[11], self.tscls[12])
                lum = self.mc_to_lum_gb(self.mass_co_core, self.GB)
                # 由于三次挖掘(3rd Dredge-up), Mc的增长变缓
                f_lambda = min(0.9, 0.3 + 0.001 * self.mass0 ** 5)
                self.mass_co_core = self.mass_co_core - f_lambda * (self.mass_co_core - mc_CO_1)
                self.mass_core = self.mass_co_core
                # 如果当前核质量大于恒星总质量, 说明包层已经损失, 由于只剩下了CO/ONe核, 根据简并与否由不同的结局(详见处理氦星时的情况)
                if self.mass_core >= self.mass:
                    self.age = 0
                    # 简并CO核质量未达到 mch , 只能变为CO白矮星
                    if mcbagb < 1.83:
                        self.type = 11
                        self.mass = self.mass_core
                    # 半简并的CO核(非中心)点燃形成简并的ONe核, 简并ONe核质量未能达到电子俘获超新星临界质量 Mecs, 只能成为ONe白矮星
                    elif mcbagb < 2.25:
                        self.type = 12
                        self.mass = self.mass_core
                    # 非简并的CO核发生超新星爆炸(这种大质量的恒星一般在进入TPAGB之前就发生了SN, 所以下面这个分支大概率用不到)
                    else:
                        SN_remnant(self, mcbagb)

                    # 改变恒星类型之后, 让新恒星的初始质量等于当前质量
                    self.mass0 = self.mass
                    envelop_lost = True

            # 检验CO/ONe核质量是否超过超新星爆炸极限质量
            if not envelop_lost and self.mass_co_core >= mcmax:
                self.mass_core = mcmax
                SN_explosion = True
                self.age = 0.0
                # 简并CO核质量达到 mch 后, 星体坍缩引发Ia超新星爆炸后不会留下恒星遗迹
                if mcbagb < 1.83:
                    self.type = 15
                    self.mass = 0.0
                    self.mass_core = 0.0
                    self.L = 1.0e-10
                    self.R = 1.0e-10
                # 半简并的碳氧核(非中心)点燃形成简并的氧氖核, 核质量达到 M_ECSN 后经电子俘获型超新星爆发, 留下中子星
                elif mcbagb < 2.25:
                    self.type = 13
                    self.mass = 1.3
                # 非简并的CO核在中心点燃, 最终重元素燃烧生成铁核, 经历铁核坍缩后发生超新星爆炸, 留下中子星或黑洞
                else:
                    SN_remnant(self, mcbagb)

            # 计算半径
            if not envelop_lost and not SN_explosion:
                self.R = self.ragbf(self.mass, self.L, self.zpars[2])
                rg = self.R

    # Naked Helium Star
    if 7 <= self.type <= 9:
        lzams = self.lzhef()
        # 这里计算半径用的是当前质量
        rzams = self.rzhef(self.mass)
        # Main Sequence
        if self.age < self.tm:
            self.type = 7
            tau = self.age / self.tm
            self.L = lzams * (1.0 + 0.45 * tau + max(0.0, 0.85 - 0.08 * self.mass0) * tau ** 2)
            self.R = rzams * (1.0 + max(0.0, 0.4 - 0.22 * np.log10(self.mass)) * (tau - tau ** 6))
            rg = rzams    # 这个变量可能之后包层演化程序中会用到
            # Star has no core mass and hence no memory of its past which is
            # why we subject mass and mt to mass loss for this phase.
            self.mass_core = 0.0
            if self.mass < self.zpars[10]:
                self.type = 10
        # Helium Shell Burning
        else:
            self.type = 8
            self.L = self.lgbtf(self.GB[8])
            self.R = self.rhehgf(self.mass, self.L, rzams, self.lums[2])
            rg = self.rhegbf(self.L)
            if self.R >= rg:
                self.type = 9
                self.R = rg
            self.mass_core = self.lum_to_mc_gb(self.L)

            # 第一种情况, 氦星包层完全被剥离, 简并CO核/简并ONe核演变成白矮星, 非简并CO核触发超新星爆炸
            # 如果He星的质量小于0.7Msun, He包层无法全部转化为CO核, 因此对小质量He星的CO核质量上限进行限制
            mcmax_1 = min(self.mass, 1.45 * self.mass - 0.31)
            # 第二种情况, 包层还在, 但CO核的质量已经达到超新星爆炸临界值, 如果初始质量小于1.83Msun, 则为简并CO核, 最大核质量上限为Mch;
            # 如果初始质量范围是1.83-2.25Msun, 则为简并ONe核, 最大核质量上限为M_ECSN;如果初始质量>2.25Msun, 最大核质量根据初始质量决定
            mcmax_2 = mch if self.mass0 < 1.83 else M_ECSN if self.mass0 < 2.25 else 0.773 * self.mass0 - 0.35
            mcmax = min(mcmax_1, mcmax_2)

            # 简并CO核, 根据核质量变成CO白矮星或引发Ia超新星
            if self.mass0 < 1.83:
                if self.mass_core >= mcmax:
                    self.age = 0
                    self.mass_core = mcmax
                    if mcmax < mcmax_2:
                        self.type = 11
                    else:
                        self.type = 15
                        self.mass = 0.0
                        self.mass_core = 0.0
                        self.L = 1.0e-10
                        self.R = 1.0e-10
            # 简并的ONe核, 根据核质量变成ONe白矮星或引发ECSN留下中子星
            elif self.mass0 < 2.25:
                if self.mass_core >= mcmax:
                    self.age = 0
                    self.mass_core = mcmax
                    if mcmax < mcmax_2:
                        self.type = 12
                        self.mass = self.mass_core
                    else:
                        self.type = 13
                        self.mass = 1.3
            # 非简并的CO核, 如果包层被剥离后还没达到SN爆炸临界值, 热核会冷却由非简并 → 简并, 根据热核质量确定最终结果(这里尚待商榷)
            else:
                if self.mass_core >= mcmax:
                    self.age = 0
                    self.mass_core = mcmax
                    if mcmax < mcmax_2:
                        if self.mass_core < 1.08:
                            self.type = 11
                            self.mass = self.mass_core
                        elif self.mass_core < M_ECSN:
                            self.type = 12
                            self.mass = self.mass_core
                        elif self.mass_core < mch:
                            self.type = 13
                            self.mass = 1.3
                        else:
                            SN_remnant(self, self.mass0)
                    else:
                        SN_remnant(self, self.mass0)

    # White dwarf
    if 10 <= self.type <= 12:
        self.mass_core = self.mass
        SN_explosion = False
        if self.type == 12:
            if self.mass_core >= M_ECSN:
                self.type = 13
                self.age = 0.0
                self.mass = 1.3
                SN_explosion = True
        else:
            if self.mass_core >= mch:
                self.type = 15
                self.age = 0.0
                self.mass = 0.0
                self.L = 1e-10
                self.R = 1e-10
                SN_explosion = True

        if not SN_explosion:
            xx = ahe if self.type == 10 else aco
            # modified-Mestel cooling  (未使用)
            # if wdflag:
            #     if aj < 9000.0:
            #         lum = 300.0 * mt * self.zpars[14] / (xx * (aj + 0.1)) ** 1.18
            #     else:
            #         fac = (9000.1 * xx) ** 5.3
            #         lum = 300.0 * fac * mt * self.zpars[14] / (xx * (aj + 0.10)) ** 6.48
            # Mestel cooling
            self.L = 635.0 * self.mass * self.zpars[14] / (xx * (self.age + 0.1)) ** 1.4

            self.R = max(1e6/Rsun, 0.0115 * np.sqrt((mch / self.mass) ** (2 / 3) - (self.mass / mch) ** (2 / 3)))
            self.R = min(0.1, self.R)
            if self.mass < 0.0005:
                self.R = 0.09
            if self.mass < 0.000005:
                self.R = 0.009

    # Neutron Star
    if self.type == 13:
        self.mass_core = self.mass
        # AIC黑洞
        if self.mass_core > mxns:
            self.type = 14
            self.age = 0.0
        else:
            self.L = 0.02 * self.mass ** (2 / 3) / (max(self.age, 0.1)) ** 2
            self.R = 1.4e-5

    # Black hole
    if self.type == 14:
        self.mass_core = self.mass
        self.L = 1.0e-10
        self.R = 4.24e-6 * self.mass

    # 计算核半径、核光度
    # 主序阶段
    if self.type <= 1 or self.type == 7:
        self.radius_core = 0.0
        lc = 0.0
    # 赫氏空隙/巨星阶段
    elif 2 <= self.type <= 3:
        # 非简并的氦核
        if self.mass0 > self.zpars[2]:
            self.radius_core = self.rzhef(self.mass_core)
            lc = self.lzhef(self.mass_core)
        # 简并氦核
        else:
            self.radius_core = 5 * 0.0115 * np.sqrt(
                max(1.48204e-6, (mch / self.mass_core) ** (2 / 3) - (self.mass_core / mch) ** (2 / 3)))
            if wdflag:
                lc = 300.0 * self.mass_core * self.zpars[14] / ((ahe * 0.1) ** 1.18)
            else:
                lc = 635.0 * self.mass_core * self.zpars[14] / ((ahe * 0.1) ** 1.4)
    # 水平分支
    elif self.type == 4:
        tau = (self.age - self.tscls[2]) / self.tscls[3]
        self.radius_core = self.rzhef(self.mass_core) * (
                    1.0 + max(0.0, 0.4 - 0.22 * np.log10(self.mass_core)) * (tau - tau ** 6))
        lc = self.lzhef(self.mass_core) * (1.0 + 0.45 * tau + max(0.0, 0.85 - 0.08 * self.mass_core) * tau ** 2)
    # EAGB 阶段
    elif self.type == 5:
        tbagb = self.tscls[2] + self.tscls[3]
        tau = 3.0 * (self.age - tbagb) / (self.tn - tbagb) if self.tn > tbagb else 0
        # 保存之前的属性
        type_temp, mass0_temp, mass_temp = self.type, self.mass0, self.mass

        # 把此时的核当作是一个氦巨星, 计算核的半径和光度
        self.type, self.mass0, self.mass = 9, self.mass_core, self.mass_core
        StellarCal(self)
        lc = self.mc_to_lum_gb(self.mass_co_core, self.GB)
        lc = self.lums[2] * (lc / self.lums[2]) ** tau if tau < 1 else lc
        rc = self.rzhef(self.mass_core)
        self.radius_core = min(self.rhehgf(self.mass_core, lc, rc, self.lums[2]), self.rhegbf(lc))

        # 恢复恒星本身类型对应的特征光度/时标
        self.type, self.mass0, self.mass = type_temp, mass0_temp, mass_temp
        StellarCal(self)
    # TPAGB/HeHG/HeGB
    elif self.type == 6 or 8 <= self.type <= 9:
        self.radius_core = 5 * 0.0115 * np.sqrt(
            max(1.48204e-6, (mch / self.mass_core) ** (2 / 3) - (self.mass_core / mch) ** (2 / 3)))
        if wdflag:
            lc = 300 * self.mass_core * self.zpars[14] / ((aco * 0.1) ** 1.18)
        else:
            lc = 635 * self.mass_core * self.zpars[14] / ((aco * 0.1) ** 1.4)
    # 致密星
    else:
        self.radius_core = self.R
        lc = 0

    # Perturb the luminosity and radius due to small envelope mass (except for MS star).
    if 2 <= self.type <= 9 and self.type != 7:
        kap = -0.5
        lum0 = 7e4
        mu = ((self.mass - self.mass_core) / self.mass) * min(5.0, max(1.2, (self.L / lum0) ** kap))
        if self.type >= 8:
            mcmax = min(self.mass, 1.45 * self.mass - 0.31)
            mu = ((mcmax - self.mass_core) / mcmax) * 5.0
        if mu < 1.0:
            lpert = self.lpert1f(self.mass, mu)
            self.L = lc * (self.L / lc) ** lpert
            if self.R <= self.radius_core:
                rpert = 0.0
            else:
                rpert = self.rpert1f(self.mass, mu, self.R, self.radius_core)
            self.R = self.radius_core * (self.R / self.radius_core) ** rpert
        self.radius_core = min(self.radius_core, self.R)

    # Calculate mass and radius of convective envelope, and envelope gyration radius.
    if self.type <= 9:
        rzams = self.rzamsf() if self.type <= 6 else self.rzhef(self.mass0)
        rtms = self.rtmsf()  # 【疑问】这里的rtms公式是否对氦星适用
        mrenv(self, rzams, rtms, rg)
    else:
        self.mass_envelop = 1.0e-10
        self.radius_envelop = 1.0e-10
        self.k2 = 0.21

    return 0


