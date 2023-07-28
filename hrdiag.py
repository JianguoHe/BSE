import numpy as np
from utils import conditional_njit
from star import star
from mrenv import mrenv
from supernova import SN_remnant
from const import wdflag, mxns, mch, Rsun, tiny, M_ECSN
from star import star
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


@conditional_njit()
def hrdiag(self, kw, aj, mass, mt, lum, r, mc, rc, k2, tm, tn, tscls, lums, GB, kick, zcnsts):
    # 设置常数
    mlp = 12.0
    ahe = 4            # 产能效率
    aco = 16

    if self.type <= 6:
        tbagb = self.tscls[2] + self.tscls[3]
        rzams = self.rzamsf()
        rtms = self.rtmsf()

        # 主序和赫氏空隙两个阶段
        if self.aj < self.tscls[1]:
            rg = self.rgbf(self.mass, lums[3])
            # 主序阶段(通常认为这个阶段核质量为 0)
            if self.aj < self.tm:
                mc = 0.0
                tau = aj / tm
                thook = self.thook_div_tBGB() * self.tscls[1]
                epsilon = 0.01
                tau1 = min(1.0, aj / thook)
                tau2 = max(0.0, min(1.0, (aj - (1.0 - epsilon) * thook) / (epsilon * thook)))

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
                mc_old = mc
                if self.mass0 <= self.zpars[2]:
                    mcEHG = self.lum_to_mc_gb(self.lums[3])
                elif mass <= self.zpars[3]:
                    mcEHG = self.mcheif(self.mass0, self.zpars[2], self.zpars[9])
                else:
                    mcEHG = self.mcheif(self.mass0, self.zpars[2], self.zpars[10])
                rho = self.mctmsf()
                tau = (self.aj - self.tm) / (self.tscls[1] - self.tm)
                mc_new = ((1.0 - tau) * rho + tau) * mcEHG
                self.mass_core = max(self.mass_core, mc_new)
                # 检验核质量是否达到当前的总质量(如果达到，则说明包层已被剥离，氦核根据是否简并分别演化为氦主序或氦白矮星)
                if self.mass_core >= self.mass:
                    # 非简并氦核, 则演变为零龄 HeMS
                    if self.mass > self.zpars[2]:
                        self.type = 7
                        self.age = 0.0
                        self.mass0 = self.mass
                        star(self)
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
                        rmin = self.rminf(mass)
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
            self.L = self.lgbtf(GB[1])
            self.R = self.rgbf(self.mass, self.L)
            rg = self.R
            # 计算核质量(对于核是否简并有不同的核质量公式)
            # 核简并时，核的质量在GB上持续增加
            if self.mass0 <= self.zpars[2]:
                self.mass_core = self.lum_to_mc_gb(self.L)
            # 非简并核的质量在GB阶段只会轻微的增加
            else:
                tau = (aj - tscls[1])/(tscls[2] - tscls[1])
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
                    star(self)
                # 简并氦核, 则演变为零龄 HeWD
                else:
                    self.type = 10
                    self.age = 0.0
                    self.mass0 = self.mass

        # 水平分支
        elif self.age < tbagb:
            if self.type == 3 and self.mass0 <= self.zpars[2]:
                self.mass0 = self.mass    # 这里为什么改变初始质量？不懂！
                star(self)
                self.age = self.tscls[2]

            # 计算核质量
            if self.mass0 <= self.zpars[2]:
                mchei = self.lum_to_mc_gb(lums[4])
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
                    rg = r

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
                star(self)
                self.age = tau * self.tm
            else:
                self.type = 4

        # 渐近巨星分支
        else:
            envelop_lost = False
            SN_explosion = False
            # 以下的 mc_CO 表示CO核的质量, 部分情况也表示ONe核的质量
            mcbagb = mcagbf(mass, zcnsts)                                               # BAGB时的核质量(He + CO)
            mc_CO_bagb = mcgbtf(tbagb, GB[8], GB, tscls[7], tscls[8], tscls[9])         # BAGB时的CO核质量
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
            if aj < tscls[13]:
                self.type = 5
                mc = mcbagb
                mc_CO = mcgbtf(aj, GB[8], GB, tscls[7], tscls[8], tscls[9])
                # 相应光度根据 L-mc_CO 关系变化
                lum = mc_to_lum_gb(mc_CO, GB)
                # 如果当前核质量大于恒星总质量, 说明包层已经损失, 但由于氦核没有全部燃烧完, 因此成为post-HeMS 裸氦星
                if mc >= mt:
                    self.type = 9
                    mass = mc
                    mt = mc
                    (tm, tn, tscls, lums, GB) = star(self.type, mass, mt, zcnsts)
                    if mc_CO <= GB[7]:
                        aj = tscls[4] - (1.0 / ((GB[5] - 1.0) * GB[8] * GB[4])) * (mc_CO ** (1.0 - GB[5]))
                    else:
                        aj = tscls[5] - (1.0 / ((GB[6] - 1.0) * GB[8] * GB[3])) * (mc_CO ** (1.0 - GB[6]))
                    aj = max(aj, tm)
                    envelop_lost = True

            # TPAGB 阶段, Mc = Mc_CO, 如果能达到 Mcmax, 则根据此时的 Mc 演化成不同的恒星类型
            else:
                self.type = 6
                mc_CO_1 = mcgbtf(tscls[13], GB[2], GB, tscls[10], tscls[11], tscls[12])  # TPAGB开始时的CO核质量
                mc_CO = mcgbtf(aj, GB[2], GB, tscls[10], tscls[11], tscls[12])         # TPAGB开始后没有三次挖掘时的CO核质量
                lum = mc_to_lum_gb(mc_CO, GB)
                # 由于三次挖掘(3rd Dredge-up), Mc的增长变缓
                f_lambda = min(0.9, 0.3 + 0.001 * mass ** 5)
                mc_CO = mc_CO - f_lambda * (mc_CO - mc_CO_1)
                mc = mc_CO
                # 如果当前核质量大于恒星总质量, 说明包层已经损失, 由于只剩下了CO/ONe核, 根据简并与否由不同的结局(详见处理氦星时的情况)
                if mc >= mt:
                    aj = 0
                    # 简并CO核质量未达到 mch , 只能变为CO白矮星
                    if mcbagb < 1.83:
                        self.type = 11
                        mt = mc
                    # 半简并的CO核(非中心)点燃形成简并的ONe核, 简并ONe核质量未能达到电子俘获超新星临界质量 Mecs, 只能成为ONe白矮星
                    elif mcbagb < 2.25:
                        self.type = 12
                        mt = mc
                    # 非简并的CO核发生超新星爆炸(这种大质量的恒星一般在进入TPAGB之前就发生了SN, 所以下面这个分支大概率用不到)
                    else:
                        (self.type, mt) = SN_remnant(mt, mc, mass, kick)

                    # 改变恒星类型之后, 让新恒星的初始质量等于当前质量
                    mass = mt
                    envelop_lost = True

            # 检验CO/ONe核质量是否超过总质量或超新星爆炸极限质量
            if not envelop_lost and mc_CO >= mcmax:
                SN_explosion = True
                aj = 0.0
                # 简并CO核质量达到 mch 后, 星体坍缩引发Ia超新星爆炸后不会留下恒星遗迹
                if mcbagb < 1.83:
                    self.type = 15
                    mt = 0.0
                    mc = 0.0
                    lum = 1.0e-10
                    r = 1.0e-10
                # 半简并的碳氧核(非中心)点燃形成简并的氧氖核, 核质量达到 M_ECSN 后经电子俘获型超新星爆发, 留下中子星
                elif mcbagb < 2.25:
                    self.type = 13
                    mt = 1.3
                # 非简并的CO核在中心点燃, 最终重元素燃烧生成铁核, 经历铁核坍缩后发生超新星爆炸, 留下中子星或黑洞
                else:
                    (self.type, mt) = SN_remnant(mt, mcmax, mcbagb, kick)

            # 计算半径
            if not envelop_lost and not SN_explosion:
                r = ragbf(mt, lum, self.zpars[2], zcnsts)
                rg = r

    # Naked Helium Star
    if 7 <= self.type <= 9:
        # print(self.type, mass, mt, mc)
        lzams = lzhef(mass)
        rzams = self.rzhef(mt)
        # Main Sequence
        if aj < tm:
            self.type = 7
            tau = aj / tm
            lum = lzams * (1.0 + 0.45 * tau + max(0.0, 0.85 - 0.08 * mass) * tau ** 2)
            r = rzams * (1.0 + max(0.0, 0.4 - 0.22 * np.log10(mt)) * (tau - tau ** 6))
            rg = rzams    # 这个变量可能之后包层演化程序中会用到
            # Star has no core mass and hence no memory of its past which is
            # why we subject mass and mt to mass loss for this phase.
            mc = 0.0
            if mt < self.zpars[10]:
                self.type = 10
        # Helium Shell Burning
        else:
            self.type = 8
            lum = self.lgbtf(GB[8])
            r = rhehgf(mt, lum, rzams, lums[2])
            rg = rhegbf(lum)
            if r >= rg:
                self.type = 9
                r = rg
            mc = lum_to_mc_gb(lum, GB, lums[6])

            # 第一种情况, 氦星包层完全被剥离, 简并CO核/简并ONe核演变成白矮星, 非简并CO核触发超新星爆炸
            # 如果He星的质量小于0.7Msun, He包层无法全部转化为CO核, 因此对小质量He星的CO核质量上限进行限制
            mcmax_1 = min(mt, 1.45 * mt - 0.31)
            # 第二种情况, 包层还在, 但CO核的质量已经达到超新星爆炸临界值, 如果初始质量小于1.83Msun, 则为简并CO核, 最大核质量上限为Mch;
            # 如果初始质量范围是1.83-2.25Msun, 则为简并ONe核, 最大核质量上限为M_ECSN;如果初始质量>2.25Msun, 最大核质量根据初始质量决定
            mcmax_2 = mch if mass < 1.83 else M_ECSN if mass < 2.25 else 0.773 * mass - 0.35
            mcmax = min(mcmax_1, mcmax_2)

            # 简并CO核, 根据核质量变成CO白矮星或引发Ia超新星
            if mass < 1.83:
                if mc >= mcmax:
                    aj = 0
                    mc = mcmax
                    if mcmax < mcmax_2:
                        self.type = 11
                        mt = mt
                    else:
                        self.type = 15
                        mt = 0.0
                        mc = 0.0
                        lum = 1.0e-10
                        r = 1.0e-10
            # 简并的ONe核, 根据核质量变成ONe白矮星或引发ECSN留下中子星
            elif mass < 2.25:
                if mc >= mcmax:
                    aj = 0
                    mc = mcmax
                    if mcmax < mcmax_2:
                        self.type = 12
                        mt = mc
                    else:
                        self.type = 13
                        mt = 1.3
            # 非简并的CO核, 如果包层被剥离后还没达到SN爆炸临界值, 热核会冷却由非简并 → 简并, 根据热核质量确定最终结果(这里尚待商榷)
            else:
                if mc >= mcmax:
                    aj = 0
                    mc = mcmax
                    if mcmax < mcmax_2:
                        if mc < 1.08:
                            self.type = 11
                            mt = mc
                        elif mc < M_ECSN:
                            self.type = 12
                            mt = mc
                        elif mc < mch:
                            self.type = 13
                            mt = 1.3
                        else:
                            (self.type, mt) = SN_remnant(mt, mc, mass, kick)
                    else:
                        (self.type, mt) = SN_remnant(mt, mc, mass, kick)

    # White dwarf
    if 10 <= self.type <= 12:
        mc = mt
        SN_explosion = False
        if self.type == 12:
            if mc >= M_ECSN:
                self.type = 13
                aj = 0.0
                mt = 1.3
                SN_explosion = True
        else:
            if mc >= mch:
                self.type = 15
                aj = 0.0
                mt = 0.0
                lum = 1e-10
                r = 1e-10
                SN_explosion = True

        if not SN_explosion:
            xx = ahe if self.type == 10 else aco
            # modified-Mestel cooling  (未使用)
            if wdflag:
                if aj < 9000.0:
                    lum = 300.0 * mt * self.zpars[14] / (xx * (aj + 0.1)) ** 1.18
                else:
                    fac = (9000.1 * xx) ** 5.3
                    lum = 300.0 * fac * mt * self.zpars[14] / (xx * (aj + 0.10)) ** 6.48
            # Mestel cooling
            else:
                lum = 635.0 * mt * self.zpars[14] / (xx * (aj + 0.1)) ** 1.4

            r = max(1e6/Rsun, 0.0115 * np.sqrt((mch / mt) ** (2 / 3) - (mt / mch) ** (2 / 3)))
            r = min(0.1, r)
            if mt < 0.0005:
                r = 0.09
            if mt < 0.000005:
                r = 0.009

    # Neutron Star
    if self.type == 13:
        mc = mt
        # AIC黑洞
        if mc > mxns:
            self.type = 14
            aj = 0.0
        else:
            lum = 0.02 * mt ** (2 / 3) / (max(aj, 0.1)) ** 2
            r = 1.4e-5

    # Black hole
    if self.type == 14:
        mc = mt
        lum = 1.0e-10
        r = 4.24e-6 * mt

    # 计算核半径、核光度以及最后形成的致密星的半径
    tau = 0.0
    # 主序阶段
    if self.type <= 1 or self.type == 7:
        rc = 0.0
        lc = 0.0
    # 赫氏空隙/巨星阶段
    elif 2 <= self.type <= 3:
        # 非简并的氦核
        if mass > self.zpars[2]:
            lc = lzhef(mc)
            rc = self.rzhef(mc)
        # 简并氦核
        else:
            if wdflag:
                lc = 300.0 * mc * self.zpars[14] / ((ahe * 0.1) ** 1.18)
            else:
                lc = 635.0 * mc * self.zpars[14] / ((ahe * 0.1) ** 1.4)
            rc = 0.0115 * np.sqrt(max(1.48204e-6, (mch / mc) ** (2.0 / 3.0) - (mc / mch) ** (2.0 / 3.0)))
            rc = 5.0 * rc
    # 水平分支
    elif self.type == 4:
        tau = (aj - tscls[2]) / tscls[3]
        # 先把此时的核当作是一个氦主序, 计算核的半径和光度
        kw_temp = 7
        (tm, tn, tscls, lums, GB) = star(kw_temp, mc, mc, zcnsts)
        lc = lums[1] * (1.0 + 0.45 * tau + max(0.0, 0.85 - 0.08 * mc) * tau ** 2)
        rc = self.rzhef(mc) * (1.0 + max(0.0, 0.4 - 0.22 * np.log10(mc)) * (tau - tau ** 6))
        # 恢复恒星本身类型对应的特征光度/时标
        (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
    # EAGB 阶段
    elif self.type == 5:
        # 先把此时的核当作是一个氦巨星, 计算核的半径和光度
        kw_temp = 9
        tbagb = tscls[2] + tscls[3]
        if tn > tbagb:
            tau = 3.0 * (aj - tbagb) / (tn - tbagb)
        (tm, tn, tscls, lums, GB) = star(kw_temp, mc, mc, zcnsts)
        lc = mc_to_lum_gb(mc_CO, GB)
        if tau < 1.0:
            lc = lums[2] * (lc / lums[2]) ** tau
        rc = self.rzhef(mc)
        rc = min(rhehgf(mc, lc, rc, lums[2]), rhegbf(lc))
        # 恢复恒星本身类型对应的特征光度/时标
        (tm, tn, tscls, lums, GB) = star(self.type, mass, mt, zcnsts)
    # TPAGB/HeHG/HeGB
    elif self.type == 6 or 8 <= self.type <= 9:
        if wdflag:
            lc = 300.0 * mc * self.zpars[14] / ((aco * 0.10) ** 1.18)
        else:
            lc = 635.0 * mc * self.zpars[14] / ((aco * 0.10) ** 1.4)
        rc = 0.01150 * np.sqrt(max(1.48204e-6, (mch / mc) ** (2.0 / 3.0) - (mc / mch) ** (2.0 / 3.0)))
        rc = 5.0 * rc
    # 致密星
    else:
        lc = 0
        rc = r

    # Perturb the luminosity and radius due to small envelope mass (except for MS star).
    if 2 <= self.type <= 9 and self.type != 7:
        kap = -0.5
        lum0 = 7e4
        mu = ((mt - mc) / mt) * min(5.0, max(1.2, (lum / lum0) ** kap))
        if self.type >= 8:
            mcmax = min(mt, 1.45 * mt - 0.31)
            mu = ((mcmax - mc) / mcmax) * 5.0
        if mu < 1.0:
            lpert = lpert1f(mt, mu)
            lum = lc * (lum / lc) ** lpert
            if r <= rc:
                rpert = 0.0
            else:
                rpert = rpert1f(mt, mu, r, rc)
            r = rc * (r / rc) ** rpert
        rc = min(rc, r)

    # Calculate mass and radius of convective envelope, and envelope gyration radius.
    if self.type <= 9:
        rtms = self.rtmsf()   # 【疑问】这里的rtms公式是否对氦星适用
        rzams = rzamsf(mass, zcnsts) if self.type <= 6 else self.rzhef(mass)
        k2 = mrenv(kw, mass, mt, mc, lum, r, rc, aj, tm, lums[2], lums[3], lums[4], rzams, rtms, rg, k2)
    else:
        menv = 1.0e-10
        renv = 1.0e-10
        k2 = 0.21

    return kw, aj, mass, mt, lum, r, mc, rc, menv, renv, k2, tm, tn, tscls, lums, GB



