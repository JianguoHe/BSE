import numpy as np
from numba import njit
from star import star
from mrenv import mrenv
from supernova import supernova
from const import SNtype, ifflag, nsflag, wdflag, mxns, mch, Rsun, tiny
from zfuncs import thook_div_tBGB, tblf, lalphaf, lbetaf, letaf, lhookf, lgbtf
from zfuncs import lmcgbf, lzhef, lpertf, rzamsf, rtmsf, ralphaf, rbetaf, rgammaf
from zfuncs import rhookf, rgbf, rminf, ragbf, rzahbf, rzhef, rhehgf, rhegbf
from zfuncs import rpertf, mcTMS_div_mcEHG, mcgbtf, mcgbf, mcheif, mcagbf


# 用途: 确定恒星目前处于哪一个演化阶段(kw, age), 然后计算当前光度、半径、核质量、恒星类型
# 通常当恒星进入一个新的阶段后, 某些变量会重置, 比如 kw, 对于特定的阶段(巨星、氦星、致密星), mass/mt 也会重置


# 恒星光度/演化时间
# mass:      初始质量 (zero-age stellar mass)
# aj:        当前时间 (Myr)
# mt:        当前质量 (used for R)
# tm:        主序时间
# tn:        核燃烧时间
# tscls:     到达不同阶段的时标
# lums:      特征光度
# GB:        巨星分支参数
# zcnsts:    区分各种质量区间的参数
# r:         恒星半径（太阳单位）
# te:        有效温度（suppressed)
# kw:        恒星类型（ 0 - 15 ）
# mc:        核质量


@njit
def hrdiag(mass, aj, mt, tm, tn, tscls, lums, GB, zcnsts, r, lum, kw, mc, rc, menv, renv, k2, kick):
    # 设置常数
    mlp = 12.0
    ahe = 4
    aco = 16
    taumin = 5e-8  # 为了防止计算主序光度/半径时浮点下溢设的最小值（公式中的幂方因子 tau**40 可能超过Python的浮点数范围）

    # Make evolutionary changes to stars that have not reached KW > 5.
    mass0 = mass
    mt0 = mt
    if mass0 > 100.0:
        mass = 100.0
    if mt0 > 100.0:
        mt = 100.0

    # 自己加的变量初始化
    mcx = 0    # 不知道这个变量是什么

    while kw <= 6:
        tbagb = tscls[2] + tscls[3]
        rzams = rzamsf(mass, zcnsts)
        rtms = rtmsf(mass, zcnsts)

        # 主序和赫氏空隙两个阶段
        if aj < tscls[1]:
            # 计算 EHG 时的半径
            rg = rgbf(mt, lums[3], zcnsts)
            # 主序阶段
            if aj < tm:
                # 通常认为主序时核与包层的轮廓不够明显, 即核质量为0
                mc = 0.0
                tau = aj / tm
                thook = thook_div_tBGB(mass, zcnsts) * tscls[1]

                # 定义湍动时标, 模拟hook的演化
                epsilon = 0.01
                tau1 = min(1.0, aj / thook)
                tau2 = max(0.0, min(1.0, (aj - (1.0 - epsilon) * thook) / (epsilon * thook)))

                # 计算主序阶段光度
                delta_L = lhookf(mass, zcnsts.zpars[1], zcnsts)
                alpha_L = lalphaf(mass, zcnsts)
                beta_L = lbetaf(mass, zcnsts)
                eta_L = letaf(mass, zcnsts)
                term1 = alpha_L * tau + (np.log10(lums[2]/lums[1]) - alpha_L) * tau ** 2
                term2 = beta_L * tau ** eta_L - beta_L * tau ** 2
                term3 = delta_L * (tau1 ** 2 - tau2 ** 2)
                if tau > taumin:
                    lum = lums[1] * 10 ** (term1 + term2 - term3)
                else:
                    lum = lums[1] * 10 ** (term1 - term3)

                # 计算主序阶段半径
                delta_R = rhookf(mass, zcnsts.zpars[1], zcnsts)
                alpha_R = ralphaf(mass, zcnsts)
                beta_R = rbetaf(mass, zcnsts)
                gamma_R = rgammaf(mass, zcnsts)
                term1 = alpha_R * tau + (np.log10(rtms / rzams) - alpha_R) * tau ** 3
                term2 = beta_R * tau ** 10 - beta_R * tau ** 3
                term3 = gamma_R * tau ** 40 - gamma_R * tau ** 3
                term4 = delta_R * (tau1 ** 3 - tau2 ** 3)
                if tau > taumin:
                    r = rzams * 10.0 ** (term1 + term2 + term3 - term4)
                else:
                    r = rzams * 10.0 ** (term1 - term4)

                # This following is given by Chris for low mass MS stars which will be substantially degenerate.
                # We need the Hydrogen abundance X, which we calculate according to X = 0.76 - 3*Z,
                # the helium abundance Y, is calculated according to Y = 0.24 + 2*Z
                # 【疑问: 没找到出处】
                if mass < zcnsts.zpars[1] - 0.3:
                    kw = 0
                    r = max(r, 0.0258 * ((1.0 + zcnsts.zpars[11]) ** (5.0 / 3.0)) * (mass ** (-1.0 / 3.0)))
                else:
                    kw = 1

            # 赫氏空隙阶段
            else:
                # 计算核质量
                mcx = mc
                if mass <= zcnsts.zpars[2]:
                    mcEHG = mcgbf(lums[3], GB, lums[6])
                elif mass <= zcnsts.zpars[3]:
                    mcEHG = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[9], zcnsts)
                else:
                    mcEHG = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
                rho = mcTMS_div_mcEHG(mass)
                tau = (aj - tm) / (tscls[1] - tm)
                mc = ((1.0 - tau) * rho + tau) * mcEHG
                mc = max(mc, mcx)
                # 检验核质量是否达到当前的总质量(如果达到，则说明包层已被剥离，氦核根据是否简并分别演化为氦主序或氦白矮星)
                if mc >= mt:
                    aj = 0.0
                    # 非简并氦核, 则演变为零龄 HeMS
                    if mass > zcnsts.zpars[2]:
                        kw = 7
                        mass = mt
                        mc = 0.0
                        (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
                    # 简并氦核, 则演变为零龄 HeWD
                    else:
                        kw = 10
                        mass = mt
                        mc = mt
                else:
                    # 计算赫氏空隙阶段光度
                    lum = lums[2] * (lums[3] / lums[2]) ** tau

                    # 计算赫氏空隙阶段半径
                    # 中低质量的 HG 末尾在 BGB 处
                    if mass <= zcnsts.zpars[3]:
                        rx = rg
                    # 大质量的 HG 末尾在 He 点燃时(at Rmin)
                    else:
                        # 首先算一下 blue loop 阶段的最小半径
                        rmin = rminf(mass, zcnsts)
                        # 然后算一下 He 点燃时的半径
                        ry = ragbf(mt, lums[4], zcnsts.zpars[2], zcnsts)
                        rx = min(rmin, ry)
                        if mass <= mlp:
                            texp = np.log(mass / mlp) / np.log(zcnsts.zpars[3] / mlp)
                            rx = rg
                            rx = rmin * (rx / rmin) ** texp
                        tau2 = tblf(mass, zcnsts.zpars[2], zcnsts.zpars[3], zcnsts)
                        if tau2 < tiny:
                            rx = ry
                    r = rtms * (rx / rtms) ** tau
                    kw = 2

        # 巨星分支
        elif aj < tscls[2]:
            kw = 3
            # 计算光度和半径
            lum = lgbtf(aj, GB[1], GB, tscls[4], tscls[5], tscls[6])
            r = rgbf(mt, lum, zcnsts)
            rg = r
            # 计算核质量(对于核是否简并有不同的核质量公式)
            # 恒星在GB阶段拥有简并核, 且核的质量不断增长
            if mass <= zcnsts.zpars[2]:
                mc = mcgbf(lum, GB, lums[6])
            # 恒星在GB阶段拥有非简并核, 核的质量轻微增长（BGB和HeI的核质量基本不怎么变）
            else:
                tau = (aj - tscls[1])/(tscls[2] - tscls[1])
                mcx = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[9], zcnsts)
                mcy = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
                mc = mcx + (mcy - mcx) * tau
            # 检验核质量是否达到当前的总质量
            if mc >= mt:
                aj = 0.0
                # 非简并氦核, 则演变为零龄 HeMS
                if mass > zcnsts.zpars[2]:
                    kw = 7
                    mass = mt
                    mc = 0.0
                    (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
                # 简并氦核, 则演变为零龄 HeWD
                else:
                    kw = 10
                    mass = mt
                    mc = mt

        # 水平分支
        elif aj < tbagb:
            if kw == 3 and mass <= zcnsts.zpars[2]:
                mass = mt
                (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
                aj = tscls[2]

            # 计算核质量
            if mass <= zcnsts.zpars[2]:
                mchei = mcgbf(lums[4], GB, lums[6])
            else:
                mchei = mcheif(mass, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
            tau = (aj - tscls[2]) / tscls[3]
            mc = mchei + (mcagbf(mass, zcnsts) - mchei) * tau

            # 低质量恒星
            if mass <= zcnsts.zpars[2]:
                lx = lums[5]
                ly = lums[7]
                rx = rzahbf(mt, mc, zcnsts.zpars[2], zcnsts)
                rg = rgbf(mt, lx, zcnsts)
                rmin = rg * zcnsts.zpars[13] ** (mass / zcnsts.zpars[2])
                texp = min(max(0.40, rmin / rx), 2.50)
                ry = ragbf(mt, ly, zcnsts.zpars[2], zcnsts)
                if rmin < rx:
                    taul = (np.log(rx/rmin))**(1.0/3.0)
                else:
                    rmin = rx
                    taul = 0.0
                tauh = (np.log(ry / rmin)) ** (1.0 / 3.0)
                tau2 = taul * (tau - 1.0) + tauh * tau
                r = rmin * np.exp(abs(tau2) ** 3)
                rg = rg + tau * (ry - rg)
                lum = lx * (ly / lx) ** (tau ** texp)

            # 大质量恒星, 氦点燃发生在 HG 上的最小半径 (Rmin) 处
            # CHeB consists of a blue phase (before tloop) and a RG phase (after tloop).
            elif mass > zcnsts.zpars[3]:
                tau2 = tblf(mass, zcnsts.zpars[2], zcnsts.zpars[3], zcnsts)
                tloop = tscls[2] + tau2 * tscls[3]
                rmin = rminf(mass, zcnsts)
                rg = rgbf(mt, lums[4], zcnsts)
                rx = ragbf(mt, lums[4], zcnsts.zpars[2], zcnsts)
                rmin = min(rmin, rx)
                if mass <= mlp:
                    texp = np.log(mass / mlp) / np.log(zcnsts.zpars[3] / mlp)
                    rx = rg
                    rx = rmin * (rx / rmin) ** texp
                else:
                    rx = rmin
                texp = min(max(0.40, rmin / rx), 2.50)
                lum = lums[4] * (lums[7] / lums[4]) ** (tau ** texp)
                if aj < tloop:
                    ly = lums[4] * (lums[7] / lums[4]) ** (tau2 ** texp)
                    ry = ragbf(mt, ly, zcnsts.zpars[2], zcnsts)
                    taul = 0.0
                    if abs(rmin - rx) > tiny:
                        taul = (np.log(rx / rmin)) ** (1.0 / 3.0)
                    tauh = 0.0
                    if ry > rmin:
                        tauh = (np.log(ry / rmin)) ** (1.0 / 3.0)
                    tau = (aj - tscls[2]) / (tau2 * tscls[3])
                    tau2 = taul * (tau - 1.0) + tauh * tau
                    r = rmin * np.exp(abs(tau2) ** 3)
                    rg = rg + tau * (ry - rg)
                else:
                    r = ragbf(mt, lum, zcnsts.zpars[2], zcnsts)
                    rg = r

            # 中等质量恒星, CHeB consists of a RG phase (before tloop) and a blue loop (after tloop).
            else:
                tau2 = 1.0 - tblf(mass, zcnsts.zpars[2], zcnsts.zpars[3], zcnsts)
                tloop = tscls[2] + tau2 * tscls[3]
                if aj < tloop:
                    tau = (tloop - aj) / (tau2 * tscls[3])
                    lum = lums[5] * (lums[4] / lums[5]) ** (tau ** 3)
                    r = rgbf(mt, lum, zcnsts)
                    rg = r
                else:
                    lx = lums[5]
                    ly = lums[7]
                    rx = rgbf(mt, lx, zcnsts)
                    rmin = rminf(mt, zcnsts)
                    texp = min(max(0.40, rmin / rx), 2.50)
                    ry = ragbf(mt, ly, zcnsts.zpars[2], zcnsts)
                    if rmin < rx:
                        taul = (np.log(rx / rmin)) ** (1.0 / 3.0)
                    else:
                        rmin = rx
                        taul = 0.0
                    tauh = (np.log(ry / rmin)) ** (1.0 / 3.0)
                    tau = (aj - tloop) / (tscls[3] - (tloop - tscls[2]))
                    tau2 = taul * (tau - 1.0) + tauh * tau
                    r = rmin * np.exp(abs(tau2) ** 3)
                    rg = rx + tau * (ry - rx)
                    lum = lx * (ly / lx) ** (tau ** texp)

            # 检验核质量是否达到当前的总质量
            if mc >= mt:
                kw = 7
                tau = (aj - tscls[2])/tscls[3]
                mass = mt   # 这里把当前的(核)质量 mt 近似处理为氦星的初始质量 mass, 因为后者的实际值无法计算
                (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
                aj = tau * tm
            else:
                kw = 4

        # 渐近巨星分支
        # 在EAGB阶段, Mc = Mc,he = Mc,bagb(常数), Mcx = Mc,CO(缓慢增长)
        # 在TPAGB阶段, Mc = Mcx = Mc,CO(缓慢增长), 如果能达到 Mcmax, 则根据此时的 Mc 演化成不同的恒星类型
        else:
            # 以下的 mcx 都表示CO核的质量
            mcbagb = mcagbf(mass, zcnsts)                                          # BAGB时的核质量(He + CO)
            mcx = mcgbtf(tbagb, GB[8], GB, tscls[7], tscls[8], tscls[9])           # BAGB时的CO核质量
            mcmax = max(mch, 0.773 * mcbagb - 0.35, 1.05 * mcx)

            # EAGB 阶段
            if aj < tscls[13]:
                # 在 EAGB 阶段把 He 核当作 mc 且不随时间变化, 而 CO 核质量随时间增长, 相应光度根据 L-mcx 关系变化
                kw = 5
                mc = mcbagb
                mcx = mcgbtf(aj, GB[8], GB, tscls[7], tscls[8], tscls[9])
                lum = lmcgbf(mcx, GB)
                # Evolved naked helium star as the envelope is lost but the star has not completed its interior burning.
                # The star becomes a post-HeMS star.
                if mc >= mt:
                    kw = 9
                    mt = mc
                    mass = mt
                    mc = mcx
                    (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
                    if mc <= GB[7]:
                        aj = tscls[4] - (1.0 / ((GB[5] - 1.0) * GB[8] * GB[4])) * (mc ** (1.0 - GB[5]))
                    else:
                        aj = tscls[5] - (1.0 / ((GB[6] - 1.0) * GB[8] * GB[3])) * (mc ** (1.0 - GB[6]))
                    aj = max(aj, tm)
                    break

            # TPAGB 阶段
            else:
                kw = 6
                mcx = mcgbtf(tscls[13], GB[2], GB, tscls[10], tscls[11], tscls[12])   # TPAGB开始时的CO核质量
                mc = mcgbtf(aj, GB[2], GB, tscls[10], tscls[11], tscls[12])           # TPAGB开始后没有三次挖掘时的CO核质量
                lum = lmcgbf(mc, GB)
                # 由于三次挖掘(3rd Dredge-up), Mc的增长变缓
                f_lambda = min(0.9, 0.3 + 0.001 * mass ** 5)
                mc = mc - f_lambda * (mc - mcx)
                mcx = mc
                mcmax = min(mt, mcmax)
            r = ragbf(mt, lum, zcnsts.zpars[2], zcnsts)
            rg = r

            # 检验CO核质量是否超过总质量或允许的最大核质量
            if mcx > mcmax:
                aj = 0.0
                mc = mcmax
                # 如果此时的(最大)核质量仍然未能达到 Mch, 则说明 mc = mcmax = mt, 包层在 TPAGB 阶段损失
                if mc < mch:
                    # ifflag(Ture) uses WD IFMR of HPE, 1995, MNRAS, 272, 800 (0).  (目前没有使用)
                    if ifflag:
                        if zcnsts.z >= 0.01:    # 有改动
                            if mass < 1.0:
                                mc = 0.46
                            else:
                                mc = max(0.54 + 0.042 * mass, min(0.36 + 0.104 * mass, 0.58 + 0.061 * mass))
                        else:
                            mc = max(0.54 + 0.073 * mass, min(0.29 + 0.178 * mass, 0.65 + 0.062 * mass))
                        mc = min(mch, mc)
                    mt = mc
                    # 简并CO核质量未达到 mch , 只能变为CO白矮星
                    if mcbagb < 1.83:
                        kw = 11
                    # 半简并的CO核(非中心)点燃形成简并的ONe核
                    else:
                        # 简并ONe核质量未能达到电子俘获超新星临界质量 Mecs, 只能成为ONe白矮星
                        if mt < 1.38:
                            kw = 12
                        # 简并ONe核质量超过 Mecs, 电子俘获超新星(ECsn)爆发, 留下中子星
                        else:
                            kw = 13
                            mt = 1.3
                    mass = mt
                # 核质量已经达到 mch, 此时CO核有简并或非简并(不可能为半简并, 因为后者会形成简并的ONe核, 质量在 1.38 时就会经历ECsn)
                else:
                    # 简并CO核质量达到 mch 后, 星体坍缩引发超新星爆炸后不会留下恒星遗迹
                    if mcbagb < 1.83:
                        kw = 15
                        aj = 0.0
                        mt = 0.0
                        lum = 1.0e-10
                        r = 1.0e-10
                    # 半简并的碳氧核(非中心)点燃形成简并的氧氖核, 核质量达到 mch 后经电子俘获型超新星爆发, 留下中子星
                    elif mcbagb < 2.25:  # 【更改】把 mass 改成 mcbagb
                        kw = 13
                        mt = 1.3
                    # 非简并的CO核在中心点燃, 最终重元素燃烧生成铁核, 经历铁核坍缩后发生超新星爆炸, 留下中子星或黑洞
                    else:
                        # nsflag(Ture) takes NS/BH mass from Fryer et al. 2012, ApJ, 749, 91. (目前正在使用)
                        if nsflag:
                            (kw, mt) = supernova(mt, mc, mcbagb, SNtype, kick)
                        else:
                            mt = 1.17 + 0.09 * mc
        break

    # Naked Helium Star
    if 7 <= kw <= 9:
        rzams = rzhef(mass)    # 【更改】把当前质量mt换成初始质量mass （78）
        # Main Sequence
        if aj < tm:
            kw = 7
            tau = aj / tm
            lum = lums[1] * (1.0 + 0.45 * tau + max(0.0, 0.85 - 0.08 * mass) * tau ** 2)
            r = rzams * (1.0 + max(0.0, 0.4 - 0.22 * np.log10(mt)) * (tau - tau ** 6))
            rg = rzams    # 这个变量可能之后包层演化程序中会用到
            # Star has no core mass and hence no memory of its past which is
            # why we subject mass and mt to mass loss for this phase.
            mc = 0.0
            if mt < zcnsts.zpars[10]:
                kw = 10
        # Helium Shell Burning
        else:
            kw = 8
            lum = lgbtf(aj, GB[8], GB, tscls[4], tscls[5], tscls[6])
            r = rhehgf(mt, lum, rzams, lums[2])
            rg = rhegbf(lum)
            if r >= rg:
                kw = 9
                r = rg
            mc = mcgbf(lum, GB, lums[6])
            mcmax = min(mt, 1.45 * mt - 0.31)    # 【疑问】这里为什么也是用当前质量mt
            mcmax = min(mcmax, max(mch, 0.773 * mass - 0.35))
            if mcmax - mc < tiny:
                aj = 0.0
                mc = mcmax
                if mc < mch:
                    # Zero-age Carbon/Oxygen White Dwarf
                    if mass < 1.83:
                        mt = max(mc, (mc + 0.31) / 1.45)
                        kw = 11
                    # Zero-age Oxygen/Neon White Dwarf
                    else:
                        mt = mc
                        kw = 12
                        if mt >= 1.38:
                            kw = 13
                            mt = 1.3
                    mass = mt
                else:
                    # Star is not massive enough to ignite C burning so no remnant is left after the SN
                    if mass < 1.83:
                        kw = 15
                        aj = 0.0
                        mt = 0.0
                        lum = 1e-10
                        r = 1e-10
                    elif mass < 2.25:
                        kw = 13
                        mt = 1.3
                    else:
                        if nsflag:
                            (kw, mt) = supernova(mt, mc, mt, SNtype, kick)
                        else:
                            mt = 1.17 + 0.09 * mc

    # White dwarf
    if 10 <= kw <= 12:
        mc = mt
        if mc >= mch:
            # AIC引发超新星爆炸, 只有ONe白矮星会坍缩成白矮星, 其他都会消失
            if kw == 12:
                kw = 13
                aj = 0.0
                mt = 1.3
            else:
                kw = 15
                aj = 0.0
                mt = 0.0
                lum = 1e-10
                r = 1e-10
        else:
            if kw == 10:
                xx = ahe
            else:
                xx = aco
            # modified-Mestel cooling  (未使用)
            if wdflag:
                if aj < 9000.0:
                    lum = 300.0 * mt * zcnsts.zpars[14] / (xx * (aj + 0.1)) ** 1.18
                else:
                    fac = (9000.1 * xx) ** 5.3
                    lum = 300.0 * fac * mt * zcnsts.zpars[14] / (xx * (aj + 0.10)) ** 6.48
            # Mestel cooling
            else:
                lum = 635.0 * mt * zcnsts.zpars[14] / (xx * (aj + 0.1)) ** 1.4

            r = max(1e6/Rsun, 0.0115 * np.sqrt((mch / mt) ** (2 / 3) - (mt / mch) ** (2 / 3)))
            r = min(0.1, r)
            if mt < 0.0005:
                r = 0.09
            if mt < 0.000005:
                r = 0.009

    # Neutron Star
    if kw == 13:
        mc = mt
        # Accretion induced Black Hole?
        if mc > mxns:
            kw = 14
            aj = 0.0
        else:
            lum = 0.02 * mt ** (2 / 3) / (max(aj, 0.1)) ** 2
            r = 1.4e-5

    # Black hole
    if kw == 14:
        mc = mt
        lum = 1.0e-10
        r = 4.24e-6 * mt

    # 计算核半径、核光度以及最后形成的致密星的半径
    tau = 0.0
    # 主序阶段
    if kw <= 1 or kw == 7:
        rc = 0.0
        lc = 0.0
    # 赫氏空隙/巨星阶段
    elif 2 <= kw <= 3:
        # 非简并的氦核
        if mass > zcnsts.zpars[2]:
            lc = lzhef(mc)
            rc = rzhef(mc)
        # 简并氦核
        else:
            if wdflag:
                lc = 300.0 * mc * zcnsts.zpars[14] / ((ahe * 0.1) ** 1.18)
            else:
                lc = 635.0 * mc * zcnsts.zpars[14] / ((ahe * 0.1) ** 1.4)
            rc = 0.0115 * np.sqrt(max(1.48204e-6, (mch / mc) ** (2.0 / 3.0) - (mc / mch) ** (2.0 / 3.0)))
            rc = 5.0 * rc
    # 水平分支
    elif kw == 4:
        tau = (aj - tscls[2]) / tscls[3]
        # 先把此时的核当作是一个氦主序, 计算核的半径和光度
        kw_temp = 7
        (tm, tn, tscls, lums, GB) = star(kw_temp, mc, mc, zcnsts)
        lc = lums[1] * (1.0 + 0.45 * tau + max(0.0, 0.85 - 0.08 * mc) * tau ** 2)
        rc = rzhef(mc) * (1.0 + max(0.0, 0.4 - 0.22 * np.log10(mc)) * (tau - tau ** 6))
        # 恢复恒星本身类型对应的特征光度/时标
        (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
    # EAGB 阶段
    elif kw == 5:
        # 先把此时的核当作是一个氦巨星, 计算核的半径和光度
        kw_temp = 9
        tbagb = tscls[2] + tscls[3]
        if tn > tbagb:
            tau = 3.0 * (aj - tbagb) / (tn - tbagb)
        (tm, tn, tscls, lums, GB) = star(kw_temp, mc, mc, zcnsts)
        lc = lmcgbf(mcx, GB)
        if tau < 1.0:
            lc = lums[2] * (lc / lums[2]) ** tau
        rc = rzhef(mc)
        rc = min(rhehgf(mc, lc, rc, lums[2]), rhegbf(lc))
        # 恢复恒星本身类型对应的特征光度/时标
        (tm, tn, tscls, lums, GB) = star(kw, mass, mt, zcnsts)
    # TPAGB/HeHG/HeGB
    elif kw == 6 or 8 <= kw <= 9:
        if wdflag:
            lc = 300.0 * mc * zcnsts.zpars[14] / ((aco * 0.10) ** 1.18)
        else:
            lc = 635.0 * mc * zcnsts.zpars[14] / ((aco * 0.10) ** 1.4)
        rc = 0.01150 * np.sqrt(max(1.48204e-6, (mch / mc) ** (2.0 / 3.0) - (mc / mch) ** (2.0 / 3.0)))
        rc = 5.0 * rc
    # 致密星
    else:
        lc = 0
        rc = r
        menv = 1.0e-10
        renv = 1.0e-10
        k2 = 0.21

    # Perturb the luminosity and radius due to small envelope mass (except for MS star).
    if 2 <= kw <= 9 and kw != 7:
        kap = -0.5
        lum0 = 7e4
        mu = ((mt - mc) / mt) * min(5.0, max(1.2, (lum / lum0) ** kap))
        if kw >= 8:
            mcmax = min(mt, 1.45 * mt - 0.31)
            mu = ((mcmax - mc) / mcmax) * 5.0
        if mu < 1.0:
            lpert = lpertf(mt, mu)
            lum = lc * (lum / lc) ** lpert
            if r <= rc:
                rpert = 0.0
            else:
                rpert = rpertf(mt, mu, r, rc)
            r = rc * (r / rc) ** rpert
        rc = min(rc, r)

    # Calculate mass and radius of convective envelope, and envelope gyration radius.
    if kw < 10:
        rtms = rtmsf(mass, zcnsts)   # 【疑问】这里的rtms公式是否对氦星适用
        rzams = rzamsf(mass, zcnsts) if kw <= 6 else rzhef(mass)
        (menv, renv, k2) = mrenv(kw, mass, mt, mc, lum, r, rc, aj, tm, lums[2], lums[3], lums[4], rzams, rtms, rg, menv,
                                 renv, k2)
    if mass > 99.990:
        mass = mass0
    if mt > 99.990:
        mt = mt0

    return mass, aj, mt, tm, tn, tscls, lums, GB, r, lum, kw, mc, rc, menv, renv, k2

