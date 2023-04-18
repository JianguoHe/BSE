import numpy as np
from numba import njit
import pysnooper
from mix import mix
from star import star
from hrdiag import hrdiag
from mlwind import mlwind
from zfuncs import vrotf, rochelobe
from vkick import vkick
from timestep import timestep
from gntage import gntage
from comenv import comenv
from corerd import corerd
from const import neta, beta_wind, alpha_wind, acc1
from const import ktype, xi, tflag, eddfac, mch, epsnov, gamma, yeardy, aursun, tiny


# @pysnooper.snoop()
@njit
def evolve(kstar, mass0, mass, rad, lumin, massc, radc, menv, renv,
           ospin, epoch, tms, tphys, tphysf, dtp, z, zcnsts, tb, ecc, kick, output):
    # 保存初始状态
    mass1i = mass0[1]
    mass2i = mass0[2]
    tbi = tb
    ecci = ecc

    ngtv = -1.0
    ngtv2 = -2.0
    trl = -1.0

    # 初始化参数
    kmin = 1
    kmax = 2
    sgl = False
    mt2 = min(mass[1], mass[2])
    kst = 0

    # 未初始化参数
    tscls = np.zeros((1, 21)).flatten()
    lums = np.zeros((1, 11)).flatten()
    GB = lums.copy()
    vs = np.array([0.0, 0, 0, 0])
    aj = np.array([0.0, 0, 0])
    aj0 = aj.copy()
    q = aj.copy()
    rol = aj.copy()
    radx = aj.copy()
    rol0 = aj.copy()
    rdot = aj.copy()
    mass00 = aj.copy()
    mcxx = aj.copy()
    jspin = aj.copy()
    dspint = aj.copy()
    djspint = aj.copy()
    djtx = aj.copy()
    dtmi = aj.copy()
    k2str = aj.copy()
    tbgb = aj.copy()
    tkh = aj.copy()
    dms = aj.copy()
    dmr = aj.copy()
    dmt = aj.copy()

    tm = 0
    tmsnew = 0
    tn = 0
    rm = 0  # 恒星半径
    lum = 0
    mc = 0
    rc = 0
    me = 0
    re = 0
    k2 = 0
    qc = 0
    dtr = 0
    dtm0 = 0
    dm22 = 0
    tphys00 = 0

    # 设置常数
    k3 = 0.21
    mr23yr = 0.4311
    loop = 20000

    # 设置循环逻辑变量，用来替代原来的goto语句
    flag5 = True
    flag7 = False
    flag130 = False
    flag135 = False
    flag140 = False

    # 计算轨道角动量
    if mt2 < 0 or tb <= 0:
        sgl = True
        if mt2 < tiny:
            mt2 = 0.0
            if mass[1] < tiny:
                if tphys < tiny:
                    mass0[1] = 0.010
                    mass[1] = mass0[1]
                    kst = 1
                else:
                    kmin = 2
                    lumin[1] = 1e-10
                    rad[1] = 1e-10
                    massc[1] = 0.0
                    dmt[1] = 0.0
                    dmr[1] = 0.0
                ospin[1] = 1e-10
                jspin[1] = 1e-10
            else:
                if tphys < tiny:
                    mass0[2] = 0.010
                    mass[2] = mass0[2]
                    kst = 2
                else:
                    kmax = 1
                    lumin[2] = 1e-10
                    rad[2] = 1e-10
                    massc[2] = 0.0
                    dmt[2] = 0.0
                    dmr[2] = 0.0
                ospin[2] = 1e-10
                jspin[2] = 1e-10
        ecc = -1.0
        tb = 0.0
        sep = 1e10
        oorb = 0.0
        jorb = 0.0
        if ospin[1] < 0.0:
            ospin[1] = 1e-10
        if ospin[2] < 0.0:
            ospin[2] = 1e-10
        q[1] = 1e10
        q[2] = 1e10
        rol[1] = 1e10
        rol[2] = 1e10
    else:
        tb = tb / yeardy     # 以下的周期都将以年作为单位
        sep = aursun * (tb * tb * (mass[1] + mass[2])) ** (1 / 3)
        oorb = 2 * np.pi / tb
        jorb = mass[1] * mass[2] / (mass[1] + mass[2]) * np.sqrt(1 - ecc ** 2) * sep * sep * oorb
        if ospin[1] < 0:
            ospin[1] = oorb
        if ospin[2] < 0:
            ospin[2] = oorb

    for k in range(kmin, kmax + 1):
        age = tphys - epoch[k]
        (tm, tn, tscls, lums, GB) = star(kstar[k], mass0[k], mass[k], zcnsts)

        (mass0[k], age, mass[k], tm, tn, tscls, lums, GB, rm, lum, kstar[k], mc, rc, me, re, k2) = hrdiag(
            mass0[k], age, mass[k], tm, tn, tscls, lums, GB, zcnsts, rm, lum, kstar[k], mc, rc, me, re, k2, kick)

        aj[k] = age
        epoch[k] = tphys - age
        rad[k] = rm
        lumin[k] = lum
        massc[k] = mc
        radc[k] = rc
        menv[k] = me
        renv[k] = re
        k2str[k] = k2
        tms[k] = tm
        tbgb[k] = tscls[1]
        if tphys < tiny and ospin[k] <= 0.001:
            ospin[k] = 45.35 * vrotf(mass[k]) / rm
        jspin[k] = ospin[k] * (k2 * rm * rm * (mass[k] - mc) + k3 * rc * rc * mc)
        if not sgl:
            q[k] = mass[k] / mass[3 - k]
            rol[k] = rochelobe(q[k]) * sep
        rol0[k] = rol[k]
        dmr[k] = 0.0
        dmt[k] = 0.0
        djspint[k] = 0.0
        dtmi[k] = 1e6

    if mt2 < tiny:
        sep = 0.0
        if kst > 0:
            mass0[kst] = 0.0
            mass[kst] = 0.0
            kmin = 3 - kst
            kmax = kmin

    # On the first entry the previous timestep is zero to prevent mass loss.
    dtm = 0.0  # 时间间隔（以百万年 Myr 为单位）
    delet = 0.0
    djorb = 0.0
    bss = False

    # Setup variables which control the output (if it is required).
    ip = 0
    jp = 0
    tsave = tphys
    isave = True
    iplot = False
    if dtp <= 0.0:
        iplot = True
        isave = False
        tsave = tphysf
    elif dtp > tphysf:
        isave = False
        tsave = tphysf

    if tphys >= tphysf:
        # goto.one_four_zero
        flag140 = True

    while not flag140:
        while True:
            while True:
                # label.four
                if flag5:
                    iter = 0
                    intpol = 0
                    inttry = False
                    change = False
                    prec = False
                    snova = False
                    coel = False
                    com = False
                    bsymb = False
                    esymb = False
                    tphys0 = tphys
                    ecc1 = ecc
                    j1 = 1
                    j2 = 2
                    if 10 <= kstar[1] <= 14:
                        dtmi[1] = 0.01
                    if 10 <= kstar[2] <= 14:
                        dtmi[2] = 0.01
                    dm1 = 0.0
                    dm2 = 0.0
                flag5 = True  # 重置flag5
                # label.five
                kw1 = kstar[1]
                kw2 = kstar[2]

                dt = 1e6 * dtm  # 时间间隔（以年为单位）
                eqspin = 0.0
                djtt = 0.0

                if intpol == 0 and abs(dtm) > tiny and not sgl:
                    vorb2 = acc1 * (mass[1] + mass[2]) / sep
                    ivsqm = 1.0 / np.sqrt(1.0 - ecc * ecc)
                    for k in range(1, 3):
                        if neta > tiny:
                            # 计算星风质量损失，用 dmr 表示
                            rlperi = rol[k] * (1.0 - ecc)
                            dmr[k] = mlwind(kstar[k], lumin[k], rad[k], mass[k], massc[k], rlperi, z)
                            # 计算伴星从星风中吸积的质量, 用 dmt 表示(Boffin & Jorissen, A&A 1988, 205, 155).
                            vwind2 = 2.0 * beta_wind * acc1 * mass[k] / rad[k]
                            omv2 = (1.0 + vorb2 / vwind2) ** (3.0 / 2.0)
                            dmt[3 - k] = ivsqm * alpha_wind * dmr[k] * ((acc1 * mass[3 - k] / vwind2) ** 2) / (
                                    2.0 * sep * sep * omv2)
                            dmt[3 - k] = min(dmt[3 - k], 0.8 * dmr[k])
                        else:
                            dmr[k] = 0.0
                            dmt[3 - k] = 0.0
                    # 诊断共生星（Symbiotic-type stars）
                    if neta > tiny and not esymb:
                        lacc = 3.14e7 * mass[j2] * dmt[j2] / rad[j2]
                        lacc = lacc / lumin[j1]
                        if (lacc > 0.01 and not bsymb) or (lacc < 0.01 and bsymb):
                            jp = min(80, jp + 1)
                            output.bpp[jp, 1] = tphys
                            output.bpp[jp, 2] = mass[1]
                            output.bpp[jp, 3] = mass[2]
                            output.bpp[jp, 4] = kstar[1]
                            output.bpp[jp, 5] = kstar[2]
                            output.bpp[jp, 6] = sep
                            output.bpp[jp, 7] = ecc
                            output.bpp[jp, 8] = rad[1] / rol[1]
                            output.bpp[jp, 9] = rad[2] / rol[2]
                            if (bsymb):
                                output.bpp[jp, 10] = 13.0
                                esymb = True
                            else:
                                output.bpp[jp, 10] = 12.0
                                bsymb = True
                    # 由于星风损失的轨道角动量
                    ecc2 = ecc * ecc
                    omecc2 = 1.0 - ecc2
                    sqome2 = np.sqrt(omecc2)

                    djorb = ((dmr[1] + q[1] * dmt[1]) * mass[2] * mass[2] + (dmr[2] + q[2] * dmt[2]) * mass[1] * mass[1]
                             ) * sep * sep * sqome2 * oorb / (mass[1] + mass[2]) ** 2
                    delet = ecc * (dmt[1] * (0.50 / mass[1] + 1.0 / (mass[1] + mass[2])) + dmt[2] * (
                            0.50 / mass[2] + 1.0 / (mass[1] + mass[2])))
                    # 密近双星的引力波辐射导致轨道角动量损失
                    if sep <= 1000.0:
                        djgr = 8.315e-10 * mass[1] * mass[2] * (mass[1] + mass[2]) / (sep * sep * sep * sep)
                        f1 = (19.0 / 6.0) + (121.0 / 96.0) * ecc2
                        sqome5 = sqome2 ** 5
                        delet1 = djgr * ecc * f1 / sqome5
                        djgr = djgr * jorb * (1.0 + 0.8750 * ecc2) / sqome5
                        djorb = djorb + djgr
                        delet = delet + delet1
                    for k in range(1, 3):
                        # 计算星风带走的恒星自旋角动量（包括吹走的和吸积过来的）
                        djtx[k] = (2.0 / 3.0) * xi * dmt[k] * rad[3 - k] * rad[3 - k] * ospin[3 - k]
                        djspint[k] = (2.0 / 3.0) * (dmr[k] * rad[k] * rad[k] * ospin[k]) - djtx[k]
                        # 计算有明显对流包层的恒星因磁制动损失的自旋角动量
                        # 包括主序星(M < 1.25)、靠近巨星分支的HG恒星以及巨星, 不包括完全对流主序星
                        if mass[k] > 0.35 and kstar[k] < 10:
                            djmb = 5.83e-16 * menv[k] * (rad[k] * ospin[k]) ** 3 / mass[k]
                            djspint[k] = djspint[k] + djmb
                            # 限制最大3%的磁制动损失的角动量。这可以保证迭代次数不会超过最大值20000, 当然2%也不会影响演化结果
                            if djmb > tiny:
                                dtj = 0.03 * jspin[k] / abs(djmb)
                                dt = min(dt, dtj)
                        # 计算圆化、轨道收缩和自旋
                        dspint[k] = 0.0
                        if (((kstar[k] <= 9 and rad[k] >= 0.010 * rol[k]) or (
                                kstar[k] >= 10 and k == j1)) and tflag > 0):
                            raa2 = (rad[k] / sep) ** 2
                            raa6 = raa2 ** 3
                            # 赫维茨多项式
                            f5 = 1.0 + ecc2 * (3.0 + ecc2 * 0.3750)
                            f4 = 1.0 + ecc2 * (1.50 + ecc2 * 0.1250)
                            f3 = 1.0 + ecc2 * (3.750 + ecc2 * (1.8750 + ecc2 * 7.8125e-2))
                            f2 = 1.0 + ecc2 * (7.50 + ecc2 * (5.6250 + ecc2 * 0.31250))
                            f1 = 1.0 + ecc2 * (15.50 + ecc2 * (31.8750 + ecc2 * (11.56250 + ecc2 * 0.3906250)))
                            if (kstar[k] == 1 and mass[k] >= 1.250) or kstar[k] == 4 or kstar[k] == 7:
                                # 辐射阻尼(Zahn, 1977, A&A, 57, 383 and 1975, A&A, 41, 329)
                                tc = 1.592e-9 * (mass[k] ** 2.840)
                                f = 1.9782e4 * np.sqrt((mass[k] * rad[k] * rad[k]) / sep ** 5) * tc * (
                                        1.0 + q[3 - k]) ** (5.0 / 6.0)
                                tcqr = f * q[3 - k] * raa6
                                rg2 = k2str[k]
                            elif kstar[k] <= 9:
                                # 对流阻尼(Hut, 1981, A&A, 99, 126)
                                tc = mr23yr * (menv[k] * renv[k] * (rad[k] - 0.50 * renv[k]) / (3.0 * lumin[k])) ** (
                                        1.0 / 3.0)
                                ttid = 2 * np.pi / (1e-10 + abs(oorb - ospin[k]))
                                f = min(1.0, (ttid / (2.0 * tc) ** 2))
                                tcqr = 2.0 * f * q[3 - k] * raa6 * menv[k] / (21.0 * tc * mass[k])
                                rg2 = (k2str[k] * (mass[k] - massc[k])) / mass[k]
                            else:
                                # 简并阻尼(Campbell, 1984, MNRAS, 207, 433)
                                f = 7.33e-9 * (lumin[k] / mass[k]) ** (5.0 / 7.0)
                                tcqr = f * q[3 - k] * q[3 - k] * raa2 * raa2 / (1.0 + q[3 - k])
                                rg2 = k3
                            # 计算圆化
                            sqome3 = sqome2 ** 3
                            delet1 = 27.0 * tcqr * (1.0 + q[3 - k]) * raa2 * (ecc / sqome2 ** 13) * (
                                    f3 - (11.0 / 18.0) * sqome3 * f4 * ospin[k] / oorb)
                            tcirc = ecc / (abs(delet1) + 1.0e-20)
                            delet = delet + delet1
                            # 计算自旋
                            dspint[k] = (3.0 * q[3 - k] * tcqr / (rg2 * omecc2 ** 6)) * (
                                    f2 * oorb - sqome3 * f5 * ospin[k])
                            # 计算无角动量转移时的平衡自旋
                            eqspin = oorb * f2 / (sqome3 * f5)
                            # 计算潮汐造成的轨道角动量变化
                            djt = (k2str[k] * (mass[k] - massc[k]) * rad[k] * rad[k] + k3 * massc[k] * radc[k] * radc[
                                k]) * dspint[k]
                            if kstar[k] <= 6 or abs(djt) / jspin[k] > 0.1:
                                djtt = djtt + djt
                    # 限制最大 2% 的轨道角动量变化
                    djtt = djtt + djorb
                    if abs(djtt) > tiny:
                        dtj = 0.002 * jorb / abs(djtt)
                        dt = min(dt, dtj)
                    dtm = dt / 1.0e6
                elif abs(dtm) > tiny and sgl:
                    for k in range(kmin, kmax + 1):
                        if neta > tiny:
                            rlperi = 0.0
                            dmr[k] = mlwind(kstar[k], lumin[k], rad[k], mass[k], massc[k], rlperi, z)
                        else:
                            dmr[k] = 0.0
                        # endif
                        dmt[k] = 0.0
                        djspint[k] = (2.0 / 3.0) * dmr[k] * rad[k] * rad[k] * ospin[k]
                        if (mass[k] > 0.35 and kstar[k] < 10):
                            djmb = 5.83e-16 * menv[k] * (rad[k] * ospin[k]) ** 3 / mass[k]
                            djspint[k] = djspint[k] + djmb
                            if djmb > tiny:
                                dtj = 0.03 * jspin[k] / abs(djmb)    #################   0.03
                                dt = min(dt, dtj)
                    dtm = dt / 1.0e6

                # 通过改变步长来控制双星的质量损失不超过特定值
                for k in range(kmin, kmax + 1):
                    dms[k] = (dmr[k] - dmt[k]) * dt
                    # 对于非致密星
                    if kstar[k] < 10:
                        dml = mass[k] - massc[k]
                        # 每次质量损失不超过包层质量
                        if dml < dms[k]:
                            dml = max(dml, 2.0 * tiny)
                            dtm = (dml / dms[k]) * dtm
                            if k == 2:
                                dms[1] = dms[1] * dml / dms[2]
                            dms[k] = dml
                            dt = 1.0e6 * dtm
                        # 限制 1% 的质量损失
                        if dms[k] > 0.01 * mass[k]:
                            dtm = 0.01 * mass[k] * dtm / dms[k]
                            if k == 2:
                                dms[1] = dms[1] * 0.01 * mass[2] / dms[2]
                            dms[k] = 0.01 * mass[k]
                            dt = 1.0e6 * dtm

                # 更新质量和自旋(检查恒星自旋没有超过临界值), 重置主序星(也可能是巨星)的 epoch.
                for k in range(kmin, kmax + 1):
                    if eqspin > 0.0 and abs(dspint[k]) > tiny:
                        if intpol == 0:
                            if dspint[k] >= 0.0:
                                dspint[k] = min(dspint[k], (eqspin - ospin[k]) / dt)
                            else:
                                dspint[k] = max(dspint[k], (eqspin - ospin[k]) / dt)
                            djt = (k2str[k] * (mass[k] - massc[k]) * rad[k] * rad[k] + k3 * massc[k] * radc[k] * radc[
                                k]) * dspint[k]
                            djorb = djorb + djt
                            djspint[k] = djspint[k] - djt
                    jspin[k] = max(1.0e-10, jspin[k] - djspint[k] * dt)
                    # 确保恒星的自旋不会导致瓦解
                    ospbru = 2 * np.pi * np.sqrt(mass[k] * aursun ** 3 / rad[k] ** 3)
                    jspbru = (k2str[k] * (mass[k] - massc[k]) * rad[k] * rad[k] + k3 * massc[k] * radc[k] * radc[
                        k]) * ospbru
                    if jspin[k] > jspbru and abs(dtm) > tiny:
                        mew = 1.0
                        if djtx[k] > 0.0:
                            mew = min(mew, (jspin[k] - jspbru) / djtx[k])
                        jspin[k] = jspbru
                        # 如果多余的物质不应该被吸积，激活下一行
                        # dms[k] = dms[k] + (1.0 - mew)*dmt[k]*dt
                    if abs(dms[k]) > tiny:
                        mass[k] = mass[k] - dms[k]
                        if kstar[k] <= 2 or kstar[k] == 7:
                            m0 = mass0[k]
                            mass0[k] = mass[k]
                            (tm, tn, tscls, lums, GB) = star(kstar[k], mass0[k], mass[k], zcnsts)
                            if kstar[k] == 2:
                                if GB[9] < massc[k] or m0 > zcnsts.zpars[3]:
                                    mass0[k] = m0
                                else:
                                    epoch[k] = tm + (tscls[1] - tm) * (aj[k] - tms[k]) / (tbgb[k] - tms[k])
                                    epoch[k] = tphys - epoch[k]
                            else:
                                epoch[k] = tphys - aj[k] * tm / tms[k]

                if not sgl:
                    ecc1 = ecc1 - delet * dt
                    ecc = max(ecc1, 0.0)
                    if ecc < 1.0e-10:
                        ecc = 0.0
                    if ecc >= 1.0:
                        # goto.one_three_five
                        flag135 = True
                        break
                    jorb = jorb - djorb * dt
                    sep = (mass[1] + mass[2]) * jorb * jorb / (
                            (mass[1] * mass[2] * 2 * np.pi) ** 2 * aursun ** 3 * (1.0 - ecc * ecc))
                    tb = (sep / aursun) * np.sqrt(sep / (aursun * (mass[1] + mass[2])))
                    oorb = 2 * np.pi / tb

                # 更新时间.
                if intpol == 0:
                    tphys0 = tphys
                    dtm0 = dtm
                tphys = tphys + dtm

                for k in range(kmin, kmax + 1):
                    # 获得明显演化年龄时的恒星参数（M, R, L, Mc & K*）
                    age = tphys - epoch[k]
                    aj0[k] = age
                    kw = kstar[k]
                    m0 = mass0[k]
                    mt = mass[k]
                    mc = massc[k]
                    if intpol == 0:
                        mcxx[k] = mc
                    if intpol > 0:
                        mc = mcxx[k]
                    mass00[k] = m0
                    # 100个太阳以上的质量不应该放在演化公式中
                    if mt > 100.0:
                        # goto.one_four_zero
                        flag140 = True
                        break
                    (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)

                    (m0, age, mt, tm, tn, tscls, lums, GB, rm, lum, kw, mc, rc, me, re, k2) = hrdiag(
                        m0, age, mt, tm, tn, tscls, lums, GB, zcnsts, rm, lum, kw, mc, rc, me, re, k2, kick)

                    if kw != 15:
                        ospin[k] = jspin[k] / (k2 * (mt - mc) * rm * rm + k3 * mc * rc * rc)

                    # 这个时候可能已经发生超新星爆发了
                    if kw != kstar[k] and kstar[k] <= 12 and (kw == 13 or kw == 14):
                        if sgl:
                            vkick_return = vkick(kw, mass[k], mt, 0.0, 0.0, -1.0, 0.0, vs, kick)
                            (kw, mass[k], mt, vs) = (vkick_return[0], vkick_return[1], vkick_return[2], vkick_return[7])
                        else:
                            (kw, mass[k], mt, mass[3 - k], ecc, sep, jorb, vs) = vkick(
                                kw, mass[k], mt, mass[3 - k], ecc, sep, jorb, vs, kick)
                            if ecc > 1.0:
                                kstar[k] = kw
                                mass[k] = mt
                                epoch[k] = tphys - age
                                # goto.one_three_five
                                flag135 = True
                                break
                            tb = (sep / aursun) * np.sqrt(sep / (aursun * (mt + mass[3 - k])))
                            oorb = 2 * np.pi / tb
                        snova = True
                    if kw != kstar[k]:
                        change = True
                        mass[k] = mt
                        dtmi[k] = 0.01
                        if kw == 15:
                            kstar[k] = kw
                            # goto.one_three_five
                            flag135 = True
                            break
                        mass0[k] = m0
                        epoch[k] = tphys - age
                        if kw > 6 and kstar[k] <= 6:
                            bsymb = False
                            esymb = False
                    # 令新生的中子星或黑洞拥有第二个周期
                    if kstar[k] == 13 or kstar[k] == 14:
                        if tphys - epoch[k] < tiny:
                            ospin[k] = 2.0e8
                            jspin[k] = k3 * rc * rc * mc * ospin[k]

                    # Set radius derivative for later interpolation.
                    if abs(dtm) > tiny:
                        rdot[k] = abs(rm - rad[k]) / dtm
                    else:
                        rdot[k] = 0.0

                    # Base new time scale for changes in radius & mass on stellar type.
                    dt = dtmi[k]
                    (dt, dtr) = timestep(kw, age, tm, tn, tscls, dt)

                    # 选择最小时标并且保留间隔
                    dtmi[k] = min(dt, dtr)

                    # Save relevent solar quantities.
                    aj[k] = age
                    kstar[k] = kw
                    rad[k] = rm
                    lumin[k] = lum
                    massc[k] = mc
                    radc[k] = rc
                    menv[k] = me
                    renv[k] = re
                    k2str[k] = k2
                    tms[k] = tm
                    tbgb[k] = tscls[1]

                    # Check for blue straggler formation.
                    if kw <= 1 and tm < tphys and not bss:
                        bss = True
                        jp = min(80, jp + 1)
                        output.bpp[jp, 1] = tphys
                        output.bpp[jp, 2] = mass[1]
                        output.bpp[jp, 3] = mass[2]
                        output.bpp[jp, 4] = float(kstar[1])
                        output.bpp[jp, 5] = float(kstar[2])
                        output.bpp[jp, 6] = sep
                        output.bpp[jp, 7] = ecc
                        output.bpp[jp, 8] = rad[1] / rol[1]
                        output.bpp[jp, 9] = rad[2] / rol[2]
                        output.bpp[jp, 10] = 14.0

                if flag135 or flag140:
                    break

                if not sgl:
                    # 确定质量比
                    for k in range(1, 3):
                        q[k] = mass[k] / mass[3 - k]
                    # 确定洛希瓣半径并且调整半径导数
                    for k in range(1, 3):
                        rol[k] = rochelobe(q[k]) * sep
                        if abs(dtm) > tiny:
                            rdot[k] = rdot[k] + (rol[k] - rol0[k]) / dtm
                            rol0[k] = rol[k]
                else:
                    for k in range(kmin, kmax + 1):
                        rol[k] = 10000.0 * rad[k]

                if (tphys < tiny and abs(dtm) < tiny and (mass2i < 0.10 or not sgl)) or snova:
                    jp = min(80, jp + 1)
                    output.bpp[jp, 1] = tphys
                    output.bpp[jp, 2] = mass[1]
                    output.bpp[jp, 3] = mass[2]
                    output.bpp[jp, 4] = float(kstar[1])
                    output.bpp[jp, 5] = float(kstar[2])
                    output.bpp[jp, 6] = sep
                    output.bpp[jp, 7] = ecc
                    output.bpp[jp, 8] = rad[1] / rol[1]
                    output.bpp[jp, 9] = rad[2] / rol[2]
                    output.bpp[jp, 10] = 1.0
                    if snova:
                        output.bpp[jp, 10] = 2.0
                        dtm = 0.0
                        # goto.four
                        continue

                if (isave and tphys >= tsave) or iplot:
                    if sgl or (rad[1] < rol[1] and rad[2] < rol[2]) or tphys < tiny:
                        ip = ip + 1
                        output.bcm[ip, 1] = tphys
                        output.bcm[ip, 2] = kstar[1]
                        output.bcm[ip, 3] = mass0[1]
                        output.bcm[ip, 4] = mass[1]
                        output.bcm[ip, 5] = np.log10(lumin[1])
                        output.bcm[ip, 6] = np.log10(rad[1])
                        teff1 = 1000.0 * ((1130.0 * lumin[1] / (rad[1] ** 2.0)) ** (1.0 / 4.0))
                        output.bcm[ip, 7] = np.log10(teff1)
                        output.bcm[ip, 8] = massc[1]
                        output.bcm[ip, 9] = radc[1]
                        output.bcm[ip, 10] = menv[1]
                        output.bcm[ip, 11] = renv[1]
                        output.bcm[ip, 12] = epoch[1]
                        output.bcm[ip, 13] = ospin[1]
                        output.bcm[ip, 14] = dmt[1] - dmr[1]
                        output.bcm[ip, 15] = rad[1] / rol[1]
                        output.bcm[ip, 16] = kstar[2]
                        output.bcm[ip, 17] = mass0[2]
                        output.bcm[ip, 18] = mass[2]
                        output.bcm[ip, 19] = np.log10(lumin[2])
                        output.bcm[ip, 20] = np.log10(rad[2])
                        teff2 = 1000.0 * ((1130.0 * lumin[2] / (rad[2] ** 2.0)) ** (1.0 / 4.0))
                        output.bcm[ip, 21] = np.log10(teff2)
                        output.bcm[ip, 22] = massc[2]
                        output.bcm[ip, 23] = radc[2]
                        output.bcm[ip, 24] = menv[2]
                        output.bcm[ip, 25] = renv[2]
                        output.bcm[ip, 26] = epoch[2]
                        output.bcm[ip, 27] = ospin[2]
                        output.bcm[ip, 28] = dmt[2] - dmr[2]
                        output.bcm[ip, 29] = rad[2] / rol[2]
                        output.bcm[ip, 30] = tb
                        output.bcm[ip, 31] = sep
                        output.bcm[ip, 32] = ecc
                        if (isave):
                            tsave = tsave + dtp

                # If not interpolating set the next timestep.
                if intpol == 0:
                    dtm = max(1e-7 * tphys, min(dtmi[1], dtmi[2]))
                    dtm = min(dtm, tsave - tphys)
                    if iter == 0:
                        dtm0 = dtm
                while not sgl:
                    # goto.nine_eight(if sgl)
                    # donor（主星）设为 j1
                    # accretor（次星）设为 j2
                    if intpol == 0:
                        if rad[1] / rol[1] >= rad[2] / rol[2]:
                            j1 = 1
                            j2 = 2
                        else:
                            j1 = 2
                            j2 = 1

                    # 检查洛希瓣渗溢是否开始
                    if rad[j1] > rol[j1]:
                        # Interpolate back until the primary is just filling its Roche lobe.
                        if rad[j1] >= 1.002 * rol[j1]:
                            if intpol == 0:
                                tphys00 = tphys
                            intpol = intpol + 1
                            if iter == 0:
                                # goto.sewen
                                flag7 = True
                                break
                            if inttry:
                                # goto.sewen
                                flag7 = True
                                break
                            if intpol >= 100:
                                # goto.one_four_zero
                                flag140 = True
                                break
                            dr = rad[j1] - 1.001 * rol[j1]
                            if abs(rdot[j1]) < tiny or prec:
                                # goto.sewen
                                flag7 = True
                                break
                            dtm = -dr / abs(rdot[j1])
                            if abs(tphys0 - tphys) > tiny:
                                dtm = max(dtm, tphys0 - tphys)
                            if kstar[1] != kw1:
                                kstar[1] = kw1
                                mass0[1] = mass00[1]
                                epoch[1] = tphys - aj0[1]
                            if kstar[2] != kw2:
                                kstar[2] = kw2
                                mass0[2] = mass00[2]
                                epoch[2] = tphys - aj0[2]
                            change = False
                        else:
                            # 进入洛希瓣渗溢
                            if tphys >= tphysf:
                                # goto.one_four_zero
                                flag140 = True
                                break
                            # goto.sewen
                            flag7 = True
                            break
                    else:
                        # Check if already interpolating.
                        if intpol > 0:
                            intpol = intpol + 1
                            if intpol >= 80:
                                inttry = True
                            if (abs(rdot[j1]) < tiny):
                                prec = True
                                dtm = 1e-7 * tphys
                            else:
                                dr = rad[j1] - 1.001 * rol[j1]
                                dtm = -dr / abs(rdot[j1])
                            if (tphys + dtm) >= tphys00:
                                dtm = 0.5 * (tphys00 - tphys0)
                                dtm = max(dtm, 1e-10)
                                prec = True
                            tphys0 = tphys
                    break

                if flag140:
                    break

                while not flag7:
                    # Go back for the next step or interpolation.
                    # label.nine_eight
                    if tphys >= tphysf and intpol == 0:
                        # goto.one_four_zero
                        flag140 = True
                        break
                    if change:
                        change = False
                        jp = min(80, jp + 1)
                        output.bpp[jp, 1] = tphys
                        output.bpp[jp, 2] = mass[1]
                        output.bpp[jp, 3] = mass[2]
                        output.bpp[jp, 4] = float(kstar[1])
                        output.bpp[jp, 5] = float(kstar[2])
                        output.bpp[jp, 6] = sep
                        output.bpp[jp, 7] = ecc
                        output.bpp[jp, 8] = rad[1] / rol[1]
                        output.bpp[jp, 9] = rad[2] / rol[2]
                        output.bpp[jp, 10] = 2.0
                    iter = iter + 1
                    if iter >= loop:
                        # goto.one_four_zero
                        flag140 = True
                        break

                    # goto.five
                    flag5 = False
                    break

                flag7 = False
                if flag140:
                    break

                if not flag5:
                    continue

                # Set the nuclear timescale in years and slow-down factor.
                # label.sewen
                km0 = dtm0 * 1e3 / tb
                if km0 < tiny:
                    km0 = 0.5

                # Force co-rotation of primary and orbit to ensure that the tides do not
                # lead to unstable Roche (not currently used).
                # if (ospin[j1] > 1.050 * oorb):
                #    ospin[j1] = oorb
                #    jspin[j1] = (k2str[j1] * rad[j1] * rad[j1] * (mass[j1] - massc[j1]) + k3 * radc[j1] * radc[j1] * massc[j1]
                #                 ) * ospin[j1]

                iter = 0
                coel = False
                change = False
                radx[j1] = max(radc[j1], rol[j1])
                radx[j2] = rad[j2]
                jp = min(80, jp + 1)
                output.bpp[jp, 1] = tphys
                output.bpp[jp, 2] = mass[1]
                output.bpp[jp, 3] = mass[2]
                output.bpp[jp, 4] = float(kstar[1])
                output.bpp[jp, 5] = float(kstar[2])
                output.bpp[jp, 6] = sep
                output.bpp[jp, 7] = ecc
                output.bpp[jp, 8] = rad[1] / rol[1]
                output.bpp[jp, 9] = rad[2] / rol[2]
                output.bpp[jp, 10] = 3.0

                if iplot and tphys > tiny:
                    ip = ip + 1
                    output.bcm[ip, 1] = tphys
                    output.bcm[ip, 2] = float(kstar[1])
                    output.bcm[ip, 3] = mass0[1]
                    output.bcm[ip, 4] = mass[1]
                    output.bcm[ip, 5] = np.log10(lumin[1])
                    output.bcm[ip, 6] = np.log10(rad[1])
                    teff1 = 1000.0 * ((1130.0 * lumin[1] / (rad[1] ** 2.0)) ** (1.0 / 4.0))
                    output.bcm[ip, 7] = np.log10(teff1)
                    output.bcm[ip, 8] = massc[1]
                    output.bcm[ip, 9] = radc[1]
                    output.bcm[ip, 10] = menv[1]
                    output.bcm[ip, 11] = renv[1]
                    output.bcm[ip, 12] = epoch[1]
                    output.bcm[ip, 13] = ospin[1]
                    output.bcm[ip, 14] = 0.0
                    output.bcm[ip, 15] = rad[1] / rol[1]
                    output.bcm[ip, 16] = float(kstar[2])
                    output.bcm[ip, 17] = mass0[2]
                    output.bcm[ip, 18] = mass[2]
                    output.bcm[ip, 19] = np.log10(lumin[2])
                    output.bcm[ip, 20] = np.log10(rad[2])
                    teff2 = 1000.0 * ((1130.0 * lumin[2] / (rad[2] ** 2.0)) ** (1.0 / 4.0))
                    output.bcm[ip, 21] = np.log10(teff2)
                    output.bcm[ip, 22] = massc[2]
                    output.bcm[ip, 23] = radc[2]
                    output.bcm[ip, 24] = menv[2]
                    output.bcm[ip, 25] = renv[2]
                    output.bcm[ip, 26] = epoch[2]
                    output.bcm[ip, 27] = ospin[2]
                    output.bcm[ip, 28] = 0.0
                    output.bcm[ip, 29] = rad[2] / rol[2]
                    output.bcm[ip, 30] = tb
                    output.bcm[ip, 31] = sep
                    output.bcm[ip, 32] = ecc

                while True:
                    # label.eight
                    # Eddington limit for accretion on to the secondary in one orbital period.
                    dme = 2.08e-3 * eddfac * (1.0 / (1.0 + zcnsts.zpars[11])) * rad[j2] * tb
                    supedd = False
                    novae = False
                    disk = False

                    # Determine whether the transferred material forms an accretion disk around the secondary or
                    # hits the secondary in a direct stream, by using eq.(1) of Ulrich & Burger (1976, ApJ, 206, 509)
                    # fitted to the calculations of Lubow & Shu (1974, ApJ, 198, 383).
                    rmin = 0.0425 * sep * (q[j2] * (1.0 + q[j2])) ** (1.0 / 4.0)
                    if rmin > rad[j2]:
                        disk = True

                    # Kelvin-Helmholtz time from the modified classical expression.
                    for k in range(1, 3):
                        tkh[k] = 1.0e7 * mass[k] / (rad[k] * lumin[k])
                        if kstar[k] <= 1 or kstar[k] == 7 or kstar[k] >= 10:
                            tkh[k] = tkh[k] * mass[k]
                        else:
                            tkh[k] = tkh[k] * (mass[k] - massc[k])

                    # Dynamical timescale for the primary.
                    tdyn = 5.05e-5 * np.sqrt(rad[j1] ** 3 / mass[j1])

                    # Identify special cases.
                    q1 = mass1i / mass2i

                    # Shao & Li 2014, doi:10.1088/0004-637X/796/1/37
                    if kstar[j1] <= 2 or (kstar[j1] == 4 and mass1i >= 12.0):
                        porb15 = 0.548 + 0.0945 * mass1i - 0.001502 * mass1i ** 2 + 1.0184e-5 * mass1i ** 3 - 6.2267e-8 * mass1i ** 4
                        porb2 = 0.1958 + 0.3278 * mass1i - 0.01159 * mass1i ** 2 + 0.0001708 * mass1i ** 3 - 9.55e-7 * mass1i ** 4
                        porb25 = 6.0143 + 0.01866 * mass1i - 0.0009386 * mass1i ** 2 - 3.709e-5 * mass1i ** 3 + 5.9106e-7 * mass1i ** 4
                        porb3 = 24.6 - 1.85 * mass1i + 0.0784 * mass1i ** 2 - 0.0015 * mass1i ** 3 + 1.024e-5 * mass1i ** 4
                        if mass1i <= 16.0:
                            porb35 = 1772.8 - 551.48 * mass1i + 66.14 * mass1i ** 2 - 3.527 * mass1i ** 3 + 0.07 * mass1i ** 4
                        else:
                            porb35 = 82.17 - 5.697 * mass1i + 0.17297 * mass1i ** 2 - 0.00238 * mass1i ** 3 + 1.206e-5 * mass1i ** 4
                        if mass1i <= 16.0:
                            porb4 = 46511.2 - 10670 * mass1i + 919.88 * mass1i ** 2 - 35.25 * mass1i ** 3 + 0.506 * mass1i ** 4
                        else:
                            porb4 = 153.03 - 8.967 * mass1i + 0.2077 * mass1i ** 2 - 0.00204 * mass1i ** 3 + 6.677e-6 * mass1i ** 4
                        if mass1i <= 24.0:
                            porb5 = 86434.6 - 15494.3 * mass1i + 1041.4 * mass1i ** 2 - 31.017 * mass1i ** 3 + 0.345 * mass1i ** 4
                        else:
                            porb5 = 566.5 - 33.123 * mass1i + 0.7589 * mass1i ** 2 - 0.00776 * mass1i ** 3 + 2.94e-5 * mass1i ** 4
                        if mass1i <= 40.0:
                            porb6 = 219152.8 - 24416.3 * mass1i + 1018.7 * mass1i ** 2 - 18.834 * mass1i ** 3 + 0.1301 * mass1i ** 4
                        else:
                            porb6 = -10744.14 + 856.43 * mass1i - 24.834 * mass1i ** 2 + 0.3147 * mass1i ** 3 - 0.00148 * mass1i ** 4
                        if tbi <= porb15:
                            if q1 <= 1.5:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi <= porb2:
                            if q1 <= 2:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi <= porb25:
                            if q1 <= 2.5:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi <= porb3:
                            if q1 <= 3:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi <= porb35:
                            if q1 <= 3.5:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi <= porb4:
                            if q1 <= 4:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi <= porb5:
                            if q1 <= 5:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi <= porb6:
                            if q1 <= 6:
                                qc = 1e4
                            else:
                                qc = 0
                        elif tbi > porb6:
                            if q1 <= 6:
                                qc = 1e4
                            else:
                                qc = 0
                    elif kstar[j1] == 3 or kstar[j1] == 5 or kstar[j1] == 6:
                        # qc = (1.67d0-zpars(7)+2.d0*(massc(j1)/mass(j1))**5)/2.13d0
                        # Alternatively use condition of Hjellming & Webbink, 1987, ApJ, 318, 794.
                        qc = 0.362 + 1.0 / (3.0 * (1.0 - massc[j1] / mass[j1]))
                    elif 7 <= kstar[j1] <= 9:
                        if tb <= 1.644e-4:
                            qc = 0.1
                        else:
                            qc = 10.0
                    else:
                        qc = 3.0

                    if kstar[j2] == 13:
                        qc = 3.5
                        if 7.0 <= kstar[j1] <= 9.0:
                            if tb <= 1.644e-4:
                                qc = 0.10
                            else:
                                qc = 10.00
                    # 黑洞双星的物质转移稳定性判据【Shao, Y., & Li, X.-D. 2021, ApJ, 920, 81】
                    elif kstar[j2] == 14:
                        if trl >= -10.0:
                            radmax = -173.8 + 45.5 * mass[j1] - 0.18 * mass[j1] ** 2
                            radmin = 6.6 - 26.1 * mass[j1] / mass[j2] + 11.4 * (mass[j1] / mass[j2]) ** 2
                            radrl = rad[j1]
                            if radrl > radmax:
                                qc = 0.0
                            elif radrl < radmin:
                                qc = 0.0
                            else:
                                qc = 100.0
                            trl = tphys - 1.0e6
                        qmax = 2.1 + 0.8 * mass[j2]
                        qrl = mass[j1] / mass[j2]

                        if qrl < 2.0:
                            qc = 100.0
                        elif qrl > qmax:
                            qc = 0.0

                        # 对于黑洞+氦星, 认为系统是稳定的（Tauris, T., Langer, N., & Podsiadlowski, P. 2015, MNRAS, 451, 2123）
                        if 7.0 <= kstar[j1] <= 9.0:
                            qc = 100.0

                    # 低质量主序星的动力学物质转移
                    if kstar[j1] == 0 and q[j1] > 0.695:
                        # This will be dynamical mass transfer of a similar nature to common-envelope evolution.
                        # The result is always a single star placed in *2.
                        taum = np.sqrt(tkh[j1] * tdyn)
                        dm1 = mass[j1]
                        if kstar[j2] <= 1:
                            # Restrict accretion to thermal timescale of secondary.
                            dm2 = taum / tkh[j2] * dm1
                            mass[j2] = mass[j2] + dm2
                            # Rejuvenate if the star is still on the main sequence.
                            mass0[j2] = mass[j2]
                            (tmsnew, tn, tscls, lums, GB) = star(kstar[j2], mass0[j2], mass[j2], zcnsts)
                            # If the star has no convective core then the effective age decreases,
                            # otherwise it will become younger still.
                            if mass[j2] < 0.350 or mass[j2] > 1.250:
                                aj[j2] = tmsnew / tms[j2] * aj[j2] * (mass[j2] - dm2) / mass[j2]
                            else:
                                aj[j2] = tmsnew / tms[j2] * aj[j2]
                            epoch[j2] = tphys - aj[j2]
                        elif kstar[j2] <= 6:
                            # Add all the material to the giant's envelope.
                            dm2 = dm1
                            mass[j2] = mass[j2] + dm2
                            if kstar[j2] == 2:
                                mass0[j2] = mass[j2]
                                (tmsnew, tn, tscls, lums, GB) = star(kstar[j2], mass0[j2], mass[j2], zcnsts)
                                aj[j2] = tmsnew + tscls[1] * (aj[j2] - tms[j2]) / tbgb[j2]
                                epoch[j2] = tphys - aj[j2]
                        elif kstar[j2] <= 12:
                            # Form a new giant envelope.
                            dm2 = dm1
                            kst = ktype[int(kstar[j1]), int(kstar[j2])]
                            if kst > 100:
                                kst = kst - 100
                            if kst == 4:
                                aj[j2] = aj[j2] / tms[j2]
                                massc[j2] = mass[j2]
                            # Check for planets or low-mass WDs.
                            if (kstar[j2] == 10 and mass[j2] < 0.050) or (kstar[j2] >= 11 and mass[j2] < 0.50):
                                kst = kstar[j1]
                                mass[j1] = mass[j2] + dm2
                                mass[j2] = 0.0
                            else:
                                mass[j2] = mass[j2] + dm2
                                (massc[j2], mass[j2], kst, mass0[j2], aj[j2]) = gntage(
                                    massc[j2], mass[j2], kst, zcnsts, mass0[j2], aj[j2])
                                epoch[j2] = tphys - aj[j2]
                            kstar[j2] = kst
                        else:
                            # The neutron star or black hole simply accretes at the Eddington rate.
                            dm2 = min(dme * taum / tb, dm1)
                            if dm2 < dm1:
                                supedd = True
                            mass[j2] = mass[j2] + dm2
                        coel = True
                        if mass[j2] > 0.0:
                            mass[j1] = 0.0
                            kstar[j1] = 15
                        else:
                            kstar[j1] = kstar[j2]
                            kstar[j2] = 15
                        # goto.one_three_five
                        flag135 = True
                        break
                    elif (kstar[j1] in {3, 5, 6, 8, 9} and (q[j1] > qc or radx[j1] <= radc[j1])) or (
                            kstar[j1] in {2, 4} and q[j1] > qc):
                        # 公共包层演化
                        m1ce = mass[j1]
                        m2ce = mass[j2]

                        comenv_return = comenv(mass0[j1], mass[j1], massc[j1], aj[j1], jspin[j1],
                                               kstar[j1], mass0[j2], mass[j2], massc[j2], aj[j2],
                                               jspin[j2], kstar[j2], zcnsts, ecc, sep, jorb, coel, z, kick)

                        (mass0[j1], mass[j1], massc[j1], aj[j1], jspin[j1],
                         kstar[j1], mass0[j2], mass[j2], massc[j2], aj[j2],
                         jspin[j2], kstar[j2], ecc, sep, jorb, coel, z) = comenv_return

                        jp = min(80, jp + 1)
                        output.bpp[jp, 1] = tphys
                        output.bpp[jp, 2] = mass[1]
                        if kstar[1] == 15:
                            output.bpp[jp, 2] = mass0[1]
                        output.bpp[jp, 3] = mass[2]
                        if kstar[2] == 15:
                            output.bpp[jp, 3] = mass0[2]
                        output.bpp[jp, 4] = float(kstar[1])
                        output.bpp[jp, 5] = float(kstar[2])
                        output.bpp[jp, 6] = sep
                        output.bpp[jp, 7] = ecc
                        output.bpp[jp, 8] = rad[1] / rol[1]
                        output.bpp[jp, 9] = rad[2] / rol[2]
                        output.bpp[jp, 10] = 7.0
                        epoch[j1] = tphys - aj[j1]
                        if coel:
                            com = True
                            # goto.one_three_five
                            flag135 = True
                            break
                        epoch[j2] = tphys - aj[j2]
                        if ecc > 1.0:
                            if kstar[1] >= 13:
                                rc = corerd(kstar[1], mass[1], mass[1], zcnsts.zpars[2])
                                ospin[1] = jspin[1] / (k3 * rc * rc * mass[1])
                            if kstar[2] >= 13:
                                rc = corerd(kstar[2], mass[2], mass[2], zcnsts.zpars[2])
                                ospin[2] = jspin[2] / (k3 * rc * rc * mass[2])
                            # goto.one_three_five
                            flag135 = True
                            break
                        # Next step should be made without changing the time.
                        dm1 = m1ce - mass[j1]
                        dm2 = mass[j2] - m2ce
                        dm22 = dm2
                        dtm = 0.0
                        # Reset orbital parameters as separation may have changed.
                        tb = (sep / aursun) * np.sqrt(sep / (aursun * (mass[1] + mass[2])))
                        oorb = 2 * np.pi / tb
                    # donor星为白矮星
                    elif 10 <= kstar[j1] <= 12 and q[j1] > 0.628:
                        # Dynamic transfer from a white dwarf.  Secondary will have KW > 9.
                        taum = np.sqrt(tkh[j1] * tdyn)
                        dm1 = mass[j1]
                        if eddfac < 10.0:
                            dm2 = min(dme * taum / tb, dm1)
                            if dm2 < dm1:
                                supedd = True
                        else:
                            dm2 = dm1
                        mass[j2] = mass[j2] + dm2
                        if kstar[j1] == 10 and kstar[j2] == 10:
                            kstar[j2] = 15
                            mass[j2] = 0.0
                        elif kstar[j1] == 10 or kstar[j2] == 10:
                            kst = 9
                            if (kstar[j2] == 10):
                                massc[j2] = dm2
                            (massc[j2], mass[j2], kst, mass0[j2], aj[j2]) = gntage(
                                massc[j2], mass[j2], kst, zcnsts, mass0[j2], aj[j2])
                            kstar[j2] = kst
                            epoch[j2] = tphys - aj[j2]
                        elif kstar[j2] <= 12:
                            mass0[j2] = mass[j2]
                            if kstar[j1] == 12 and kstar[j2] == 11:
                                kstar[j2] = 12
                        kstar[j1] = 15
                        mass[j1] = 0.0
                        if kstar[j2] <= 11 and mass[j2] > mch:
                            kstar[j2] = 15
                            mass[j2] = 0.0
                        coel = True
                        # goto.one_three_five
                        flag135 = True
                        break
                    # donor星为中子星
                    elif kstar[j1] == 13:
                        # Gamma ray burster?
                        dm1 = mass[j1]
                        mass[j1] = 0.0
                        kstar[j1] = 15
                        dm2 = dm1
                        mass[j2] = mass[j2] + dm2
                        kstar[j2] = 14
                        coel = True
                        # goto.one_three_five
                        flag135 = True
                        break
                    # donor星为黑洞
                    elif kstar[j1] == 14:
                        # Both stars are black holes.  Let them merge quietly.
                        dm1 = mass[j1]
                        mass[j1] = 0.0
                        kstar[j1] = 15
                        dm2 = dm1
                        mass[j2] = mass[j2] + dm2
                        coel = True
                        # goto.one_three_five
                        flag135 = True
                        break
                    else:
                        dm1 = 3.0e-6 * tb * (np.log(rad[j1] / rol[j1]) ** 3) * min(mass[j1], 5.0) ** 2
                        if kstar[j1] == 2:
                            mew = (mass[j1] - massc[j1]) / mass[j1]
                            dm1 = max(mew, 0.01) * dm1
                        elif kstar[j1] >= 10:
                            dm1 = dm1 * 1.0e3 * mass[j1] / max(rad[j1], 1.0e-4)
                        kst = kstar[j2]
                        if 2 <= kstar[j1] <= 9 and kstar[j1] != 7:
                            dm1 = min(dm1, mass[j1] * tb / tkh[j1])
                        elif rad[j1] > 10.0 * rol[j1] or (kstar[j1] <= 1 and kstar[j2] <= 1 and q[j1] > qc):
                            m1ce = mass[j1]
                            m2ce = mass[j2]
                            (mass0, mass, aj, kstar) = mix(mass0, mass, aj, kstar, zcnsts)
                            dm1 = m1ce - mass[j1]
                            dm2 = mass[j2] - m2ce
                            dtm = 0.0
                            epoch[1] = tphys - aj[1]
                            coel = True
                            # goto.one_three_five
                            flag135 = True
                            break
                        else:
                            dm1 = min(dm1, mass[j1] * tb / tdyn)
                        vorb2 = acc1 * (mass[1] + mass[2]) / sep
                        ivsqm = 1.0 / np.sqrt(1.0 - ecc * ecc)
                        for k in range(1, 3):
                            if neta > tiny:
                                rlperi = rol[k] * (1.0 - ecc)
                                dmr[k] = mlwind(kstar[k], lumin[k], radx[k], mass[k], massc[k], rlperi, z)
                                vwind2 = 2.0 * beta_wind * acc1 * mass[k] / radx[k]
                                omv2 = (1.0 + vorb2 / vwind2) ** (3.0 / 2.0)
                                dmt[3 - k] = ivsqm * alpha_wind * dmr[k] * ((acc1 * mass[3 - k] / vwind2) ** 2) / (
                                        2.0 * sep * sep * omv2)
                                dmt[3 - k] = min(dmt[3 - k], dmr[k])
                            else:
                                dmr[k] = 0.0
                                dmt[3 - k] = 0.0
                        for k in range(1, 3):
                            dms[k] = (dmr[k] - dmt[k]) * tb
                        km = min(2.0 * km0, 5.0e-3 / max(abs(dm1 + dms[j1]) / mass[j1], dms[j2] / mass[j2]))
                        km0 = km
                        dt = km * tb
                        dtm = dt / 1.0e6
                        if iter <= 1000:
                            dtm = min(dtm, dtmi[1], dtmi[2])
                        dtm = min(dtm, tsave - tphys)
                        dt = dtm * 1.0e6
                        km = dt / tb
                        taum = mass[j2] / dm1 * tb
                        if kstar[j2] <= 2 or kstar[j2] == 4:
                            ospbru2 = 2 * np.pi * np.sqrt(mass[2] * aursun ** 3 / rad[2] ** 3)
                            kkk = 1.0 - ospin[2] / ospbru2
                            dm2 = min(1.00, kkk) * dm1
                        elif 7 <= kstar[j2] <= 9:
                            if kstar[j1] >= 7:
                                dm2 = min(1.0, 10.0 * taum / tkh[j2]) * dm1
                            else:
                                dm2 = dm1
                                dmchk = dm2 - 1.050 * dms[j2]
                                if dmchk > 0.0 and dm2 / mass[j2] > 1.0e-04:
                                    kst = min(6, 2 * kstar[j2] - 10)
                                    if kst == 4:
                                        aj[j2] = aj[j2] / tms[j2]
                                        mcx = mass[j2]
                                    else:
                                        mcx = massc[j2]
                                    mt2 = mass[j2] + km * (dm2 - dms[j2])
                                    (mcx, mt2, kst, mass0[j2], aj[j2]) = gntage(mcx, mt2, kst, zcnsts, mass0[j2],
                                                                                       aj[j2])
                                    epoch[j2] = tphys + dtm - aj[j2]
                                    jp = min(80, jp + 1)
                                    output.bpp[jp, 1] = tphys
                                    output.bpp[jp, 2] = mass[j1]
                                    output.bpp[jp, 3] = mt2
                                    output.bpp[jp, 4] = float(kstar[j1])
                                    output.bpp[jp, 5] = float(kst)
                                    output.bpp[jp, 6] = sep
                                    output.bpp[jp, 7] = ecc
                                    output.bpp[jp, 8] = rad[1] / rol[1]
                                    output.bpp[jp, 9] = rad[2] / rol[2]
                                    output.bpp[jp, 10] = 8.0
                                    if j1 == 2:
                                        output.bpp[jp, 2] = mt2
                                        output.bpp[jp, 3] = mass[j1]
                                        output.bpp[jp, 4] = float(kst)
                                        output.bpp[jp, 5] = float(kstar[j1])
                        # 白矮星吸积富氢物质
                        elif kstar[j1] <= 6 and 10 <= kstar[j2] <= 12:
                            # 持续吸积直到新星爆发, 同时吹散大部分的吸积物质
                            if dm1 / tb < 1.03e-07:
                                novae = True
                                dm2 = min(dm1, dme)
                                if dm2 < dm1:
                                    supedd = True
                                # 白矮星保留很少的吸积物质(Hurley 2002 eq.66)
                                dm22 = epsnov * dm2
                            # 在白矮星表面稳定燃烧(X射线源)
                            elif 1.03e-07 <= dm1 / tb < 2.71e-07:
                                dm2 = dm1
                            # 白矮星出现新的巨星包层
                            else:
                                dm2 = dm1
                                if (kstar[j2] == 10 and mass[j2] < 0.05) or (kstar[j2] >= 11 and mass[j2] < 0.5):
                                    kst = kstar[j2]
                                else:
                                    kst = min(6, 3 * kstar[j2] - 27)
                                    mt2 = mass[j2] + km * (dm2 - dms[j2])
                                    (massc[j2], mt2, kst, mass0[j2], aj[j2]) = gntage(
                                        massc[j2], mt2, kst, zcnsts, mass0[j2], aj[j2])
                                    epoch[j2] = tphys + dtm - aj[j2]
                                    jp = min(80, jp + 1)
                                    output.bpp[jp, 1] = tphys
                                    output.bpp[jp, 2] = mass[j1]
                                    output.bpp[jp, 3] = mt2
                                    output.bpp[jp, 4] = float(kstar[j1])
                                    output.bpp[jp, 5] = float(kst)
                                    output.bpp[jp, 6] = sep
                                    output.bpp[jp, 7] = ecc
                                    output.bpp[jp, 8] = rad[1] / rol[1]
                                    output.bpp[jp, 9] = rad[2] / rol[2]
                                    output.bpp[jp, 10] = 8.0
                                    if j1 == 2:
                                        output.bpp[jp, 2] = mt2
                                        output.bpp[jp, 3] = mass[j1]
                                        output.bpp[jp, 4] = float(kst)
                                        output.bpp[jp, 5] = float(kstar[j1])
                        elif kstar[j2] >= 10:
                            dm2 = min(dm1, dme)
                            if dm2 < dm1:
                                supedd = True
                        # 巨星包层可以吸收所有的吸积物质
                        else:
                            dm2 = dm1
                        if not novae:
                            dm22 = dm2
                        if 10 <= kst <= 12:
                            mt2 = mass[j2] + km * (dm22 - dms[j2])
                            # 氦白矮星吸积超过0.7太阳质量，
                            if kstar[j1] <= 10 and kst == 10 and mt2 >= 0.7:
                                mass[j1] = mass[j1] - km * (dm1 + dms[j1])
                                mass[j2] = 0.0
                                kstar[j2] = 15
                                # goto.one_three_five
                                flag135 = True
                                break
                            elif kstar[j1] <= 10 and kst >= 11:
                                if (mt2 - mass0[j2]) >= 0.15:
                                    # 碳氧白矮星吸积超过一定的氦，触发Ia超新星爆炸
                                    if kst == 11:
                                        mass[j1] = mass[j1] - km * (dm1 + dms[j1])
                                        mass[j2] = 0.0
                                        kstar[j2] = 15
                                        # goto.one_three_five
                                        flag135 = True
                                        break
                                    mass0[j2] = mt2
                            else:
                                mass0[j2] = mt2
                            # He白矮星和CO白矮星的质量超过Mch, 发生超新星爆炸
                            if kst == 10 or kst == 11:
                                if mt2 >= mch:
                                    dm1 = mch - mass[j2] + km * dms[j2]
                                    mass[j1] = mass[j1] - dm1 - km * dms[j1]
                                    mass[j2] = 0.0
                                    kstar[j2] = 15
                                    # goto.one_three_five
                                    flag135 = True
                                    break
                        dm1 = km * dm1
                        dm2 = km * dm2
                        dm22 = km * dm22
                        dme = km * dme
                        djorb = ((dmr[1] + q[1] * dmt[1]) * mass[2] * mass[2] +
                                 (dmr[2] + q[2] * dmt[2]) * mass[1] * mass[1]) / (mass[1] + mass[2]) ** 2
                        djorb = djorb * dt
                        if supedd or novae or gamma < -1.50:
                            djorb = djorb + (dm1 - dm22) * mass[j1] * mass[j1] / (mass[1] + mass[2]) ** 2
                        elif gamma >= 0.0:
                            djorb = djorb + gamma * (dm1 - dm2)
                        else:
                            djorb = djorb + (dm1 - dm2) * mass[j2] * mass[j2] / (mass[1] + mass[2]) ** 2
                        ecc2 = ecc * ecc
                        omecc2 = 1.0 - ecc2
                        sqome2 = np.sqrt(omecc2)
                        djorb = djorb * sep * sep * sqome2 * oorb
                        delet = 0.0
                        if sep <= 10.0:
                            djgr = 8.315e-10 * mass[1] * mass[2] * (mass[1] + mass[2]) / (sep * sep * sep * sep)
                            f1 = (19.0 / 6.0) + (121.0 / 96.0) * ecc2
                            sqome5 = sqome2 ** 5
                            delet1 = djgr * ecc * f1 / sqome5
                            djgr = djgr * jorb * (1.0 + 0.8750 * ecc2) / sqome5
                            djorb = djorb + djgr * dt
                            delet = delet + delet1 * dt
                        for k in range(1, 3):
                            dms[k] = km * dms[k]
                            if kstar[k] < 10:
                                dms[k] = min(dms[k], mass[k] - massc[k])
                            djspint[k] = (2.0 / 3.0) * (dmr[k] * radx[k] * radx[k] * ospin[k] -
                                                        xi * dmt[k] * radx[3 - k] * radx[3 - k] * ospin[3 - k])
                            djspint[k] = djspint[k] * dt
                            if mass[k] > 0.350 and kstar[k] < 10:
                                djmb = 5.83e-16 * menv[k] * (rad[k] * ospin[k]) ** 3 / mass[k]
                                djspint[k] = djspint[k] + djmb * dt
                        djt = dm1 * radx[j1] * radx[j1] * ospin[j1]
                        djspint[j1] = djspint[j1] + djt
                        djorb = djorb - djt
                        if disk:
                            djt = dm2 * 2 * np.pi * aursun * np.sqrt(aursun * mass[j2] * radx[j2])
                            djspint[j2] = djspint[j2] - djt
                            djorb = djorb + djt
                        else:
                            rdisk = 1.70 * rmin
                            djt = dm2 * 2 * np.pi * aursun * np.sqrt(aursun * mass[j2] * rdisk)
                            djspint[j2] = djspint[j2] - djt
                            djorb = djorb + djt
                        djtx[2] = djt
                        if novae:
                            djt = (dm2 - dm22) * radx[j2] * radx[j2] * ospin[j2]
                            djspint[j2] = djspint[j2] + djt
                            djtx[2] = djtx[2] - djt
                        for k in range(1, 3):
                            dspint[k] = 0.0
                            if ((kstar[k] <= 9 and rad[k] >= 0.010 * rol[k]) or (
                                    kstar[k] >= 10 and k == j1)) and tflag > 0:
                                raa2 = (radx[k] / sep) ** 2
                                raa6 = raa2 ** 3
                                f5 = 1.0 + ecc2 * (3.0 + ecc2 * 0.3750)
                                f4 = 1.0 + ecc2 * (1.50 + ecc2 * 0.1250)
                                f3 = 1.0 + ecc2 * (3.750 + ecc2 * (1.8750 + ecc2 * 7.8125e-02))
                                f2 = 1.0 + ecc2 * (7.50 + ecc2 * (5.6250 + ecc2 * 0.31250))
                                f1 = 1.0 + ecc2 * (15.50 + ecc2 * (31.8750 + ecc2 * (11.56250 + ecc2 * 0.3906250)))
                                if (kstar[k] == 1 and mass[k] >= 1.250) or kstar[k] == 4 or kstar[k] == 7:
                                    tc = 1.592e-09 * (mass[k] ** 2.840)
                                    f = 1.9782e+04 * np.sqrt((mass[k] * radx[k] * radx[k]) / sep ** 5) * tc * (
                                            1.0 + q[3 - k]) ** (
                                                5.0 / 6.0)
                                    tcqr = f * q[3 - k] * raa6
                                    rg2 = k2str[k]
                                elif kstar[k] <= 9:
                                    renv[k] = min(renv[k], radx[k] - radc[k])
                                    renv[k] = max(renv[k], 1.0e-10)
                                    tc = mr23yr * (
                                            menv[k] * renv[k] * (radx[k] - 0.50 * renv[k]) / (3.0 * lumin[k])) ** (
                                                 1.0 / 3.0)
                                    ttid = 2 * np.pi / (1.0e-10 + abs(oorb - ospin[k]))
                                    f = min(1.0, (ttid / (2.0 * tc) ** 2))
                                    tcqr = 2.0 * f * q[3 - k] * raa6 * menv[k] / (21.0 * tc * mass[k])
                                    rg2 = (k2str[k] * (mass[k] - massc[k])) / mass[k]
                                else:
                                    f = 7.33e-09 * (lumin[k] / mass[k]) ** (5.0 / 7.0)
                                    tcqr = f * q[3 - k] * q[3 - k] * raa2 * raa2 / (1.0 + q[3 - k])
                                    rg2 = k3
                                sqome3 = sqome2 ** 3
                                delet1 = 27.0 * tcqr * (1.0 + q[3 - k]) * raa2 * (ecc / sqome2 ** 13) * (
                                        f3 - (11.0 / 18.0) * sqome3 * f4 * ospin[k] / oorb)
                                tcirc = ecc / (abs(delet1) + 1.0e-20)
                                delet = delet + delet1 * dt
                                dspint[k] = (3.0 * q[3 - k] * tcqr / (rg2 * omecc2 ** 6)) * (
                                        f2 * oorb - sqome3 * f5 * ospin[k])
                                eqspin = oorb * f2 / (sqome3 * f5)
                                if dt > 0.0:
                                    if dspint[k] >= 0.0:
                                        dspint[k] = min(dt * dspint[k], eqspin - ospin[k]) / dt
                                    else:
                                        dspint[k] = max(dt * dspint[k], eqspin - ospin[k]) / dt
                                djt = (k2str[k] * (mass[k] - massc[k]) * radx[k] * radx[k] + k3 * massc[k] * radc[k] *
                                       radc[k]
                                       ) * dspint[k]
                                djorb = djorb + djt * dt
                                djspint[k] = djspint[k] - djt * dt
                            jspin[k] = max(1.0e-10, jspin[k] - djspint[k])
                            ospbru = 2 * np.pi * np.sqrt(mass[k] * aursun ** 3 / radx[k] ** 3)
                            jspbru = (k2str[k] * (mass[k] - massc[k]) * radx[k] * radx[k] + k3 * massc[k] * radc[k] *
                                      radc[k]) * ospbru
                            if jspin[k] > jspbru:
                                mew = 1.0
                                if djtx[2] > 0.0:
                                    mew = min(mew, (jspin[k] - jspbru) / djtx[2])
                                djorb = djorb - (jspin[k] - jspbru)
                                jspin[k] = jspbru
                        kstar[j2] = kst
                        mass[j1] = mass[j1] - dm1 - dms[j1]
                        if kstar[j1] <= 1 or kstar[j1] == 7:
                            mass0[j1] = mass[j1]
                        mass[j2] = mass[j2] + dm22 - dms[j2]
                        if kstar[j2] <= 1 or kstar[j2] == 7:
                            mass0[j2] = mass[j2]
                        if kstar[j1] == 2 and mass0[j1] <= zcnsts.zpars[3]:
                            m0 = mass0[j1]
                            mass0[j1] = mass[j1]
                            (tmsnew, tn, tscls, lums, GB) = star(kstar[j1], mass0[j1], mass[j1], zcnsts)
                            if GB[9] < massc[j1]:
                                mass0[j1] = m0
                        if kstar[j2] == 2 and mass0[j2] <= zcnsts.zpars[3]:
                            m0 = mass0[j2]
                            mass0[j2] = mass[j2]
                            (tmsnew, tn, tscls, lums, GB) = star(kstar[j2], mass0[j2], mass[j2], zcnsts)
                            if GB[9] < massc[j2]:
                                mass0[j2] = m0
                        ecc = ecc - delet
                        ecc = max(ecc, 0.0)
                        if ecc < 1.0e-10:
                            ecc = 0.0
                        if ecc >= 1.0:
                            # goto.one_three_five
                            flag135 = True
                            break
                        jorb = max(1.0, jorb - djorb)
                        sep = (mass[1] + mass[2]) * jorb * jorb / (
                                (mass[1] * mass[2] * 2 * np.pi) ** 2 * aursun ** 3 * (1.0 - ecc * ecc))
                        tb = (sep / aursun) * np.sqrt(sep / (aursun * (mass[1] + mass[2])))
                        oorb = 2 * np.pi / tb

                    if kstar[j1] <= 2 or kstar[j1] == 7:
                        (tmsnew, tn, tscls, lums, GB) = star(kstar[j1], mass0[j1], mass[j1], zcnsts)
                        if kstar[j1] == 2:
                            aj[j1] = tmsnew + (tscls[1] - tmsnew) * (aj[j1] - tms[j1]) / (tbgb[j1] - tms[j1])
                        else:
                            aj[j1] = tmsnew / tms[j1] * aj[j1]
                        epoch[j1] = tphys - aj[j1]

                    if kstar[j2] <= 2 or kstar[j2] == 7:
                        (tmsnew, tn, tscls, lums, GB) = star(kstar[j2], mass0[j2], mass[j2], zcnsts)
                        if kstar[j2] == 2:
                            aj[j2] = tmsnew + (tscls[1] - tmsnew) * (aj[j2] - tms[j2]) / (tbgb[j2] - tms[j2])
                        elif (mass[j2] < 0.350 or mass[j2] > 1.250) and kstar[j2] != 7:
                            aj[j2] = tmsnew / tms[j2] * aj[j2] * (mass[j2] - dm22) / mass[j2]
                        else:
                            aj[j2] = tmsnew / tms[j2] * aj[j2]
                        epoch[j2] = tphys - aj[j2]

                    tphys = tphys + dtm
                    for k in range(1, 3):
                        age = tphys - epoch[k]
                        m0 = mass0[k]
                        mt = mass[k]
                        mc = massc[k]
                        if mt > 100.0:
                            # goto.one_four_zero
                            flag140 = True
                            break
                        kw = kstar[k]
                        (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
                        (m0, age, mt, tm, tn, tscls, lums, GB, rm, lum, kw, mc, rc, me, re, k2) = hrdiag(
                            m0, age, mt, tm, tn, tscls, lums, GB, zcnsts, rm, lum, kw, mc, rc, me, re, k2, kick)

                        if kw != kstar[k] and kstar[k] <= 12 and (kw == 13 or kw == 14):
                            dms[k] = mass[k] - mt
                            (kw, mass[k], mt, mass[3 - k], ecc, sep, jorb, vs) = vkick(
                                kw, mass[k], mt, mass[3 - k], ecc, sep, jorb, vs, kick)
                            if ecc > 1.0:
                                kstar[k] = kw
                                mass[k] = mt
                                epoch[k] = tphys - age
                                # goto.one_three_five
                                flag135 = True
                                break
                            tb = (sep / aursun) * np.sqrt(sep / (aursun * (mt + mass[3 - k])))
                            oorb = 2 * np.pi / tb

                        if kw != kstar[k]:
                            change = True
                            if (kw == 13 or kw == 14) and kstar[k] <= 12:
                                snova = True
                            mass[k] = mt
                            if kw == 15:
                                kstar[k] = kw
                                # goto.one_three_five
                                flag135 = True
                                break
                            mass0[k] = m0
                            epoch[k] = tphys - age

                        if kw <= 9:
                            (dt, dtr) = timestep(kw, age, tm, tn, tscls, dt)
                            dtmi[k] = min(dt, dtr)
                            dtmi[k] = max(1.0e-7, dtmi[k])
                        else:
                            dtmi[k] = 1.0e10

                        aj[k] = age
                        kstar[k] = kw
                        rad[k] = rm
                        radx[k] = rm
                        lumin[k] = lum
                        massc[k] = mc
                        radc[k] = rc
                        menv[k] = me
                        renv[k] = re
                        k2str[k] = k2
                        tms[k] = tm
                        tbgb[k] = tscls[1]

                        if kw <= 1 and tm < tphys and not bss:
                            bss = True
                            jp = min(80, jp + 1)
                            output.bpp[jp, 1] = tphys
                            output.bpp[jp, 2] = mass[1]
                            output.bpp[jp, 3] = mass[2]
                            output.bpp[jp, 4] = float(kstar[1])
                            output.bpp[jp, 5] = float(kstar[2])
                            output.bpp[jp, 6] = sep
                            output.bpp[jp, 7] = ecc
                            output.bpp[jp, 8] = rad[1] / rol[1]
                            output.bpp[jp, 9] = rad[2] / rol[2]
                            output.bpp[jp, 10] = 14.0

                    if flag135 or flag140:
                        break

                    for k in range(1, 3):
                        q[k] = mass[k] / mass[3 - k]
                        rol[k] = rochelobe(q[k]) * sep

                    if rad[j1] > rol[j1]:
                        radx[j1] = max(radc[j1], rol[j1])

                    for k in range(1, 3):
                        ospin[k] = jspin[k] / (
                                k2str[k] * (mass[k] - massc[k]) * radx[k] * radx[k] + k3 * massc[k] * radc[k] *
                                radc[k])

                    if (isave and tphys >= tsave) or iplot:
                        ip = ip + 1
                        output.bcm[ip, 1] = tphys
                        output.bcm[ip, 2] = float(kstar[1])
                        output.bcm[ip, 3] = mass0[1]
                        output.bcm[ip, 4] = mass[1]
                        output.bcm[ip, 5] = np.log10(lumin[1])
                        output.bcm[ip, 6] = np.log10(rad[1])
                        teff1 = 1000.0 * ((1130.0 * lumin[1] / (rad[1] ** 2.0)) ** (1.0 / 4.0))
                        output.bcm[ip, 7] = np.log10(teff1)
                        output.bcm[ip, 8] = massc[1]
                        output.bcm[ip, 9] = radc[1]
                        output.bcm[ip, 10] = menv[1]
                        output.bcm[ip, 11] = renv[1]
                        output.bcm[ip, 12] = epoch[1]
                        output.bcm[ip, 13] = ospin[1]
                        output.bcm[ip, 15] = rad[1] / rol[1]
                        output.bcm[ip, 16] = float(kstar[2])
                        output.bcm[ip, 17] = mass0[2]
                        output.bcm[ip, 18] = mass[2]
                        output.bcm[ip, 19] = np.log10(lumin[2])
                        output.bcm[ip, 20] = np.log10(rad[2])
                        teff2 = 1000.0 * ((1130.0 * lumin[2] / (rad[2] ** 2.0)) ** (1.0 / 4.0))
                        output.bcm[ip, 21] = np.log10(teff2)
                        output.bcm[ip, 22] = massc[2]
                        output.bcm[ip, 23] = radc[2]
                        output.bcm[ip, 24] = menv[2]
                        output.bcm[ip, 25] = renv[2]
                        output.bcm[ip, 26] = epoch[2]
                        output.bcm[ip, 27] = ospin[2]
                        output.bcm[ip, 29] = rad[2] / rol[2]
                        output.bcm[ip, 30] = tb
                        output.bcm[ip, 31] = sep
                        output.bcm[ip, 32] = ecc
                        dt = max(dtm, 1.0e-12) * 1.0e6
                        if j1 == 1:
                            output.bcm[ip, 14] = (-1.0 * dm1 - dms[1]) / dt
                            output.bcm[ip, 28] = (dm2 - dms[2]) / dt
                        else:
                            output.bcm[ip, 14] = (dm2 - dms[1]) / dt
                            output.bcm[ip, 28] = (-1.0 * dm1 - dms[2]) / dt
                        if isave:
                            tsave = tsave + dtp

                    if tphys >= tphysf:
                        # goto.one_four_zero
                        flag140 = True
                        break

                    if change:
                        change = False
                        jp = min(80, jp + 1)
                        output.bpp[jp, 1] = tphys
                        output.bpp[jp, 2] = mass[1]
                        output.bpp[jp, 3] = mass[2]
                        output.bpp[jp, 4] = float(kstar[1])
                        output.bpp[jp, 5] = float(kstar[2])
                        output.bpp[jp, 6] = sep
                        output.bpp[jp, 7] = ecc
                        output.bpp[jp, 8] = rad[1] / rol[1]
                        output.bpp[jp, 9] = rad[2] / rol[2]
                        output.bpp[jp, 10] = 2.0

                    if rad[j1] > rol[j1] and not snova:
                        if rad[j2] > rol[j2]:
                            # goto.one_three_zero
                            flag130 = True
                            break
                        iter = iter + 1
                        # goto.eight
                        continue
                    else:
                        jp = min(80, jp + 1)
                        output.bpp[jp, 1] = tphys
                        output.bpp[jp, 2] = mass[1]
                        output.bpp[jp, 3] = mass[2]
                        output.bpp[jp, 4] = float(kstar[1])
                        output.bpp[jp, 5] = float(kstar[2])
                        output.bpp[jp, 6] = sep
                        output.bpp[jp, 7] = ecc
                        output.bpp[jp, 8] = rad[1] / rol[1]
                        output.bpp[jp, 9] = rad[2] / rol[2]
                        output.bpp[jp, 10] = 4.0
                        dtm = 0.0

                    # goto.four
                    break

                if flag130 or flag135 or flag140:
                    break

            # label.one_three_zero
            if flag130:
                flag130 = False  # 重置flag130，以防下次直接进入循环
                # Contact system.
                coel = True
                m1ce = mass[j1]
                m2ce = mass[j2]
                rrl1 = min(999.9990, rad[1] / rol[1])
                rrl2 = min(999.9990, rad[2] / rol[2])
                jp = min(80, jp + 1)
                output.bpp[jp, 1] = tphys
                output.bpp[jp, 2] = mass[1]
                output.bpp[jp, 3] = mass[2]
                output.bpp[jp, 4] = float(kstar[1])
                output.bpp[jp, 5] = float(kstar[2])
                output.bpp[jp, 6] = sep
                output.bpp[jp, 7] = ecc
                output.bpp[jp, 8] = rrl1
                output.bpp[jp, 9] = rrl2
                output.bpp[jp, 10] = 5.0

                if 2 <= kstar[j1] <= 9 and kstar[j1] != 7:
                    comenv_return = comenv(mass0[j1], mass[j1], massc[j1], aj[j1], jspin[j1],
                                           kstar[j1], mass0[j2], mass[j2], massc[j2], aj[j2],
                                           jspin[j2], kstar[j2], zcnsts, ecc, sep, jorb, coel, z, kick)
                    (mass0[j1], mass[j1], massc[j1], aj[j1], jspin[j1],
                     kstar[j1], mass0[j2], mass[j2], massc[j2], aj[j2],
                     jspin[j2], kstar[j2], ecc, sep, jorb, coel, z) = comenv_return

                    com = True
                elif 2 <= kstar[j2] <= 9 and kstar[j2] != 7:
                    comenv_return = comenv(mass0[j2], mass[j2], massc[j2], aj[j2], jspin[j2],
                                           kstar[j2], mass0[j1], mass[j1], massc[j1], aj[j1],
                                           jspin[j1], kstar[j1], zcnsts, ecc, sep, jorb, coel, z, kick)
                    (mass0[j2], mass[j2], massc[j2], aj[j2], jspin[j2],
                     kstar[j2], mass0[j1], mass[j1], massc[j1], aj[j1],
                     jspin[j1], kstar[j1], ecc, sep, jorb, coel, z) = comenv_return

                    com = True
                else:
                    (mass0, mass, aj, kstar) = mix(mass0, mass, aj, kstar, zcnsts)

                if com:
                    jp = min(80, jp + 1)
                    output.bpp[jp, 1] = tphys
                    output.bpp[jp, 2] = mass[1]
                    if kstar[1] == 15:
                        output.bpp[jp, 2] = mass0[1]
                    output.bpp[jp, 3] = mass[2]
                    if kstar[2] == 15:
                        output.bpp[jp, 3] = mass0[2]
                    output.bpp[jp, 4] = float(kstar[1])
                    output.bpp[jp, 5] = float(kstar[2])
                    output.bpp[jp, 6] = sep
                    output.bpp[jp, 7] = ecc
                    rrl1 = min(rrl1, 0.990)
                    rrl2 = min(rrl2, 0.990)
                    output.bpp[jp, 8] = rrl1
                    output.bpp[jp, 9] = rrl2
                    output.bpp[jp, 10] = 7.0

                epoch[1] = tphys - aj[1]
                epoch[2] = tphys - aj[2]
                if not coel:
                    if ecc > 1.0:
                        if kstar[1] >= 13:
                            rc = corerd(kstar[1], mass[1], mass[1], zcnsts.zpars[2])
                            ospin[1] = jspin[1] / (k3 * rc * rc * mass[1])
                        if kstar[2] >= 13:
                            rc = corerd(kstar[2], mass[2], mass[2], zcnsts.zpars[2])
                            ospin[2] = jspin[2] / (k3 * rc * rc * mass[2])
                        # goto.one_three_five
                        flag135 = True
                        break
                    dtm = 0.0
                    tb = (sep / aursun) * np.sqrt(sep / (aursun * (mass[1] + mass[2])))
                    oorb = 2 * np.pi / tb
                    # goto.four
                    continue
            if not flag140:
                flag135 = True
            break

        # label.one_three_five
        if flag135:
            flag135 = False  # 重置flag135
            sgl = True
            if kstar[1] != 15 or kstar[2] != 15:
                if com:
                    com = False
                else:
                    jp = min(80, jp + 1)
                    output.bpp[jp, 1] = tphys
                    output.bpp[jp, 2] = mass[1]
                    if kstar[1] == 15:
                        output.bpp[jp, 2] = mass0[1]
                    output.bpp[jp, 3] = mass[2]
                    if kstar[2] == 15:
                        output.bpp[jp, 3] = mass0[2]
                    output.bpp[jp, 4] = float(kstar[1])
                    output.bpp[jp, 5] = float(kstar[2])
                    output.bpp[jp, 6] = 0
                    output.bpp[jp, 7] = 0
                    output.bpp[jp, 8] = 0
                    output.bpp[jp, 9] = ngtv
                    if coel:
                        output.bpp[jp, 10] = 6.0
                    elif ecc > 1.0:
                        output.bpp[jp, 6] = sep
                        output.bpp[jp, 7] = ecc
                        output.bpp[jp, 9] = ngtv2
                        output.bpp[jp, 10] = 11.0
                    else:
                        output.bpp[jp, 10] = 9.0
                if kstar[2] == 15:
                    kmax = 1
                    rol[2] = -1.0 * rad[2]
                    dtmi[2] = tphysf
                elif kstar[1] == 15:
                    kmin = 2
                    rol[1] = -1.0 * rad[1]
                    dtmi[1] = tphysf
                ecc = -1.0
                sep = 0.0
                dtm = 0.0
                coel = False
                # goto.four
                continue
        break

    # label.one_four_zero
    if com:
        com = False
    else:
        jp = min(80, jp + 1)
        output.bpp[jp, 1] = tphys
        output.bpp[jp, 2] = mass[1]
        if kstar[1] == 15 and output.bpp[jp - 1, 4] < 15.0:
            output.bpp[jp, 2] = mass0[1]
        output.bpp[jp, 3] = mass[2]
        if kstar[2] == 15 and output.bpp[jp - 1, 5] < 15.0:
            output.bpp[jp, 3] = mass0[2]
        output.bpp[jp, 4] = float(kstar[1])
        output.bpp[jp, 5] = float(kstar[2])
        output.bpp[jp, 6] = 0
        output.bpp[jp, 7] = 0
        output.bpp[jp, 8] = 0
        if coel:
            output.bpp[jp, 9] = ngtv
            output.bpp[jp, 10] = 6.0
        elif kstar[1] == 15 and kstar[2] == 15:
            output.bpp[jp, 9] = ngtv2
            output.bpp[jp, 10] = 9.0
        else:
            output.bpp[jp, 6] = sep
            output.bpp[jp, 7] = ecc
            output.bpp[jp, 8] = rad[1] / rol[1]
            output.bpp[jp, 9] = rad[2] / rol[2]
            output.bpp[jp, 10] = 10.0

    if (isave and tphys >= tsave) or iplot:
        ip = ip + 1
        output.bcm[ip, 1] = tphys
        output.bcm[ip, 2] = float(kstar[1])
        output.bcm[ip, 3] = mass0[1]
        output.bcm[ip, 4] = mass[1]
        output.bcm[ip, 5] = np.log10(lumin[1])
        output.bcm[ip, 6] = np.log10(rad[1])
        teff1 = 1000.0 * ((1130.0 * lumin[1] / (rad[1] ** 2.0)) ** (1.0 / 4.0))
        output.bcm[ip, 7] = np.log10(teff1)
        output.bcm[ip, 8] = massc[1]
        output.bcm[ip, 9] = radc[1]
        output.bcm[ip, 10] = menv[1]
        output.bcm[ip, 11] = renv[1]
        output.bcm[ip, 12] = epoch[1]
        output.bcm[ip, 13] = ospin[1]
        output.bcm[ip, 15] = rad[1] / rol[1]
        output.bcm[ip, 16] = float(kstar[2])
        output.bcm[ip, 17] = mass0[2]
        output.bcm[ip, 18] = mass[2]
        output.bcm[ip, 19] = np.log10(lumin[2])
        output.bcm[ip, 20] = np.log10(rad[2])
        teff2 = 1000.0 * ((1130.0 * lumin[2] / (rad[2] ** 2.0)) ** (1.0 / 4.0))
        output.bcm[ip, 21] = np.log10(teff2)
        output.bcm[ip, 22] = massc[2]
        output.bcm[ip, 23] = radc[2]
        output.bcm[ip, 24] = menv[2]
        output.bcm[ip, 25] = renv[2]
        output.bcm[ip, 26] = epoch[2]
        output.bcm[ip, 27] = ospin[2]
        output.bcm[ip, 29] = rad[2] / rol[2]
        output.bcm[ip, 30] = tb
        output.bcm[ip, 31] = sep
        output.bcm[ip, 32] = ecc
        dt = max(dtm, 1.0e-12) * 1.0e6
        if j1 == 1:
            output.bcm[ip, 14] = (-1.0 * dm1 - dms[1]) / dt
            output.bcm[ip, 28] = (dm2 - dms[2]) / dt
        else:
            output.bcm[ip, 14] = (dm2 - dms[1]) / dt
            output.bcm[ip, 28] = (-1.0 * dm1 - dms[2]) / dt
        if isave:
            tsave = tsave + dtp
        if tphysf <= 0.0:
            ip = ip + 1
            for k in range(1, 33):
                output.bcm[ip, k] = output.bcm[ip - 1, k]

    tphysf = tphys
    if sgl:
        if 0.0 <= ecc <= 1.0:
            ecc = -1.0
        tb = -1.0
    tb = tb * yeardy

    output.bcm[ip + 1, 1] = -1.0
    output.bpp[jp + 1, 1] = -1.0
