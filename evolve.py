import numpy as np
from utils import conditional_njit
import pysnooper
from mix import mix
from star import star
from hrdiag import hrdiag
from mlwind import mlwind
from zfuncs import vrotf
from utils import rochelobe, mb_judgment, magnetic_braking
from supernova import SN_kick
from timestep import timestep
from gntage import gntage
from comenv import comenv
from corerd import corerd
from const import neta, beta_wind, alpha_wind, acc1, mass_accretion_model, max_WD_mass
from const import ktype, xi, tflag, eddfac, mch, epsnov, gamma, yeardy, aursun, tiny
from const import mxns
from MT_stability import MT_stability_MS


# B I N A R Y
# ***********

# Roche lobe overflow.
# --------------------

# Developed by Jarrod Hurley, IOA, Cambridge.
# .........................................................

# Advice by Christopher Tout, Onno Pols & Sverre Aarseth.
# ++++++++++++++++++++++++++++++++++++++++++++++++++

# Adapted from Aarseth's code 21st September 1996.
# Fully revised on 27th November 1996 to remove vestiges of N-body code and incorporate corrections.
# Fully revised on 1st April 1998 to include new stellar evolution formulae and associated binary evolution changes.
# Fully revised on 4th July 1998 to include eccentricity, tidal circularization, wind accretion,
# velocity kicks for supernovae and all associated orbital momentum changes.
# Revised on 31st October 2000 to upgrade restrictions imposed on the timestep owing to magnetic braking
# and orbital angular momentum changes.

# See Tout et al., 1997, MNRAS, 291, 732 for a description of many of the
# processes in this code as well as the relevant references mentioned
# within the code.

# Reference for the stellar evolution formulae is Hurley, Pols & Tout, 2000, MNRAS, 315, 543 (SSE paper).
# Reference for the binary evolution algorithm is Hurley, Tout & Pols, 2002, MNRAS, 329, 897 (BSE paper).

# March 2001
# Changes since version 3, i.e. since production of Paper3:
# 1) The Eddington limit flag (on/off) has been replaced by an Eddington limit multiplicative factor (eddfac).
#    So if you want to neglect the Eddington limit you would set eddfac to a large value.
#
# 2) To determine whether material transferred during RLOF forms an accretion disk around the secondary or hits the
#    secondary in a direct stream we calculate a minimum radial distance, rmin, of the mass stream from the secondary.
#    This is taken from eq.(1) of Ulrich & Burger (1976, ApJ, 206, 509) which they fitted to the calculations of
#    Lubow & Shu (1974, ApJ, 198, 383). If rmin is less than the radius of the secondary then an accretion disk
#    is not formed. Note that the formula for rmin given by Ulrich & Burger is valid for all q whereas that given by
#    Nelemans et al. (2001, A&A, submitted) in their eq.(6) is only valid for q < 1 where they define
#    q = Mdonor/Maccretor, i.e. DD systems.
#
# 3) The changes to orbital and spin angular momentum owing to RLOF mass transfer have been improved,
#    and a new input option now exists. When mass is lost from the system during RLOF there are now
#    three choices as to how the orbital angular momentum is affected:
#    a. the lost material carries with it a fraction gamma of the orbital angular momentum, i.e.
#    dJorb = gamma*dm*a^2*omega_orb;
#    b. the material carries with it the specific angular momentum of the primary, i.e. dJorb = dm*a_1^2*omega_orb;
#    c. assume the material is lost from the system as if a wind from the secondary, i.e. dJorb = dm*a_2^2*omega_orb.
#    The parameter gamma is an input option. Choice c. is used if the mass transfer is super-Eddington or the system
#    is experiencing novae eruptions.
#    In all other cases choice a. is used if gamma > 0.0, b. if gamma = -1.0 and c. is used if gamma = -2.0.
#    The primary spin angular momentum is reduced by an amount dm1*r_1^2*omega_1 when an amount of mass dm1
#    is transferred from the primary. If the secondary accretes through a disk then its spin angular momentum
#    is altered by assuming that the material falls onto the star from the inner edge of a Keplerian disk
#    and that the system is in a steady state, i.e. an amount dm2*sqrt(G*m_2*r_2).
#    If there is no accretion disk then we calculate the angular momentum of the transferred material by
#    using the radius at which the disk would have formed (rdisk = 1.7*rmin, see Ulrich & Burger 1976) if allowed,
#    i.e. the angular momentum of the inner Lagrangian point, and add this directly to the secondary,
#    i.e. an amount dm2*SQRT(G*m_2*rdisk). Total angular momentum is conserved in this model.
#
# 4) Now using q_crit = 3.0 for MS-MS Roche systems (previously we had nothing). This corresponds roughly to
#    R proportional to M^5 which should be true for the majority of the MS (varies from (M^17 -> M^2).
#    If q > q_crit then contact occurs. For CHeB primaries we also take q_crit = 3.0 and allow
#    common-envelope to occur if this is exceeded.
#
# 5) The value of lambda used in calculations of the envelope binding energy for giants in common-envelope is
#    now variable (see function in zfuncs). The lambda function has been fitted by Onno to detailed
#    models ... he will write about this soon!
#
# 6) Note that eq.42 in the paper is missing a SQRT around the MR^2/a^5 part. This needs to be corrected
#    in any code update paper with a thanks to Jeremy Sepinsky (student at NorthWestern). It is ok in the code.

# March 2003
# New input options added:
# ifflag - for the mass of a WD you can choose to use the mass that results from the evolution algorithm
#          (basically a competition between core-mass growth and envelope mass-loss) or use the IFMR
#          proposed by Han, Podsiadlowski & Eggleton, 1995, MNRAS, 272, 800 [True activates HPE IFMR].
#
# wdflag - for the cooling of WDs you can choose to use either the standard Mestel cooling law (see SSE paper)
#          or a modified-Mestel law that is better matched to detailed models (provided by Brad Hansen ...
#          see Hurley & Shara, 2003, ApJ, May 20, in press) [True activates modified-Mestel].
#
# bhflag - choose whether black holes should get velocity kicks at formation [False: no kick; True: kick].
#
# nsflag - for the mass of neutron stars and black holes you can use either the SSE prescription or the prescription
#          presented by Belczynski et al. 2002, ApJ, 572, 407 who found that SSE was underestimating the masses
#          of these stars. In either case you also need to set the maximum NS mass (mxns) for the prescription
#          [0= SSE, mxns=1.8; >0 Belczynski, mxns=3.0].

# Sept 2004
# Input options added/changed:
# ceflag - set to 3 this uses de Kool (or Podsiadlowski) CE prescription,
#          other options, such as Yungelson, could be added as well.
#
# hewind - factor to control the amount of Helium star mass-loss, i.e. 1.0e-13*hewind*L^(2/3) gives He star mass-loss.


@conditional_njit()
def evolve(kstar, mass0, mass, rad, lumin, massc, radc, menv, renv,
           ospin, epoch, tms, tphys, tphysf, dtp, z, zcnsts, tb, ecc, kick, output, index):
    # 保存初始状态
    mass1i = mass0[1]
    mass2i = mass0[2]
    tbi = tb
    ecci = ecc

    ngtv = -1.0
    ngtv2 = -2.0


    # 初始化参数
    kmin = 1
    kmax = 2
    sgl = False
    mt2 = min(mass[1], mass[2])
    kst = 0

    # 未初始化参数
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

    tmsnew = 0
    rm = 0  # 恒星半径
    lum = 0
    mc = 0
    rc = 0
    me = 0
    re = 0
    k2 = 0
    dtr = 0
    jorb = 0
    dtm0 = 0
    djmb = 0
    djgr = 0
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

        (kstar[k], age, mass0[k], mass[k], lum, rm, mc, rc, me, re, k2, tm, tn, tscls, lums, GB) = hrdiag(
            kstar[k], age, mass0[k], mass[k], lum, rm, mc, rc, k2, tm, tn, tscls, lums, GB, kick, zcnsts)

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
    ip = -1
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
                    # 判断是否是共生星（Symbiotic-type stars）
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
                        djgr = djgr * jorb * (1.0 + 0.8750 * ecc2) / sqome5
                        delet1 = djgr * ecc * f1 / sqome5
                        djorb = djorb + djgr
                        delet = delet + delet1
                    for k in range(1, 3):
                        # 计算星风带走的恒星自旋角动量（包括吹走的和吸积过来的）
                        djtx[k] = (2.0 / 3.0) * xi * dmt[k] * rad[3 - k] * rad[3 - k] * ospin[3 - k]
                        djspint[k] = (2.0 / 3.0) * (dmr[k] * rad[k] * rad[k] * ospin[k]) - djtx[k]
                        # 计算有明显对流包层的恒星因磁制动损失的自旋角动量
                        # 包括主序星(M < 1.25)、靠近巨星分支的HG恒星以及巨星, 不包括完全对流主序星
                        # 这里的menv是对流包层的质量, 因此这里的条件判据只去除了完全对流的主序星和简并星, 这里的自旋以每年为单位
                        # if (0.35 < mass[k] < 1.25 and kstar[k] <= 1) or 2 <= kstar[k] <= 9:
                        if mb_judgment(mass[k], kstar[k]):
                            djmb = magnetic_braking(mass[k], menv[k], rad[k], ospin[k])
                            djspint[k] = djspint[k] + djmb
                            # 限制最大3%的磁制动损失的角动量, 这是在迭代次数上限20000时的最优解, 当然2%也不会影响演化结果
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
                            # 计算没有角动量能被转移时的平衡自旋
                            eqspin = oorb * f2 / (sqome3 * f5)
                            # 计算潮汐造成的轨道角动量变化
                            djt = (k2str[k] * (mass[k] - massc[k]) * rad[k] * rad[k] + k3 * massc[k] * radc[k] * radc[
                                k]) * dspint[k]
                            if kstar[k] <= 6 or abs(djt) / jspin[k] > 0.1:
                                djtt = djtt + djt
                    # 限制最大 2% 的轨道角动量变化 【改动: 原来这里是0.002】
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
                        if mb_judgment(mass[k], kstar[k]):
                            djmb = magnetic_braking(mass[k], menv[k], rad[k], ospin[k])
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
                        print(mass1i, mass2i, tbi, ecci)
                        raise ValueError('mass exceeded')

                    (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)

                    (kw, age, m0, mt, lum, rm, mc, rc, me, re, k2, tm, tn, tscls, lums, GB) = hrdiag(
                        kw, age, m0, mt, lum, rm, mc, rc, k2, tm, tn, tscls, lums, GB, kick, zcnsts)

                    if kw != 15:
                        ospin[k] = jspin[k] / (k2 * (mt - mc) * rm * rm + k3 * mc * rc * rc)

                    # 这个时候可能已经发生超新星爆发了
                    if kw != kstar[k] and kstar[k] <= 12 and (kw == 13 or kw == 14):
                        if not sgl:
                            (ecc, sep, jorb) = SN_kick(kw, mass[k], mt, mass[3 - k], ecc, sep, kick, index)
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

                    # Base new timescale for changes in radius & mass on stellar type.
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
                        output.bcm[ip, 34] = jorb
                        output.bcm[ip, 35] = djmb
                        output.bcm[ip, 36] = djgr
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
                            if abs(rdot[j1]) < tiny:
                                prec = True
                                dtm = 1e-7 * tphys
                            else:
                                dr = rad[j1] - 1.001 * rol[j1]
                                dtm = -dr / abs(rdot[j1])
                            # If this occurs, it is likely that the star is a high mass type 4
                            # where the radius can change very sharply, or there may be a
                            # discontinuity in the radius as a function of time. HRDIAG needs to be checked!
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
                        print('iterations exceeds', mass1i, mass2i, tbi, ecci)
                        raise ValueError('The number of iterations exceeds the maximum.')
                        # flag140 = True
                        # break

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
                    output.bcm[ip, 34] = jorb
                    output.bcm[ip, 35] = djmb
                    output.bcm[ip, 36] = djgr


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
                        tkh[k] = 1e7 * mass[k] / (rad[k] * lumin[k])
                        if kstar[k] <= 1 or kstar[k] == 7 or kstar[k] >= 10:
                            tkh[k] = tkh[k] * mass[k]
                        else:
                            tkh[k] = tkh[k] * (mass[k] - massc[k])

                    # donor星的动力学时标
                    tdyn = 5.05e-5 * np.sqrt(rad[j1] ** 3 / mass[j1])

                    # Identify special cases.
                    if kstar[j1] in {0, 1, 2, 4}:
                        qc = MT_stability_MS(kstar[j1], mass1i, mass2i, tbi)
                    elif kstar[j1] in {3, 5, 6}:
                        # qc = (1.67d0-zpars(7)+2.d0*(massc(j1)/mass(j1))**5)/2.13d0
                        # Alternatively use condition of Hjellming & Webbink, 1987, ApJ, 318, 794.
                        qc = 0.362 + 1.0 / (3.0 * (1.0 - massc[j1] / mass[j1]))
                    elif kstar[j1] in {7, 8, 9}:
                        if tb <= 1.644e-4:
                            qc = 0.1
                        else:
                            qc = 10.0
                    else:
                        qc = 3.0

                    # if kstar[j2] == 13:
                    #     qc = 3.5
                    #     if 7.0 <= kstar[j1] <= 9.0:

                    # 中子星/黑洞 + 非简并伴星的物质转移稳定性判据【Shao, Y., & Li, X.-D. 2021, ApJ, 920, 81】
                    if kstar[j2] == 13 or kstar[j2] == 14:
                        qrl = mass[j1] / mass[j2]
                        qmax = 2.1 + 0.8 * mass[j2]
                        if qrl < 2.0:
                            qc = 100.0
                        elif qrl > qmax:
                            qc = 0.0
                        else:
                            radmax = -173.8 + 45.5 * mass[j1] - 0.18 * mass[j1] ** 2
                            radmin = 6.6 - 26.1 * mass[j1] / mass[j2] + 11.4 * (mass[j1] / mass[j2]) ** 2
                            if rad[j1] > radmax:
                                qc = 0.0
                            elif rad[j1] < radmin:
                                qc = 0.0
                            else:
                                qc = 100.0

                        # 对于中子星+氦星, 如果周期小于0.06d, 则可能会发生CE (Tauris, T. 2015, MNRAS, 451, 2123)
                        if 7.0 <= kstar[j1] <= 9.0 and kstar[j2] == 13:
                            if tb <= 1.644e-4:
                                qc = 0.10
                            else:
                                qc = 10.00

                        # 对于黑洞+氦星, 认为系统是稳定的 (Tauris, T. 2015, MNRAS, 451, 2123)
                        if 7.0 <= kstar[j1] <= 9.0 and kstar[j2] == 14:
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
                            if (kstar[j2] == 10 and mass[j2] < 0.05) or (kstar[j2] >= 11 and mass[j2] < 0.5):
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
                        output.bcm[ip, 33] = 1
                        m1ce = mass[j1]
                        m2ce = mass[j2]

                        comenv_return = comenv(mass0[j1], mass[j1], massc[j1], aj[j1], jspin[j1], kstar[j1],
                                               mass0[j2], mass[j2], massc[j2], aj[j2], jspin[j2], kstar[j2],
                                               ecc, sep, jorb, kick, zcnsts, index)

                        (mass0[j1], mass[j1], massc[j1], aj[j1], jspin[j1], kstar[j1],
                         mass0[j2], mass[j2], massc[j2], aj[j2], jspin[j2], kstar[j2],
                         ecc, sep, jorb, coel) = comenv_return

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
                    # 当donor星为白矮星, 且质量比大于临界值时, 也会发生动力学物质转移
                    elif kstar[j1] in {10, 11, 12} and kstar[j2] in {10, 11, 12} and q[j1] > 0.628:
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
                            # Assume the energy released by ignition of the triple-alpha reaction
                            # is enough to destroy the star.
                            kstar[j2] = 15
                            mass[j2] = 0.0
                        elif kstar[j1] == 10 or kstar[j2] == 10:
                            # Should be helium overflowing onto a CO or ONe core in which case the
                            # helium swells up to form a giant envelope so a HeGB star is formed.
                            # Allowance for the rare case of CO or ONe flowing onto He is made.
                            kst = 9
                            if kstar[j2] == 10:
                                massc[j2] = dm2
                            (massc[j2], mass[j2], kst, mass0[j2], aj[j2]) = gntage(
                                massc[j2], mass[j2], kst, zcnsts, mass0[j2], aj[j2])
                            kstar[j2] = kst
                            epoch[j2] = tphys - aj[j2]
                        elif kstar[j2] <= 12:
                            mass0[j2] = mass[j2]
                            if kstar[j1] == 12 and kstar[j2] == 11:
                                # Mixture of ONe and CO will result in an ONe product.
                                kstar[j2] = 12
                        kstar[j1] = 15
                        mass[j1] = 0.0

                        # Might be a supernova that destroys the system.
                        if kstar[j2] <= 11 and mass[j2] > mch:
                            kstar[j2] = 15
                            mass[j2] = 0.0
                        coel = True
                        # goto.one_three_five
                        flag135 = True
                        break
                    # 白矮星向中子星/黑洞转移物质, 存在动力学稳定的最大WD质量
                    elif kstar[j1] in {10, 11, 12} and kstar[j2] in {13, 14} and mass[j1] > max_WD_mass:
                        # 白矮星潮汐瓦解之后, 如果存在吸积盘, 应该会进入吸积盘, 否则应该加到中子星黑洞质量上？
                        # 有可能发生长伽马暴？
                        kstar[j1] = 15
                        mass[j1] = 0.0
                        coel = True
                        # goto.one_three_five
                        flag135 = True
                        break
                    # donor星为中子星
                    elif kstar[j1] == 13:
                        # Gamma ray burster?
                        mass[j2] = mass[j2] + mass[j1]
                        kstar[j2] = 13 if mass[j2] <= mxns else 14
                        kstar[j1] = 15
                        mass[j1] = 0.0
                        coel = True
                        # goto.one_three_five
                        flag135 = True
                        break
                    # donor星为黑洞
                    elif kstar[j1] == 14:
                        # Both stars are black holes.  Let them merge quietly.
                        mass[j2] = mass[j2] + mass[j1]
                        kstar[j1] = 15
                        mass[j1] = 0.0
                        coel = True
                        # goto.one_three_five
                        flag135 = True
                        break
                    else:
                        # Mass transfer in one Kepler orbit.
                        # 动力学稳定的物质转移(核时标物质转移)
                        dm1 = 3.0e-6 * tb * (np.log(rad[j1] / rol[j1]) ** 3) * min(mass[j1], 5.0) ** 2
                        if kstar[j1] == 2:
                            mew = (mass[j1] - massc[j1]) / mass[j1]
                            dm1 = max(mew, 0.01) * dm1
                        # 对致密星WD来说, 物质转移速率会很大, 通常与质量成正比, 因此WD向NS转移物质的初期时标很短, 很快WD的质量
                        # 就下降到0.1Msun以下, 之后就是长期的低速率物质转移, 甚至可以到哈勃时标结束
                        elif kstar[j1] >= 10:
                            dm1 = dm1 * 1e3 * mass[j1] / max(rad[j1], 1e-4)
                        kst = kstar[j2]
                        # Possibly mass transfer needs to be reduced if primary is rotating
                        # faster than the orbit (not currently implemented).
                        # spnfac = MIN(3.d0,MAX(ospin(j1)/oorb,1.d0))
                        # dm1 = dm1/spnfac**2
                        # Limit mass transfer to the thermal rate for remaining giant-like stars
                        # and to the dynamical rate for all others.
                        if 2 <= kstar[j1] <= 9 and kstar[j1] != 7:
                            dm1 = min(dm1, mass[j1] * tb / tkh[j1])
                        elif rad[j1] > 10.0 * rol[j1] or (kstar[j1] <= 1 and kstar[j2] <= 1 and q[j1] > qc):
                            # Allow the stars to merge with the product in *1.
                            m1ce = mass[j1]
                            m2ce = mass[j2]
                            (mass0, mass, aj, kstar) = mix(mass0, mass, aj, kstar, zcnsts)
                            dm1 = m1ce - mass[j1]
                            dm2 = mass[j2] - m2ce
                            # Next step should be made without changing the time.
                            dtm = 0.0
                            epoch[1] = tphys - aj[1]
                            coel = True
                            # goto.one_three_five
                            flag135 = True
                            break
                        else:
                            dm1 = min(dm1, mass[j1] * tb / tdyn)
                        # Calculate wind mass loss from the stars during one orbit.
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
                        # Increase timescale to relative mass loss of 0.5% but not more than twice.
                        # KM is the number of orbits for the timestep.
                        km = min(2.0 * km0, 5.0e-3 / max(abs(dm1 + dms[j1]) / mass[j1], dms[j2] / mass[j2]))
                        km0 = km
                        # Modify time-step & mass loss terms by speed-up factor.
                        dt = km * tb
                        dtm = dt / 1.0e6
                        # Take the stellar evolution timestep into account but don't let it
                        # be overly restrictive for long-lived phases.
                        if iter <= 1000:
                            dtm = min(dtm, dtmi[1], dtmi[2])
                        dtm = min(dtm, tsave - tphys)
                        dt = dtm * 1.0e6
                        km = dt / tb
                        # Decide between accreted mass by secondary and/or system mass loss.
                        taum = mass[j2] / dm1 * tb
                        if kstar[j2] in {0, 1, 2, 4}:
                            if mass_accretion_model == 1:
                                ospbru2 = 2 * np.pi * np.sqrt(mass[2] * aursun ** 3 / rad[2] ** 3)
                                kkk = 1.0 - ospin[2] / ospbru2
                                dm2 = min(1.00, kkk) * dm1
                            elif mass_accretion_model == 2:
                                dm2 = 0.5 * dm1
                            elif mass_accretion_model == 3:
                                # Limit according to the thermal timescale of the secondary.
                                dm2 = min(1.0, 10.0 * taum / tkh[j2]) * dm1
                            else:
                                raise ValueError('Please provide an allowed scheme of mass_accretion_model.')
                        elif kstar[j2] in {7, 8, 9}:
                            # Naked helium star secondary swells up to a core helium burning star
                            # or SAGB star unless the primary is also a helium star.
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
                        # 白矮星吸积富氢物质,
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
                                # Check for planets or low-mass WDs.
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
                            # Assume the half accretion and impose the Eddington limit.
                            dm2 = min(0.5 * dm1, dme)
                            if dm2 < dm1:
                                supedd = True
                        # 巨星包层可以吸收任何吸积物质
                        else:
                            dm2 = dm1

                        if not novae:
                            dm22 = dm2
                        if 10 <= kst <= 12:
                            mt2 = mass[j2] + km * (dm22 - dms[j2])
                            # 氦白矮星只能吸积富氦物质达到0.7Msun, 否则就会发生Ia超新星
                            if kstar[j1] <= 10 and kst == 10 and mt2 >= 0.7:
                                mass[j1] = mass[j1] - km * (dm1 + dms[j1])
                                mass[j2] = 0.0
                                kstar[j2] = 15
                                # goto.one_three_five
                                flag135 = True
                                break
                            elif kstar[j1] <= 10 and kst >= 11:
                                if (mt2 - mass0[j2]) >= 0.15:
                                    # CO and ONeWDs accrete helium-rich material until the accumulated
                                    # material exceeds a mass of 0.15 when it ignites. For a COWD with
                                    # mass less than 0.95 the system will be destroyed as an ELD in a
                                    # possible Type 1a SN. COWDs with mass greater than 0.95 and ONeWDs
                                    # will survive with all the material converted to ONe (JH 30/09/99).
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
                            # If the Chandrasekhar limit is exceeded for a white dwarf then destroy
                            # the white dwarf in a supernova. If the WD is ONe then a neutron star
                            # will survive the supernova, and we let HRDIAG take care of this when
                            # the stars are next updated.
                            if kst == 10 or kst == 11:
                                if mt2 >= mch:
                                    dm1 = mch - mass[j2] + km * dms[j2]
                                    mass[j1] = mass[j1] - dm1 - km * dms[j1]
                                    mass[j2] = 0.0
                                    kstar[j2] = 15
                                    # goto.one_three_five
                                    flag135 = True
                                    break
                        # Modify mass loss terms by speed-up factor.
                        dm1 = km * dm1
                        dm2 = km * dm2
                        dm22 = km * dm22
                        dme = km * dme
                        # Calculate orbital angular momentum change due to system mass loss.
                        djorb = ((dmr[1] + q[1] * dmt[1]) * mass[2] * mass[2] +
                                 (dmr[2] + q[2] * dmt[2]) * mass[1] * mass[1]) / (mass[1] + mass[2]) ** 2
                        djorb = djorb * dt
                        # For super-Eddington mass transfer rates, for gamma = -2.0, and for novae systems,
                        # assume that material is lost from the system as if a wind from the secondary.
                        # If gamma = -1.0 then assume the lost material carries with it the specific angular momentum
                        # the specific angular momentum of the primary and for all gamma > 0.0 assume that
                        # it takes away a fraction gamma of the orbital angular momentum.
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
                        # For very close systems include angular momentum loss mechanisms.
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
                            # Calculate change in the intrinsic spin of the star.
                            djspint[k] = (2.0 / 3.0) * (dmr[k] * radx[k] * radx[k] * ospin[k] -
                                                        xi * dmt[k] * radx[3 - k] * radx[3 - k] * ospin[3 - k])
                            djspint[k] = djspint[k] * dt
                            if mb_judgment(mass[k], kstar[k]):
                                djmb = magnetic_braking(mass[k], menv[k], rad[k], ospin[k])
                                djspint[k] = djspint[k] + djmb * dt
                        # Adjust the spin angular momentum of each star owing to mass transfer
                        # and conserve total angular momentum.
                        djt = dm1 * radx[j1] * radx[j1] * ospin[j1]
                        djspint[j1] = djspint[j1] + djt
                        djorb = djorb - djt
                        if disk:
                            # Alter spin of the degenerate secondary by assuming that material
                            # falls onto the star from the inner edge of a Keplerian accretion
                            # disk and that the system is in a steady state.
                            djt = dm2 * 2 * np.pi * aursun * np.sqrt(aursun * mass[j2] * radx[j2])
                            djspint[j2] = djspint[j2] - djt
                            djorb = djorb + djt
                        else:
                            # No accretion disk.
                            # Calculate the angular momentum of the transferred material by
                            # using the radius of the disk (see Ulrich & Burger) that would
                            # have formed if allowed.
                            rdisk = 1.70 * rmin
                            djt = dm2 * 2 * np.pi * aursun * np.sqrt(aursun * mass[j2] * rdisk)
                            djspint[j2] = djspint[j2] - djt
                            djorb = djorb + djt
                        djtx[2] = djt
                        # Adjust the secondary spin if a nova eruption has occurred.
                        if novae:
                            djt = (dm2 - dm22) * radx[j2] * radx[j2] * ospin[j2]
                            djspint[j2] = djspint[j2] + djt
                            djtx[2] = djtx[2] - djt
                        # Calculate circularization, orbital shrinkage and spin up.
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
                            # Ensure that the star does not spin up beyond break-up, and transfer
                            # the excess angular momentum back to the orbit.
                            ospbru = 2 * np.pi * np.sqrt(mass[k] * aursun ** 3 / radx[k] ** 3)
                            jspbru = (k2str[k] * (mass[k] - massc[k]) * radx[k] * radx[k] + k3 * massc[k] * radc[k] *
                                      radc[k]) * ospbru
                            if jspin[k] > jspbru:
                                mew = 1.0
                                if djtx[2] > 0.0:
                                    mew = min(mew, (jspin[k] - jspbru) / djtx[2])
                                djorb = djorb - (jspin[k] - jspbru)
                                jspin[k] = jspbru
                                # If excess material should not be accreted, activate next line.
                                # dm22 = (1.d0 - mew)*dm22
                        # Update the masses.
                        kstar[j2] = kst
                        mass[j1] = mass[j1] - dm1 - dms[j1]
                        if kstar[j1] <= 1 or kstar[j1] == 7:
                            mass0[j1] = mass[j1]
                        mass[j2] = mass[j2] + dm22 - dms[j2]
                        if kstar[j2] <= 1 or kstar[j2] == 7:
                            mass0[j2] = mass[j2]
                        # For a HG star check if the initial mass can be reduced.
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
                        # Ensure that Jorb does not become negative which could happen if the
                        # primary overfills its Roche lobe initially. In this case we simply
                        # allow contact to occur.
                        jorb = max(1.0, jorb - djorb)
                        sep = (mass[1] + mass[2]) * jorb * jorb / (
                                (mass[1] * mass[2] * 2 * np.pi) ** 2 * aursun ** 3 * (1.0 - ecc * ecc))
                        tb = (sep / aursun) * np.sqrt(sep / (aursun * (mass[1] + mass[2])))
                        oorb = 2 * np.pi / tb

                    # Always rejuvenate the secondary and age the primary if they are on the main sequence.
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

                    # Obtain the stellar parameters for the next step.
                    tphys = tphys + dtm
                    for k in range(1, 3):
                        age = tphys - epoch[k]
                        m0 = mass0[k]
                        mt = mass[k]
                        mc = massc[k]
                        # Masses over 100Msun should probably not be trusted in the evolution formulae.
                        if mt > 100.0:
                            # goto.one_four_zero
                            # flag140 = True
                            print(mass1i, mass2i, tbi, ecci)
                            raise ValueError('mass exceeded')

                        kw = kstar[k]
                        (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)

                        (kw, age, m0, mt, lum, rm, mc, rc, me, re, k2, tm, tn, tscls, lums, GB) = hrdiag(
                            kw, age, m0, mt, lum, rm, mc, rc, k2, tm, tn, tscls, lums, GB, kick, zcnsts)

                        # Check for a supernova and correct the semi-major axis if so.
                        if kw != kstar[k] and kstar[k] <= 12 and (kw == 13 or kw == 14):
                            dms[k] = mass[k] - mt
                            (ecc, sep, jorb) = SN_kick(kw, mass[k], mt, mass[3 - k], ecc, sep, kick, index)
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

                        # Determine stellar evolution timescale for nuclear burning types.
                        if kw <= 9:
                            (dt, dtr) = timestep(kw, age, tm, tn, tscls, dt)
                            dtmi[k] = min(dt, dtr)
                            dtmi[k] = max(1.0e-7, dtmi[k])
                        else:
                            dtmi[k] = 1.0e10

                        # Save relevent solar quantities.
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
                        output.bcm[ip, 34] = jorb
                        output.bcm[ip, 35] = djmb
                        output.bcm[ip, 36] = djgr
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

                    # Test whether the primary still fills its Roche lobe.
                    if rad[j1] > rol[j1] and not snova:
                        # Test for a contact system
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
                # If *1 or *2 is giant-like this will be common-envelope evolution.
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
                                           jspin[j2], kstar[j2], ecc, sep, jorb, kick, zcnsts, index)
                    (mass0[j1], mass[j1], massc[j1], aj[j1], jspin[j1],
                     kstar[j1], mass0[j2], mass[j2], massc[j2], aj[j2],
                     jspin[j2], kstar[j2], ecc, sep, jorb, coel) = comenv_return
                    # com 是 common envelop的意思
                    com = True
                elif 2 <= kstar[j2] <= 9 and kstar[j2] != 7:
                    comenv_return = comenv(mass0[j2], mass[j2], massc[j2], aj[j2], jspin[j2],
                                           kstar[j2], mass0[j1], mass[j1], massc[j1], aj[j1],
                                           jspin[j1], kstar[j1], ecc, sep, jorb, kick, zcnsts, index)
                    (mass0[j2], mass[j2], massc[j2], aj[j2], jspin[j2],
                     kstar[j2], mass0[j1], mass[j1], massc[j1], aj[j1],
                     jspin[j1], kstar[j1], ecc, sep, jorb, coel) = comenv_return

                    com = True
                else:
                    (mass0, mass, aj, kstar) = mix(mass0, mass, aj, kstar, zcnsts)

                if com:
                    output.bcm[ip, 33] = 1
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
                    # Next step should be made without changing the time.
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
                    # Reset orbital parameters as separation may have changed.
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
                        # Binary dissolved by a supernova or tides.
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
            # Cases of accretion induced supernova or single star supernova. No remnant is left in either case.
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
        output.bcm[ip, 34] = jorb
        output.bcm[ip, 35] = djmb
        output.bcm[ip, 36] = djgr
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


