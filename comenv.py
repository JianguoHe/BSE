import numpy as np
from utils import conditional_njit
from star import star
from supernova import SN_kick
from Lambda import lambda_cal
from hrdiag import hrdiag
from gntage import gntage
from utils import rochelobe
from dgcore import dgcore
from const import alpha, ktype, ceflag, aursun, HG_survive_CE


# Common Envelope Evolution.
# Author: C. A. Tout
# Date:   18th September 1996
# Redone: J. R. Hurley
# Date:   7th July 1998

# Common envelope evolution - entered only when kw1 = 2, 3, 4, 5, 6, 8 or 9.
# For simplicity energies are divided by -G.


@conditional_njit()
def comenv(m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, kick, zcnsts, index):
    # 设置初始参数
    k21 = 0          # 计算巨星包层的自旋角动量
    k22 = 0
    k3 = 0.21        # 计算WD/NS/致密对流核的自旋角动量  Jspin = k3*omega*MR2
    coel = False
    r1 = 0
    r2 = 0
    lum1 = 0
    lum2 = 0
    rc1 = 0
    rc2 = 0
    mc3 = 0
    mc22 = 0
    sepl = 0
    fage1 = 0
    fage2 = 0
    ebindf = 0

    # 获得核质量和半径
    kw = kw1
    (tm1, tn, tscls1, lums, GB) = star(kw1, m01, m1, zcnsts)
    (kw1, aj1, m01, m1, lum1, r1, mc1, rc1, menv, renv, k21, tm1, tn, tscls1, lums, GB) = hrdiag(
        kw1, aj1, m01, m1, lum1, r1, mc1, rc1, k21, tm1, tn, tscls1, lums, GB, kick, zcnsts)

    ospin1 = jspin1 / (k21 * r1 * r1 * (m1 - mc1) + k3 * rc1 * rc1 * mc1)

    kw = kw2
    (tm2, tn, tscls2, lums, GB) = star(kw2, m02, m2, zcnsts)
    (kw2, aj2, m02, m2, lum2, r2, mc2, rc2, menv, renv, k22, tm2, tn, tscls2, lums, GB) = hrdiag(
        kw2, aj2, m02, m2, lum2, r2, mc2, rc2, k22, tm2, tn, tscls2, lums, GB, kick, zcnsts)
    ospin2 = jspin2 / (k22 * r2 * r2 * (m2 - mc2) + k3 * rc2 * rc2 * mc2)

    # 计算公共包层结合能因子λ(Xiao-Jie Xu, & Xiang-Dong Li (2010))
    lambda_bind = lambda_cal(m01, r1, zcnsts.z)

    # 计算巨星包层结合能
    # sss = random.uniform(0, 1)
    sss = 1.0
    ebindi = m1 * (m1 - mc1) / (sss * lambda_bind * r1)

    # 如果次星也是类巨星的话, 加上它包层的能量, 同时计算初始轨道能量【改动】
    if 2 <= kw2 <= 9 and kw2 != 7:
        lambda_bind_2 = lambda_cal(m02, r2, zcnsts.z)
        ebindi = ebindi + m2 * (m2 - mc2) / (lambda_bind_2 * r2)
        eorbi = m1 * m2 / (2.0 * sep) if ceflag == 3 else mc1 * mc2 / (2.0 * sep)
    else:
        eorbi = m1 * m2 / (2.0 * sep) if ceflag == 3 else mc1 * m2 / (2.0 * sep)

    # 考虑偏心轨道
    ecirc = eorbi / (1.0 - ecc * ecc)

    # 计算没有合并的最终轨道能量
    eorbf = eorbi + ebindi / alpha

    # 是否允许 HG donor 离开CE
    if not HG_survive_CE and kw1 == 2:
        coel = True

    # 如果次星是主序星的话看它是否充满了洛希瓣
    if kw2 <= 1 or kw2 == 7:
        sepf = mc1 * m2 / (2.0 * eorbf)
        q1 = mc1 / m2
        q2 = 1.0 / q1
        rl1 = rochelobe(q1)
        rl2 = rochelobe(q2)
        # The helium core of a very massive star of type 4 may actually fill
        # its Roche lobe in a wider orbit with a very low-mass secondary.
        if rc1 / rl1 >= r2 / rl2:
            if rc1 > rl1 * sepf:
                coel = True
                sepl = rc1 / rl1
        else:
            if r2 > rl2 * sepf:
                coel = True
                sepl = r2 / rl2
        if coel:
            kw = ktype[int(kw1), int(kw2)] - 100
            mc3 = mc1
            if kw2 == 7 and kw == 4:
                mc3 = mc3 + m2
            # 并合，计算最终结合能
            eorbf = max(mc1 * m2 / (2.0 * sepl), eorbi)
            ebindf = ebindi - alpha * (eorbf - eorbi)
        else:
            # 主星变成了一个黑洞/中子星/白矮星/氦星
            MF = m1
            m1 = mc1
            (tm1, tn, tscls1, lums, GB) = star(kw1, m01, m1, zcnsts)
            (kw1, aj1, m01, m1, lum1, r1, mc1, rc1, menv, renv, k21, tm1, tn, tscls1, lums, GB) = hrdiag(
                kw1, aj1, m01, m1, lum1, r1, mc1, rc1, k21, tm1, tn, tscls1, lums, GB, kick, zcnsts)

            if kw1 >= 13:
                (ecc, sepf, jorb) = SN_kick(kw1, MF, m1, m2, ecc, sepf, kick, index)
                if ecc > 1.0:
                    sep = sepf
                    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel
    # Degenerate or giant secondary. Check if the least massive core fills its Roche lobe.
    else:
        sepf = mc1 * mc2 / (2.0 * eorbf)
        q1 = mc1 / mc2
        q2 = 1.0 / q1
        rl1 = rochelobe(q1)
        rl2 = rochelobe(q2)
        if rc1 / rl1 >= rc2 / rl2:
            if rc1 > rl1 * sepf:
                coel = True
                sepl = rc1 / rl1
        else:
            if rc2 > rl2 * sepf:
                coel = True
                sepl = rc2 / rl2

        # 如果最终并合
        if coel:
            # If the secondary was a neutron star or black hole the outcome
            # is an unstable Thorne-Zytkow object that leaves only the core.
            sepf = 0.0
            if kw2 >= 13:
                mc1 = mc2
                m1 = mc1
                mc2 = 0.0
                m2 = 0.0
                kw1 = kw2
                kw2 = 15
                aj1 = 0.0
                # 这种情况下不需要包层的质量
                sep = sepf
                return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel
            kw = ktype[int(kw1), int(kw2)] - 100
            mc3 = mc1 + mc2
            # 计算最终包层结合能
            eorbf = max(mc1 * mc2 / (2.0 * sepl), eorbi)
            ebindf = ebindi - alpha * (eorbf - eorbi)

            # Check if we have the merging of two degenerate cores and if so
            # then see if the resulting core will survive or change form.
            if kw1 == 6 and (kw2 == 6 or kw2 >= 11):
                (kw1, kw2, kw, mc1, mc2, mc3, ebindf) = dgcore(kw1, kw2, kw, mc1, mc2, mc3, ebindf)
            if kw1 <= 3 and m01 <= zcnsts.zpars[2]:
                if (2 <= kw2 <= 3 and m02 <= zcnsts.zpars[2]) or kw2 == 10:
                    (kw1, kw2, kw, mc1, mc2, mc3, ebindf) = dgcore(kw1, kw2, kw, mc1, mc2, mc3, ebindf)
                    if kw >= 10:
                        kw1 = kw
                        m1 = mc3
                        mc1 = mc3
                        if kw < 15:
                            m01 = mc3
                        aj1 = 0.0
                        mc2 = 0.0
                        m2 = 0.0
                        kw2 = 15
                        sep = sepf
                        return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel
        # The cores do not coalesce - assign the correct masses and ages.
        else:
            mf = m1
            m1 = mc1
            (tm1, tn, tscls1, lums, GB) = star(kw1, m01, m1, zcnsts)
            (kw1, aj1, m01, m1, lum1, r1, mc1, rc1, menv, renv, k21, tm1, tn, tscls1, lums, GB) = hrdiag(
                kw1, aj1, m01, m1, lum1, r1, mc1, rc1, k21, tm1, tn, tscls1, lums, GB, kick, zcnsts)

            if kw1 >= 13:
                (ecc, sepf, jorb) = SN_kick(kw1, mf, m1, m2, ecc, sepf, kick, index)
                if ecc > 1.0:
                    sep = sepf
                    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel
            mf = m2
            kw = kw2
            m2 = mc2
            (tm2, tn, tscls2, lums, GB) = star(kw2, m02, m2, zcnsts)
            (kw2, aj2, m02, m2, lum2, r2, mc2, rc2, menv, renv, k22, tm2, tn, tscls2, lums, GB) = hrdiag(
                kw2, aj2, m02, m2, lum2, r2, mc2, rc2, k22, tm2, tn, tscls2, lums, GB, kick, zcnsts)

            if kw2 >= 13 and kw < 13:
                (ecc, sepf, jorb) = SN_kick(kw2, mf, m2, m1, ecc, sepf, kick, index)
                if ecc > 1.0:
                    sep = sepf
                    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel

    if coel:
        mc22 = mc2
        if kw == 4 or kw == 7:
            # If making a helium burning star calculate the fractional age
            # depending on the amount of helium that has burnt.
            if kw1 <= 3:
                fage1 = 0.0
            elif kw1 >= 6:
                fage1 = 1.0
            else:
                fage1 = (aj1 - tscls1[2]) / (tscls1[13] - tscls1[2])
            if kw2 <= 3 or kw2 == 10:
                fage2 = 0.0
            elif kw2 == 7:
                fage2 = aj2 / tm2
                mc22 = m2
            elif kw2 >= 6:
                fage2 = 1.0
            else:
                fage2 = (aj2 - tscls2[2]) / (tscls2[13] - tscls2[2])

    if coel:
        # Calculate the orbital spin just before coalescence.
        TB = (sepl / aursun) * np.sqrt(sepl / (aursun * (mc1 + mc2)))
        oorb = 2 * np.pi / TB

        xx = 1.0 + zcnsts.zpars[7]
        if ebindf <= 0.0:
            mf = mc3
        else:
            const = ((m1 + m2) ** xx) * (m1 - mc1 + m2 - mc22) * ebindf / ebindi
            mf = max(mc1 + mc22, (m1 + m2) * (ebindf / ebindi) ** (1.0 / xx))
            dely = (mf ** xx) * (mf - mc1 - mc22) - const
            while abs(dely/mf) > 1.0e-3:
                deri = mf ** zcnsts.zpars[7] * ((1.0 + xx) * mf - xx * (mc1 + mc22))
                delmf = dely / deri
                mf = mf - delmf
                dely = (mf ** xx) * (mf - mc1 - mc22) - const

        # 设置质量和间距
        if mc22 == 0.0:
            mf = max(mf, mc1+m2)
        m2 = 0.0
        m1 = mf
        kw2 = 15

        # Combine the core masses.
        if kw == 2:
            (tm2, tn, tscls2, lums, GB) = star(kw, m1, m1, zcnsts)
            if GB[9] >= mc1:
                m01 = m1
                aj1 = tm2 + (tscls2[1] - tm2) * (aj1 - tm1) / (tscls1[1] - tm1)
                (tm1, tn, tscls1, lums, GB) = star(kw, m01, m1, zcnsts)
        elif kw == 7:
            m01 = m1
            (tm1, tn, tscls1, lums, GB) = star(kw, m01, m1, zcnsts)
            aj1 = tm1 * (fage1 * mc1 + fage2 * mc22) / (mc1 + mc22)
        elif kw == 4 or mc2 > 0.0 or kw != kw1:
            if (kw == 4):
                aj1 = (fage1 * mc1 + fage2 * mc22) / (mc1 + mc22)
            mc1 = mc1 + mc2
            mc2 = 0.0
            # 为巨星获得一个新的年龄
            (mc1, m1, kw, m01, aj1) = gntage(mc1, m1, kw, zcnsts, m01, aj1)
            (tm1, tn, tscls1, lums, GB) = star(kw, m01, m1, zcnsts)

        (kw, aj1, m01, m1, lum1, r1, mc1, rc1, menv, renv, k21, tm1, tn, tscls1, lums, GB) = hrdiag(
            kw, aj1, m01, m1, lum1, r1, mc1, rc1, k21, tm1, tn, tscls1, lums, GB, kick, zcnsts)

        jspin1 = oorb * (k21 * r1 * r1 * (m1 - mc1) + k3 * rc1 * rc1 * mc1)
        kw1 = kw
        ecc = 0.0
    else:
        if eorbf < ecirc:
            ecc = np.sqrt(1.0 - eorbf / ecirc)
        else:
            ecc = 0.0
        TB = (sepf / aursun) * np.sqrt(sepf / (aursun * (m1 + m2)))
        oorb = 2 * np.pi / TB
        jorb = m1 * m2 / (m1 + m2) * np.sqrt(1.0 - ecc * ecc) * sepf * sepf * oorb
        jspin1 = ospin1 * (k21 * r1 * r1 * (m1 - mc1) + k3 * rc1 * rc1 * mc1)
        jspin2 = ospin2 * (k22 * r2 * r2 * (m2 - mc2) + k3 * rc2 * rc2 * mc2)

    sep = sepf
    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel


