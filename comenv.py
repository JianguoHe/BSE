import numpy as np
from numba import njit
from star import star
from vkick import vkick
from Lambda import lambda_cal
from hrdiag import hrdiag
from gntage import gntage
from zfuncs import rzamsf, rochelobe
from dgcore import dgcore
from const import alpha, ktype, ceflag, aursun


# Common Envelope Evolution.
# Author: C. A. Tout
# Date:   18th September 1996
# Redone: J. R. Hurley
# Date:   7th July 1998

# Common envelope evolution - entered only when kw1 = 2, 3, 4, 5, 6, 8 or 9.
# For simplicity energies are divided by -G.


@njit
def comenv(m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, zcnsts, ecc, sep, jorb, coel, z, kick):
    # 设置初始参数
    # tm1 = 0
    # tm2 = 0
    # tn = 0
    K3 = 0.21     # kMR2 转动惯量中的参数, 用于白矮星、中子星或致密对流核
    coel = False
    r1 = 0
    r2 = 0
    lum1 = 0
    lum2 = 0
    rc1 = 0
    rc2 = 0
    mc3 = 0
    menv = 0
    mc22 = 0
    renv = 0
    k21 = 0
    k22 = 0
    sepl = 0
    fage1 = 0
    fage2 = 0
    ebindf = 0

    vs = np.array([0.0, 0, 0, 0])
    # tscls1 = np.zeros((1, 21)).flatten()
    # tscls2 = tscls1.copy()
    # lums = np.zeros((1, 11)).flatten()
    # GB = lums.copy()

    # 获得核质量和半径
    kw = kw1
    (tm1, tn, tscls1, lums, GB) = star(kw1, m01, m1, zcnsts)
    (m01, aj1, m1, tm1, tn, tscls1, lums, GB, r1, lum1, kw1, mc1, rc1, menv, renv, k21) = hrdiag(
        m01, aj1, m1, tm1, tn, tscls1, lums, GB, zcnsts, r1, lum1, kw1, mc1, rc1, menv, renv, k21, kick)
    ospin1 = jspin1 / (k21 * r1 * r1 * (m1 - mc1) + K3 * rc1 * rc1 * mc1)
    # menvd = menv / (m1 - mc1)
    # rzams = rzamsf(m01, zcnsts)
    kw = kw2
    (tm2, tn, tscls2, lums, GB) = star(kw2, m02, m2, zcnsts)
    (m02, aj2, m2, tm2, tn, tscls2, lums, GB, r2, lum2, kw2, mc2, rc2, menv, renv, k22) = hrdiag(
        m02, aj2, m2, tm2, tn, tscls2, lums, GB, zcnsts, r2, lum2, kw2, mc2, rc2, menv, renv, k22, kick)
    ospin2 = jspin2 / (k22 * r2 * r2 * (m2 - mc2) + K3 * rc2 * rc2 * mc2)

    # 计算公共包层结合能因子λ(Xiao-Jie Xu, & Xiang-Dong Li (2010))
    Lambda = lambda_cal(m01, r1, z)

    # 计算巨星包层结合能
    # sss = random.uniform(0, 1)
    sss = 1.0
    ebindi = m1 * (m1 - mc1) / (sss * Lambda * r1)

    # 如果次星也是类巨星的话, 加上它包层的能量, 同时计算初始轨道能量
    if 2 <= kw2 <= 9 and kw2 != 7:
        # menvd = menv / (m2 - mc2)
        # rzams = rzamsf(m02, zcnsts)
        Lamb2 = Lambda
        ebindi = ebindi + m2 * (m2 - mc2) / (Lamb2 * r2)
        eorbi = m1 * m2 / (2.0 * sep) if ceflag == 3 else mc1 * mc2 / (2.0 * sep)
    else:
        eorbi = m1 * m2 / (2.0 * sep) if ceflag == 3 else mc1 * m2 / (2.0 * sep)

    # 考虑偏心轨道
    ecirc = eorbi / (1.0 - ecc * ecc)

    # 计算没有合并的最终轨道能量
    eorbf = eorbi + ebindi / alpha

    # 如果次星是主序星的话看它是否充满了洛希瓣
    if kw2 <= 1 or kw2 == 7:
        sepf = mc1 * m2 / (2.0 * eorbf)
        q1 = mc1 / m2
        q2 = 1.0 / q1
        rl1 = rochelobe(q1)
        rl2 = rochelobe(q2)
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
            (m01, aj1, m1, tm1, tn, tscls1, lums, GB, r1, lum1, kw1, mc1, rc1, menv, renv, k21) = hrdiag(
                m01, aj1, m1, tm1, tn, tscls1, lums, GB, zcnsts, r1, lum1, kw1, mc1, rc1, menv, renv, k21, kick)
            if kw1 >= 13:
                (kw1, MF, m1, m2, ecc, sepf, jorb, vs) = vkick(kw1, MF, m1, m2, ecc, sepf, jorb, vs, kick)
                if ecc > 1.0:
                    sep = sepf
                    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel, z
    # 次星是类巨星
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
                return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel, z
            kw = ktype[int(kw1), int(kw2)] - 100
            mc3 = mc1 + mc2
            # 计算最终包层结合能
            eorbf = max(mc1 * mc2 / (2.0 * sepl), eorbi)
            ebindf = ebindi - alpha * (eorbf - eorbi)

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
                        return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel, z
        # 如果最终没有并合
        else:
            mf = m1
            m1 = mc1
            (tm1, tn, tscls1, lums, GB) = star(kw1, m01, m1, zcnsts)
            (m01, aj1, m1, tm1, tn, tscls1, lums, GB, r1, lum1, kw1, mc1, rc1, menv, renv, k21) = hrdiag(
                m01, aj1, m1, tm1, tn, tscls1, lums, GB, zcnsts, r1, lum1, kw1, mc1, rc1, menv, renv, k21, kick)
            if kw1 >= 13:
                (kw1, mf, m1, m2, ecc, sepf, jorb, vs) = vkick(kw1, mf, m1, m2, ecc, sepf, jorb, vs, kick)
                if ecc > 1.0:
                    sep = sepf
                    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel, z
            mf = m2
            kw = kw2
            m2 = mc2
            (tm2, tn, tscls2, lums, GB) = star(kw2, m02, m2, zcnsts)
            (m02, aj2, m2, tm2, tn, tscls2, lums, GB, r2, lum2, kw2, mc2, rc2, menv, renv, k22) = hrdiag(
                m02, aj2, m2, tm2, tn, tscls2, lums, GB, zcnsts, r2, lum2, kw2, mc2, rc2, menv, renv, k22, kick)
            if kw2 >= 13 and kw < 13:
                (kw2, mf, m2, m1, ecc, sepf, jorb, vs) = vkick(kw2, mf, m2, m1, ecc, sepf, jorb, vs, kick)
                if ecc > 1.0:
                    sep = sepf
                    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel, z

    if coel:
        mc22 = mc2
        if kw == 4 or kw == 7:
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

        (m01, aj1, m1, tm1, tn, tscls1, lums, GB, r1, lum1, kw, mc1, rc1, menv, renv, k21) = hrdiag(
            m01, aj1, m1, tm1, tn, tscls1, lums, GB, zcnsts, r1, lum1, kw, mc1, rc1, menv, renv, k21, kick)
        jspin1 = oorb * (k21 * r1 * r1 * (m1 - mc1) + K3 * rc1 * rc1 * mc1)
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
        jspin1 = ospin1 * (k21 * r1 * r1 * (m1 - mc1) + K3 * rc1 * rc1 * mc1)
        jspin2 = ospin2 * (k22 * r2 * r2 * (m2 - mc2) + K3 * rc2 * rc2 * mc2)

    sep = sepf
    return m01, m1, mc1, aj1, jspin1, kw1, m02, m2, mc2, aj2, jspin2, kw2, ecc, sep, jorb, coel, z


