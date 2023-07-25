import numpy as np
from star import star
from zfuncs import mcheif,mcagbf,mheif,mbagbf,lum_to_mc_gb,mc_to_lum_gb,lbgbf,lbgbdf
from utils import conditional_njit
from const import tiny


@conditional_njit()
def gntage(mc, mt, kw, zcnsts, m0, aj):
    # 设置常数
    macc = 0.00001
    lacc = 0.0001
    flag1 = True    # 仅供程序中跳转功能用（用来实现 Fortran 中的 goto 功能）
    flag2 = True    # 仅供程序中跳转功能用
    flag3 = True    # 仅供程序中跳转功能用

    jmax = 30

    if kw == 4:
        mcy = mcheif(zcnsts.zpars[2], zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
        if mc <= mcy:
            kw = 3

    if kw == 3:
        mcy = mcheif(zcnsts.zpars[3], zcnsts.zpars[2], zcnsts.zpars[9], zcnsts)
        if mc >= mcy:
            kw = 4
            aj = 0

    if kw == 6:
        mcy = 0.440 * 2.250 + 0.4480
        if mc > mcy:
            mcx = (mc + 0.350) / 0.7730
        elif mc >= 0.8:
            mcx = (mc - 0.4480) / 0.440
        else:
            mcx = mc
        m0 = mbagbf(mcx, zcnsts)
        if m0 < tiny:
            kw = 14
        else:
            (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
            aj = tscls[13]

    if kw == 5:
        m0 = mbagbf(mc, zcnsts)
        if m0 < tiny:
            kw = 14
        else:
            (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
            aj = tscls[2] + tscls[3]

    while kw == 4:
        if aj < 0.0 or aj > 1.0:
            aj = 0.0
        mcy = mcagbf(zcnsts.zpars[2], zcnsts)
        if mc >= mcy:
            mmin = mbagbf(mc, zcnsts)
        else:
            mmin = zcnsts.zpars[2]
        mmax = mheif(mc, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
        if aj < tiny:
            m0 = mmax
            flag1 = False
        elif aj >= 1.0:
            m0 = mmin
            flag1 = False
        if flag1:
            fmid = (1.0 - aj) * mcheif(mmax, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts) + aj * mcagbf(mmax, zcnsts) - mc
            f = (1.0 - aj) * mcheif(mmin, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts) + aj * mcagbf(mmin, zcnsts) - mc
            if f * fmid >= 0.0:
                kw = 3
                break
            m0 = mmin
            dm = mmax - mmin
            for j in range(1, jmax+1):
                dm = 0.50 * dm
                mmid = m0 + dm
                fmid = (1.0 - aj) * mcheif(mmid, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts) + aj * mcagbf(mmid,
                                                                                                          zcnsts) - mc
                if fmid < 0.0:
                    m0 = mmid
                if abs(dm) < macc or abs(fmid) < tiny:
                    break
                if j == jmax:
                    m0 = mt
                    aj = 0.0
        (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
        aj = tscls[2] + aj * tscls[3]
        break

    if kw == 3:
        mcy = mcheif(zcnsts.zpars[3],zcnsts.zpars[2],zcnsts.zpars[9], zcnsts)
        if mc >= mcy:
            mc = 0.99 * mcy
        mcx = mcheif(zcnsts.zpars[2],zcnsts.zpars[2],zcnsts.zpars[9], zcnsts)
        if mc > mcx:
            m0 = mheif(mc,zcnsts.zpars[2],zcnsts.zpars[9], zcnsts)
        else:
            m0 = zcnsts.zpars[2]
            (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
            lum = mc_to_lum_gb(mc,GB)
            j = 0
            while True:
                dell = lbgbf(m0, zcnsts) - lum
                if abs(dell / lum) <= lacc:
                    break
                derl = lbgbdf(m0, zcnsts)
                m0 = m0 - dell / derl
                j = j + 1
                if j == jmax:
                    m0 = zcnsts.zpars[2]
                    m0 = max(m0, mt)
                    break
        (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
        aj = tscls[1] + 1.0e-6 * (tscls[2] - tscls[1])

    if kw == 8 or kw == 9:
        kw = 8
        mmin = mc
        (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
        mcx = lum_to_mc_gb(lums[2], GB, lums[6])
        if mcx >= mc:
            m0 = mt
            flag2 = False
        if flag2:
            f = mcx - mc
            mmax = mt
            for j in range(1, jmax+1):
                (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
                mcy = lum_to_mc_gb(lums[2], GB, lums[6])
                if mcy > mc:
                    break
                mmax = 2.0 * mmax
                if j == jmax:
                    m0 = mt
                    flag3 = False
        while flag2 and flag3:
            fmid = mcy - mc
            if f * fmid >= 0.0:
                m0 = mt
                break
            m0 = mmin
            dm = mmax - mmin
            for j in range(1, jmax+1):
                dm = 0.5 * dm
                mmid = m0 + dm
                (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
                mcy = lum_to_mc_gb(lums[2], GB, lums[6])
                fmid = mcy - mc
                if fmid < 0.0:
                    m0 = mmid
                if abs(dm) < macc or abs(fmid) < tiny:
                    break
                if j == jmax:
                    m0 = mt
                    break
            break
        (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
        aj = tm + 1e-10 * tm

    if kw == 14:
        kw = 4
        m0 = mt
        mcy = mcagbf(m0, zcnsts)
        aj = mc / mcy
        (tm, tn, tscls, lums, GB) = star(kw, m0, mt, zcnsts)
        if m0 <= zcnsts.zpars[2]:
            mcx = lum_to_mc_gb(lums[4], GB, lums[6])
        else:
            mcx = mcheif(m0, zcnsts.zpars[2], zcnsts.zpars[10], zcnsts)
        mc = mcx + (mcy - mcx) * aj
        aj = tscls[2] + aj * tscls[3]

    return mc, mt, kw, m0, aj
