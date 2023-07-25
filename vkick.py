import numpy as np
from const import bhflag, SNtype, yearsc
from numba import njit
import random


@njit
def vkick(kw, m1, m1n, m2, ecc, sep, jorb, vs, kick):
    # 设置常量
    rsunkm = 6.96e5
    # Conversion factor to ensure velocities are in km/s using mass and radius in solar units.
    gmrkm = 1.906125e5

    # 初始化局部变量
    u1 = 0
    theta = 0
    r = 0
    v = np.zeros((1, 5)).flatten()

    # 对电子俘获型超新星、铁核坍缩型超新星诞生的中子星速度踢的麦克斯韦分布, 有不同的sigma值
    if 1.299 < m1n <= 1.301:
        sigma = 30.0
    else:
        sigma = 265.0

    for k in range(1, 4):
        vs[k] = 0.0

    # Find the initial separation by randomly choosing a mean anomaly.
    if sep > 0.0 and ecc >= 0.0:
        xx = random.random()
        mm = 2 * np.pi * xx
        em = mm
        dif = em - ecc * np.sin(em) - mm
        while abs(dif / mm) > 1e-4:
            der = 1.0 - ecc * np.cos(em)
            Del = dif / der
            em = em - Del
            dif = em - ecc * np.sin(em) - mm
        r = sep * (1.0 - ecc * np.cos(em))
        # Find the initial relative velocity vector.
        salpha = np.sqrt((sep * sep * (1.0 - ecc * ecc)) / (r * (2.0 * sep - r)))
        calpha = (-1.0 * ecc * np.sin(em)) / np.sqrt(1.0 - ecc * ecc * np.cos(em) * np.cos(em))
        vr2 = gmrkm * (m1 + m2) * (2.0 / r - 1.0 / sep)
        vr = np.sqrt(vr2)
    else:
        vr = 0.0
        vr2 = 0.0
        salpha = 0.0
        calpha = 0.0

    # Generate Kick Velocity using Maxwellian Distribution (Phinney 1992).
    # Use Henon's method for pairwise components (Douglas Heggie 22/5/97).
    for k in range(1, 3):
        u1 = random.random()
        u2 = random.random()
        # Generate two velocities from polar coordinates S & THETA.
        s = sigma * np.sqrt(-2.0 * np.log(1.0 - u1))
        theta = 2 * np.pi * u2
        v[2 * k - 1] = s * np.cos(theta)
        v[2 * k] = s * np.sin(theta)

    vk2 = v[1] ** 2 + v[2] ** 2 + v[3] ** 2
    vk = np.sqrt(vk2)

    # bhflag(Ture) allows velocity kick at BH formation.
    if (kw == 14 and not bhflag) or kw < 0:
        vk2 = 0.0
        vk = 0.0
        if kw < 0:
            kw = 13

    # 对于黑洞, 受到的速度踢在中子星的基础上乘上一个回落因子
    if kw == 14:
        vk2 = vk2 * (1.0 - kick.f_fb) ** 2.0
        vk = np.sqrt(vk2)

    # 对于 stochastic SN, 速度踢服从一定的正态分布(高斯分布)
    if SNtype == 3:
        vk = abs(random.gauss(kick.meanvk, kick.sigmavk))
        vk2 = vk ** 2

    sphi = -1.0 + 2.0 * u1
    phi = np.arcsin(sphi)
    cphi = np.cos(phi)
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    if sep > 0.0 and ecc >= 0.0:
        # Determine the magnitude of the new relative velocity.
        vn2 = vk2 + vr2 - 2.0 * vk * vr * (ctheta * cphi * salpha - stheta * cphi * calpha)
        # Calculate the new semi-major axis.
        sep = 2.0 / r - vn2 / (gmrkm * (m1n + m2))
        sep = 1.0 / sep
        # Determine the magnitude of the cross product of the separation vector and the new relative velocity.
        v1 = vk2 * sphi * sphi
        v2 = (vk * ctheta * cphi - vr * salpha) ** 2
        hn2 = r * r * (v1 + v2)
        # Calculate the new eccentricity.
        ecc2 = 1.0 - hn2 / (gmrkm * sep * (m1n + m2))
        ecc2 = max(ecc2, 0.0)
        ecc = np.sqrt(ecc2)
        # Calculate the new orbital angular momentum taking care to convert hn to units of Rsun^2/yr.
        jorb = (m1n * m2 / (m1n + m2)) * np.sqrt(hn2) * (yearsc / rsunkm)
        # Determine the angle between the new and old orbital angular momentum vectors.
        cmu = (vr * salpha - vk * ctheta * cphi) / np.sqrt(v1 + v2)
        mu = np.arccos(cmu)
    # Calculate the components of the velocity of the new centre-of-mass.
    if ecc <= 1.0:
        mx1 = vk * m1n / (m1n + m2)
        mx2 = vr * (m1 - m1n) * m2 / ((m1n + m2) * (m1 + m2))
        vs[1] = mx1 * ctheta * cphi + mx2 * salpha
        vs[2] = mx1 * stheta * cphi + mx2 * calpha
        vs[3] = mx1 * sphi
    # Calculate the relative hyperbolic velocity at infinity (simple method).
    else:
        sep = r / (ecc - 1.0)
        mu = np.arccos(1.0 / ecc)
        vr2 = gmrkm * (m1n + m2) / sep
        vr = np.sqrt(vr2)
        vs[1] = vr * np.sin(mu)
        vs[2] = vr * np.cos(mu)
        vs[3] = 0.0
        ecc = min(ecc, 99.99)

    return kw, m1, m1n, m2, ecc, sep, jorb, vs
