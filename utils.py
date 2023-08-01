import random
import numpy as np
from numba import njit
from numba.experimental import jitclass
from const_new import mch, yeardy, period_to_sep, Msun, Rsun, yearsc
from const_new import IMF_scheme, mb_model, gamma_mb
from const_new import m1_min, m1_max, m2_min, m2_max, sep_min, sep_max, num_evolve
from const_new import njit_enabled, jitclass_enabled


# 定义一个njit条件装饰器
def conditional_njit():
    def decorator(func):
        if njit_enabled:
            return njit(func)
        else:
            return func
    return decorator


# 定义一个jitclass条件装饰器
def conditional_jitclass(spec):
    def decorator(cls):
        if jitclass_enabled:
            return jitclass(spec)(cls)
        else:
            return cls
    return decorator


# 判定是否考虑磁制动
@conditional_njit()
def mb_judgment(mass, kw):
    # if mass > 0.35 and kw <= 9:
    if (0.35 < mass < 1.25 and kw <= 1) or 2 <= kw <= 9:
        return True
    else:
        return False


# 计算有明显对流包层的恒星因磁制动损失的自旋角动量, 包括主序星(M < 1.25)、靠近巨星分支的HG恒星以及巨星, 不包括完全对流主序星
@conditional_njit()
def magnetic_braking(mass, menv, R, ospin):
    if mb_model == 'Rappaport1983':
        djmb = 3.8e-30 * mass * R ** gamma_mb * ospin ** 3 * Rsun ** 2 / yearsc
    elif mb_model == 'Hurley2002':
        djmb = 5.83e-16 * menv * (R * ospin) ** 3 / mass
    else:
        raise ValueError("Please set the proper magnetic braking model.")
    return djmb


# 恒星的初始质量函数
@conditional_njit()
def xi(M):
    if IMF_scheme == 'Kroupa1993':
        # Kroupa et al. 1993
        if M <= 0.1:
            return 0
        elif 0.1 < M <= 0.5:
            return 0.29056 * M ** -1.3
        elif 0.5 < M <= 1:
            return 0.15571 * M ** -2.2
        else:
            return 0.15571 * M ** -2.7
    elif IMF_scheme == 'Weisz2015':
        # Weisz et al. 2015
        c = 0.2074
        if 0.08 < M <= 0.5:
            return c * M ** -1.3
        elif 0.5 < M <= 1:
            return c * M ** -2.3
        elif 1 <= M <= 100:
            return c * M ** -2.45
        else:
            return 0
    else:
        raise ValueError("Please provide an allowed IMF scheme.")


# 计算每组参数对总数的贡献Wb，即权重
@conditional_njit()
def Wb_binary(M1, M2):
    n_grid = num_evolve ** (1 / 3)
    delta_lnM1 = np.log(m1_max / m1_min) / (n_grid - 1)
    delta_lnM2 = np.log(m2_max / m2_min) / (n_grid - 1)
    delta_lna = np.log(sep_max / sep_min) / (n_grid - 1)
    Phi_lnM1 = M1 * xi(M1)
    varphi_lnM2 = M2 / M1
    Psi_lna = 1 / np.log(sep_max / sep_min)
    Wb = Phi_lnM1 * varphi_lnM2 * Psi_lna * delta_lnM1 * delta_lnM2 * delta_lna
    return Wb


# 银河系内的恒星形成率
@conditional_njit()
def SFR_Galaxy():
    return 3e6


# 计算 M31 星系在某一时刻的恒星形成率
@conditional_njit()
def SFR_M31(Time):
    times = np.array(
        [0, 6.1e3, 7.7e3, 9e3, 1e4, 1.08e4, 1.15e4, 1.2e4, 1.24e4, 1.27e4, 1.3e4, 1.321e4, 1.337e4, 1.35e4, 1.36e4,
         1.4e4])
    SFR_values = np.array([18, 3.9, 0.36, 4.8, 5.7, 1.65, 18.6, 4.8, 9, 5.4, 4.8, 2.94, 2.16, 0.81, 0.39]) * 1e6
    index = np.searchsorted(times, Time, side='right') - 1
    SFR = SFR_values[index]
    return SFR  # unit: /Myr


# 计算某个金属丰度在 M31 星系历史中对应的时刻（单位：Gyr）
@conditional_njit()
def M31_History_Time(z):
    if 0.00022233 <= z <= 0.00813205:
        return 252.853507 * z - 0.05621717
    elif 0.00813205 < z <= 0.01999862:
        return 674.1628264 * z - 3.4823253
    elif 0.01999862 < z <= 0.0223795:
        return 2100.0799346 * z - 31.9986978


# 根据[Fe/H]计算金属丰度z (Bertelli et al. 1994)
@conditional_njit()
def cal_z(Fe_H):
    return 10 ** (0.977 * Fe_H - 1.699)


# 估算洛希瓣半径
@conditional_njit()
def rochelobe(q):
    p = q ** (1 / 3)
    rl_relative_a = 0.49 * p * p / (0.6 * p * p + np.log(1 + p))
    return rl_relative_a


# 求解开普勒方程中的偏近点角(Eccentric Anomaly), e 为离心率, M 为平均近点角(mean anomaly)
@conditional_njit()
def equation_EA(x, e, M):
    return x - e * np.sin(x) - M


# 牛顿法(一种求解方程的迭代数值方法, 适用于求解一般的非线性方程)
@conditional_njit()
def solve_equation(a, b, initial_guess=0, max_iterations=1000000, tolerance=1e-5):
    x = initial_guess
    for _ in range(max_iterations):
        f = equation_EA(x, a, b)
        if abs(f) < tolerance:
            return x
        df = 1 - a * np.cos(x)
        x = x - f / df
    raise ValueError('The iterations exceed the maximum when solve the eccentric anomaly of the system.')


# 计算白矮星半径
def cal_WD_radius(m):
    return 0.0115 * np.sqrt((mch / m) ** (2 / 3) - (m / mch) ** (2 / 3))


# 计算双星轨道间距
def cal_separation(period, mass1, mass2):
    period = period / yeardy
    separation = period_to_sep * (period ** 2 * (mass1 + mass2)) ** (1 / 3)
    return separation


# 计算双星的轨道角动量
def cal_jorb(m1, m2, tb, sep, ecc):
    oorb = 2 * np.pi / (tb * 24 * 3600)
    jorb = m1 * m2 / (m1 + m2) * np.sqrt(1 - ecc ** 2) * sep * sep * oorb
    jorb = jorb * Msun * Rsun ** 2
    return jorb


# 计算银河系源的距离
def get_dL_galaxy():
    r = random.uniform(0, 15)
    alpha = random.uniform(0, np.pi)
    dL_kpc = np.sqrt(64.0 + r ** 2 - 16.0 * r * np.cos(alpha))
    return dL_kpc


# 计算M31源的距离
def get_dL_M31():
    dL_kpc = 780
    return dL_kpc


# 目标源的条件即需要筛选的源
def select(df):
    # 保证双星没有瓦解
    condition_binary = (df['tb'] < 1e5) & (0 <= df['ecc']) & (df['ecc'] < 1)

    # 目标源的筛选条件
    condition_BH_BH = (df['kw1'] == 14) & (df['kw2'] == 14)
    condition_BH_NS = ((df['kw1'] == 14) & (df['kw2'] == 13)) | ((df['kw2'] == 14) & (df['kw1'] == 13))
    condition_BH_WD = ((df['kw1'] == 14) & (10 <= df['kw2']) & (df['kw2'] <= 12)) | \
                      ((df['kw2'] == 14) & (10 <= df['kw1']) & (df['kw1'] <= 12))
    condition_NS_NS = (df['kw1'] == 13) & (df['kw2'] == 13)
    condition_NS_WD = ((df['kw1'] == 13) & (10 <= df['kw2']) & (df['kw2'] <= 12)) | \
                      ((df['kw2'] == 13) & (10 <= df['kw1']) & (df['kw1'] <= 12))
    condition_WD_WD = (10 <= df['kw1']) & (df['kw1'] <= 12) & (10 <= df['kw2']) & (df['kw2'] <= 12)

    # 挑选目标源
    condition_CS_CS = [condition_BH_BH, condition_BH_NS, condition_BH_WD,
                       condition_NS_NS, condition_NS_WD, condition_WD_WD]

    return [condition & condition_binary for condition in condition_CS_CS]
