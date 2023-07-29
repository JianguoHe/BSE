import random
import numpy as np
from const import G, Msun, Rsun
from const import SNtype, yearsc, mxns
from const import rng_MA, rng_vkick_maxwell
from utils import conditional_njit, solve_equation


# 超新星kick速度
@conditional_njit()
def SN_kick(kw, m1, m1_new, m2, ecc, sep, kick, index):
    # 设置常量
    rsunkm = 6.96e5
    G_new = G * Msun / Rsun / 1e10    # 计算轨道速度vorb2 = GM(2/r-1/a)时的单位化简【质量Msun, 长度Rsun, 速度km/s】

    # 平均近点角(mean anomaly),在(0-2pi之间均匀分布)
    M = rng_MA[index]

    # 求解超新星爆炸前的偏近点角(Eccentric Anomaly)
    E = solve_equation(a=ecc, b=M, initial_guess=M)

    # 超新星爆炸前的双星轨道间距, 以及对应的矢量
    r = sep * (1.0 - ecc * np.cos(E))
    r_vector = np.array([0, r, 0])

    # 超新星爆炸前的轨道速度大小
    v_orb = np.sqrt(G_new * (m1 + m2) * (2.0 / r - 1.0 / sep))

    # 超新星爆炸前的轨道速度矢量和轨道间距矢量之间的夹角alpha
    sin_alpha = np.sqrt((sep ** 2 * (1.0 - ecc ** 2)) / (r * (2.0 * sep - r)))
    cos_alpha = (-1.0 * ecc * np.sin(E)) / np.sqrt(1.0 - ecc * ecc * np.cos(E) * np.cos(E))

    # 超新星爆炸前的轨道速度矢量
    v_orb_vector = np.array([-v_orb * sin_alpha, v_orb * cos_alpha, 0])

    # 对于rapid/delayed SN, natal kick服从麦克斯韦分布
    if SNtype == 1 or SNtype == 2:
        # 对电子俘获型超新星、铁核坍缩型超新星诞生的中子星速度踢的麦克斯韦分布, 有不同的sigma值
        sigma = 30.0 if 1.299 < m1_new <= 1.301 else 265.0
        # 超新星爆炸后的kick速度向量及模值
        v_kick_vector = sigma * rng_vkick_maxwell[index]
        v_kick_magnitude = np.linalg.norm(v_kick_vector)
        # 计算kick的极角和方位角
        theta = np.arcsin(v_kick_vector[2] / v_kick_magnitude)
        phi = np.arctan2(v_kick_vector[1], v_kick_vector[0])
        # 对于黑洞, 受到的速度踢在中子星的基础上乘上一个回落因子
        if kw == 14:
            v_kick_vector = v_kick_vector * (1.0 - kick.f_fb)
    # 对于 stochastic SN, 速度踢服从一定的正态分布(高斯分布)
    elif SNtype == 3:
        v_kick_vector = np.random.normal(kick.meanvk, kick.sigmavk, size=(3,))
    else:
        raise ValueError('Please give the proper supernova model')

    # 超新星爆炸后的新速度矢量(原轨道速度+kick速度)
    v_new_vector = v_orb_vector + v_kick_vector

    # 计算新的椭圆轨道半长轴
    term = 2 / r - np.linalg.norm(v_new_vector) ** 2 / (G_new * (m1_new + m2))
    sep_new = 1 / term

    # 爆炸后的比轨道角动量矢量(specific angular momentum)
    h_new_vector = np.cross(r_vector, v_new_vector)

    # 计算新的偏心率
    ecc_new2 = 1.0 - np.linalg.norm(h_new_vector) ** 2 / (G_new * (m1_new + m2) * sep_new)
    if ecc_new2 < 0:
        raise ValueError('The square of orbital eccentricity after SN is negative!')

    ecc_new = np.sqrt(ecc_new2)
    if ecc_new > 1.0:
        ecc_new = min(ecc_new, 99.99)
        sep_new = r / (ecc_new - 1.0)

    jorb = (m1_new * m2 / (m1_new + m2)) * np.linalg.norm(h_new_vector) * (yearsc / rsunkm)

    return ecc_new, sep_new, jorb


# 对于各种超新星模型, 输出不同的致密星类型(NS/BH)和当前质量
# 输入变量: mt 为当前质量, mc 为SN爆发前的CO核质量, mcbagb 为bagb时的核质量(包括He+CO核)
@conditional_njit()
def SN_remnant(self, mcbagb):
    if SNtype == 1:     # rapid SN, origin from Fryer et al. 2012, ApJ, 749, 91
        mproto = 1.0
        if self.mass_core < 2.5:
            mfb = 0.2
        elif 2.5 <= self.mass_core < 6:
            mfb = 0.286 * self.mass_core - 0.514
        elif 6 <= self.mass_core < 7:
            mfb = self.mass - mproto
        elif 7 <= self.mass_core < 11:
            a1 = 0.25 - 1.275 / (self.mass - mproto)
            b1 = -11.0 * a1 + 1.0
            mfb = (self.mass - mproto) * (a1 * self.mass_core + b1)
        else:
            mfb = self.mass - mproto
        self.f_fb = mfb / (self.mass - mproto)
        mrem_bar = mfb + mproto                                         # 遗迹重子质量
        mrem1 = -6.6667 + 0.6667 * (100 + 30 * mrem_bar) ** 0.5         # 中子星引力质量
        mrem2 = 0.9 * mrem_bar                                          # 黑洞引力质量
        # 中子星
        if mrem1 <= mxns:
            self.type = 13
            self.mass = mrem1
        # 黑洞
        else:
            self.type = 14
            self.mass = mrem2
    elif SNtype == 2:     # delayed SN, origin from Fryer et al. 2012, ApJ, 749, 91
        if self.mass_core <= 3.5:
            mproto = 1.2
        elif 3.5 < self.mass_core <= 6.0:
            mproto = 1.3
        elif 6.0 < self.mass_core <= 11.0:
            mproto = 1.4
        else:
            mproto = 1.6
        if self.mass_core <= 2.5:
            mfb = 0.2
        elif 2.5 < self.mass_core <= 3.5:
            mfb = 0.5 * self.mass_core - 1.05
        elif 3.5 < self.mass_core <= 11.0:
            a2 = 0.133 - 0.093 / (self.mass - mproto)
            b2 = -11.0 * a2 + 1.0
            mfb = (self.mass - mproto) * (a2 * self.mass_core + b2)
        else:
            mfb = self.mass - mproto
        self.f_fb = mfb / (self.mass - mproto)
        mrem_bar = mfb + mproto                                         # 遗迹重子质量
        mrem1 = -6.6667 + 0.6667 * (100 + 30 * mrem_bar) ** 0.5         # 中子星引力质量
        mrem2 = 0.9 * mrem_bar                                          # 黑洞引力质量
        # 中子星
        if mrem1 <= mxns:
            self.type = 13
            self.mass = mrem1
        # 黑洞
        else:
            self.type = 14
            self.mass = mrem2
    else:               # stochastic SN, origin from Mandel et al. 2020, MNRAS 499, 3214–3221
        m11 = 2.0
        m22 = 3.0
        m33 = 7.0
        m44 = 8.0
        meanbh = 0.8
        sigmabh = 0.5
        p1 = random.random()
        p2 = random.random()
        # 计算黑洞形成时物质完全回落(complete fallback)的概率
        if m11 <= self.mass_core < m44:
            pcf = (self.mass_core - m11) / (m44 - m11)
        else:
            pcf = 1.0
        # 中子星
        if self.mass_core < m11:
            mean0 = 1.2
            sigma0 = 0.02
            self.type = 13
            self.mass = max(1.13, random.gauss(mean0, sigma0))
        # 中子星或黑洞
        elif m11 <= self.mass_core < m33:
            # 计算遗迹是黑洞的概率
            pbh = (self.mass_core - m11) / (m33 - m11)
            # 黑洞
            if p1 <= pbh:
                self.type = 14
                # 完全回落
                if p2 <= pcf:
                    self.mass = mcbagb
                # 不完全回落
                else:
                    self.mass = max(2.0, random.gauss(meanbh * self.mass_core, sigmabh))
            # 中子星
            else:
                self.type = 13
                if m11 <= self.mass_core < m22:
                    mean0 = 1.4 + 0.5 * (self.mass_core - m11) / (m22 - m11)
                    sigma0 = 0.05
                else:
                    mean0 = 1.4 + 0.4 * (self.mass_core - m22) / (m33 - m22)
                    sigma0 = 0.05
                self.mass = max(1.13, random.gauss(mean0, sigma0))
        # 黑洞
        else:
            self.type = 14
            # 完全回落
            if p2 <= pcf:
                self.mass = mcbagb
            # 不完全回落
            else:
                self.mass = max(2.0, random.gauss(meanbh * self.mass_core, sigmabh))
        # 对于 stochastic SN, 速度踢服从一定的正态分布(高斯分布)
        if self.type == 13:
            self.meanvk = 520.0 * (self.mass_core - self.mass) / self.mass
            self.sigmavk = 0.3 * self.meanvk
        elif self.type == 14:
            self.meanvk = 200.0 * max((self.mass_core - self.mass) / self.mass, 0.0)
            self.sigmavk = 0.3 * self.meanvk
    return 0





