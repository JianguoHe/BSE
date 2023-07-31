from numba import float64, int64, types
from numba.experimental import jitclass
import numpy as np
from const_new import gamma_mb, mb_model, yearsc, Zsun, neta, bwind, f_WR, f_LBV, Rsun, Teffsun
from utils import conditional_jitclass
from zcnst import zcnsts_set
from StellarCal import StellarCal
from StellarProp import StellarProp


# Single star class

spec = [
    ('type', float64),                      # steller type
    ('Z', float64),                         # initial mass fraction of metals
    ('mass0', float64),                     # initial mass (solar units)
    ('mass', float64),                      # current mass (solar units)
    ('R', float64),                         # radius (solar units)
    ('L', float64),                         # luminosity (solar units)
    ('dt', float64),                        # evolution timestep
    ('Teff', float64),                      # effective temperature (K)
    ('spin', float64),                      # 自旋角频率(unit: /yr)
    ('jspin', float64),                     # 自旋角动量(unit: Msun * Rsun2 / yr)
    ('rochelobe', float64),                 # 洛希瓣半径(unit: Rsun)
    ('mass_core', float64),                 # in solar units
    ('mass_he_core', float64),              # in solar units
    ('mass_c_core', float64),               # in solar units
    ('mass_o_core', float64),               # in solar units
    ('mass_co_core', float64),              # in solar units
    ('mass_envelop', float64),              # in solar units
    ('radius_core', float64),               # in solar units
    ('radius_he_core', float64),            # in solar units
    ('radius_c_core', float64),             # in solar units
    ('radius_o_core', float64),             # in solar units
    ('radius_co_core', float64),            # in solar units
    ('radius_envelop', float64),            # in solar units
    ('mdot_wind_loss', float64),            # 星风质量损失率
    ('mdot_wind_accrete', float64),         # 星风质量吸积率
    ('jdot_spin_wind', float64),            # 星风提取的自旋角动量
    ('jdot_spin_mb', float64),              # 磁制动提取的自旋角动量
    ('max_time', float64),                  # 最长演化时间          [unit: Myr]
    ('time', float64),                      # 当前的演化时间        [unit: Myr]
    ('age', float64),                       # 当前type的演化时间    [unit: Myr]
    ('max_step', float64),                  # 最大演化步长
    ('step', int64),                        # 当前的演化步长
    ('data', float64[:, :]),                # 存储每个步长的属性
    ('zpars', float64[:]),                  # 与金属丰度相关的常数
    ('msp', float64[:]),                    # 主序分支系数
    ('gbp', float64[:]),                    # 巨星分支系数
    ('tm', float64),                        # 主序时间
    ('tn', float64),                        # 核燃烧时间
    ('tscls', float64[:]),                  # 到达不同阶段的时标
    ('lums', float64[:]),                   # 特征光度
    ('GB', float64[:]),                     # 巨星分支参数
    ('f_fb', float64),                      # 超新星爆炸后回落物质所占比例
    ('meanvk', float64),                    # stochastic模型下Natal Kick服从正态分布均值
    ('sigmavk', float64),                   # stochastic模型下Natal Kick数值标准差
    ('k3', float64),                        # 恒星核的自旋角动量jspin_core=k2*omega*mc*rc**2
    ('k2', float64),                        # 恒星包层的自旋角动量jspin_envelop=k3*omega*me*re**2
]


@conditional_jitclass(spec)
class SingleStar:
    def __init__(self, type, Z, mass, R=0, L=0, dt=1e6, Teff=0, spin=0, jspin=0, rochelobe=0,
                 mass_core=0, mass_he_core=0, mass_c_core=0, mass_o_core=0, mass_co_core=0, mass_envelop=0,
                 radius_core=0, radius_he_core=0, radius_c_core=0, radius_o_core=0, radius_co_core=0,
                 radius_envelop=0, mdot_wind_loss=0, mdot_wind_accrete=0, jdot_spin_wind=0, jdot_spin_mb=0,
                 max_time=10000, time=0, age=0, max_step=20000, step=0):
        self.type = type
        self.Z = Z
        self.mass0 = mass
        self.mass = mass
        self.R = R
        self.L = L
        self.dt = dt
        self.Teff = Teff
        self.spin = spin
        self.jspin = jspin
        self.rochelobe = rochelobe
        self.mass_core = mass_core
        self.mass_he_core = mass_he_core
        self.mass_c_core = mass_c_core
        self.mass_o_core = mass_o_core
        self.mass_co_core = mass_co_core
        self.mass_envelop = mass_envelop
        self.radius_core = radius_core
        self.radius_he_core = radius_he_core
        self.radius_c_core = radius_c_core
        self.radius_o_core = radius_o_core
        self.radius_co_core = radius_co_core
        self.radius_envelop = radius_envelop
        self.mdot_wind_loss = mdot_wind_loss
        self.mdot_wind_accrete = mdot_wind_accrete
        self.jdot_spin_wind = jdot_spin_wind
        self.jdot_spin_mb = jdot_spin_mb
        self.max_time = max_time
        self.time = time
        self.age = age
        self.max_step = max_step
        self.step = step
        self.data = np.zeros((max_step, 30))
        self.zpars = np.zeros(20)
        self.msp = np.zeros(200)
        self.gbp = np.zeros(200)
        self.tm = 0
        self.tn = 0
        self.tscls = np.zeros(20)
        self.lums = np.zeros(10)
        self.GB = np.zeros(10)
        self.f_fb = 0
        self.meanvk = 0
        self.sigmavk = 0
        self.k3 = 0.21
        self.k2 = 0
        zcnsts_set(self)            # 设置金属丰度相关常数

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    演化单星
    # ------------------------------------------------------------------------------------------------------------------
    def evolve(self):
        # 首先保存单星的初始属性
        self.save()
        # 确定恒星的不同演化阶段的时标、标志性光度、巨星分支参数
        StellarCal(self)
        # 确定恒星的光度、半径、核质量、核半径、对流包层质量/半径/转动惯量系数
        StellarProp(self)
        # 跳过
        pass

    # 计算表面温度
    def cal_Teff(self):
        self.Teff = Teffsun * (self.L / self.R ** 2.0) ** (1.0 / 4.0)

    # 更新质量和自旋角动量
    def reset(self, dt):
        self.mass = self.mass + (self.mdot_wind_accrete + self.mdot_wind_accrete) * dt
        self.jspin = self.jspin + (self.jdot_spin_mb + self.jdot_spin_wind) * dt

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    保存当前属性
    # ------------------------------------------------------------------------------------------------------------------
    def save(self):
        self.data[self.step, 0] = self.time
        self.data[self.step, 1] = self.type
        self.data[self.step, 2] = self.mass
        self.data[self.step, 3] = self.mass_core
        self.data[self.step, 4] = self.mass_envelop
        self.data[self.step, 5] = np.log10(self.R)
        self.data[self.step, 6] = np.log10(self.L)
        self.data[self.step, 7] = self.spin
        self.data[self.step, 8] = self.Teff

    # ------------------------------------------------------------------------------------------------------------------
    #                                                      磁制动
    # ------------------------------------------------------------------------------------------------------------------
    # 考虑磁制动的影响
    def magnetic_braking(self):
        # 计算有明显对流包层的恒星因磁制动损失的自旋角动量, 包括主序星(M < 1.25)、靠近巨星分支的HG恒星以及巨星, 不包括完全对流主序星
        if self.mass > 0.35 and self.type < 10:
            if mb_model == 'Rappaport1983':
                self.jdot_spin_mb = -3.8e-30 * self.mass * self.R ** gamma_mb * self.spin ** 3 * Rsun ** 2 / yearsc
            elif mb_model == 'Hurley2002':
                self.jdot_spin_mb = -5.83e-16 * self.mass_envelop * (self.R * self.spin) ** 3 / self.mass
            else:
                raise ValueError("Please set the proper magnetic braking model.")

        # 限制最大3%的磁制动损失的角动量。这可以保证迭代次数不会超过最大值20000, 当然2%也不会影响演化结果
        if self.jdot_spin_mb > 0:
            dt_max = 0.03 * self.jspin / abs(self.jdot_spin_mb)
            self.dt = min(self.dt, dt_max)

    # ------------------------------------------------------------------------------------------------------------------
    #                                               总的星风质量损失(Msun/yr)
    # ------------------------------------------------------------------------------------------------------------------
    # 计算总的星风质量损失(Hurley模型)
    def cal_mdot_wind_Hurley(self, ecc):
        mdot_NJ = self.cal_mdot_NJ()
        mdot_KR = self.cal_mdot_KR(ecc=ecc)
        mdot_VW = self.cal_mdot_VW()
        mdot_WR = self.cal_mdot_WR()
        mdot_LBV_Hurley = self.cal_mdot_LBV_Hurley()

        if 0 <= self.type <= 6:
            mdot_wind = max(mdot_NJ, mdot_KR, mdot_VW, mdot_WR) + mdot_LBV_Hurley
        elif 7 <= self.type <= 9:
            mdot_wind = max(mdot_NJ, mdot_KR, mdot_WR)
        else:
            mdot_wind = 0
        return mdot_wind

    # 计算总的星风质量损失(Belczynski模型)
    def cal_mdot_wind_Belczynski(self, ecc):
        mdot_OB = self.cal_mdot_OB()
        mdot_KR = self.cal_mdot_KR(ecc=ecc)
        mdot_WR = self.cal_mdot_WR(z_dependent=True)
        mdot_LBV_Belczynski = self.cal_mdot_LBV_Belczynski()

        # LBV星
        if mdot_LBV_Belczynski > 0:
            return mdot_LBV_Belczynski
        # 氦星
        if 7 <= self.type <= 9:
            return max(mdot_KR, mdot_WR)
        # OB星
        if mdot_OB > 0:
            return mdot_OB
        # 其他情况
        else:
            return self.cal_mdot_wind_Hurley(ecc=ecc)

    # -------------------------------------------------------------------------------------------------------------------
    #                                              各种星风质量损失(Msun/yr)
    # -------------------------------------------------------------------------------------------------------------------
    # calculate mass loss rate for massive stars (L > 4000Lsun) over the entire HRD
    # Nieuwenhuijzen & de Jager 1990, A&A, 231, 134
    def cal_mdot_NJ(self):
        if self.L > 4000.0:
            term1 = 9.631e-15 * min(1.0, (self.L - 4000.0) / 500.0)
            term2 = self.R ** 0.81 * self.L ** 1.24 * self.mass ** 0.16 * (self.Z / Zsun) ** 0.5
            mdot_NJ = term1 * term2
        else:
            mdot_NJ = 0
        return mdot_NJ

    # Calculate mass loss rate for massive OB stars using the Vink et al. 2001 prescription
    # Vink et al. 2001, eqs 24 & 25; Belczynski et al. 2010, eqs 6 & 7
    def cal_mdot_OB(self):
        if 1.25e4 < self.Teff <= 2.5e4:
            term1 = - 6.688 + 2.21 * np.log10(self.L / 1.0e5)
            term2 = - 1.339 * np.log10(self.mass / 30.0) - 1.601 * np.log10(1.3 / 2.0)
            term3 = 1.07 * np.log10(self.Teff / 2.0e4) + 0.85 * np.log10(self.Z / Zsun)
            mdot_OB = 10 ** (term1 + term2 + term3)
        elif 2.5e4 < self.Teff <= 5.0e4:
            term1 = - 6.697 + 2.194 * np.log10(self.L / 1.0e5) - 1.313 * np.log10(self.mass / 30.0)
            term2 = - 1.226 * np.log10(2.6 / 2.0) + 0.933 * np.log10(self.Teff / 4.0e4)
            term3 = - 10.92 * np.log10(self.Teff / 4.0e4) ** 2 + 0.85 * np.log10(self.Z / Zsun)
            mdot_OB = 10 ** (term1 + term2 + term3)
        else:
            mdot_OB = 0
        return mdot_OB

    # calculate mass loss rate on the GB and beyond
    # Hurley et al. 2000, eq 106 (based on a prescription taken from Kudritzki & Reimers, 1978, A&A, 70, 227)
    def cal_mdot_KR(self, ecc):
        if 2 <= self.type <= 9:
            mdot_KR = neta * 4.0e-13 * self.R * self.L / self.mass
            # 考虑 mdot_KR 受潮汐增强(如果应用, 这里可能还需要考虑偏心轨道的情况)
            if self.rochelobe > 0.0:
                rochelobe_periastron = self.rochelobe * (1.0 - ecc)
                mdot_KR = mdot_KR * (1.0 + bwind * (min(0.5, (self.R / rochelobe_periastron))) ** 6)
        else:
            mdot_KR = 0
        return mdot_KR

    # calculate mass loss rate on the AGB based on the Mira pulsation period
    # Hurley et al. 2000, just after eq 106 (from Vassiliadis & Wood, 1993, ApJ, 413, 641)
    def cal_mdot_VW(self):
        if 5 <= self.type <= 6:
            p0 = min(1995, 8.51e-3 * self.R ** 1.94 / self.mass ** 0.9)
            p1 = 100.0 * max(self.mass - 2.5, 0)
            mdot_VW = min(10.0 ** (-11.4 + 0.0125 * (p0 - p1)), 1.36e-9 * self.L)
        else:
            mdot_VW = 0
        return mdot_VW

    # calculate mass loss of Wolf–Rayet like star with small H-envelope mass
    # Hurley et al. 2000, just after eq 106 (taken from Hamann, Koesterke & Wessolowski 1995, Hamann & Koesterke 1998)
    # Belczynski et al. 2010, eq 9 when z_dependent is True
    def cal_mdot_WR(self, z_dependent=False):
        lum0 = 7e4
        kap = -0.5
        mu = (self.mass - self.mass_core) / self.mass * min(5.0, max(1.2, (self.L / lum0) ** kap))
        mdot_WR = f_WR * 1.0e-13 * self.L ** 1.5 * (1.0 - mu) if mu < 1.0 else 0
        mdot_WR = mdot_WR * (self.Z / Zsun) ** 0.86 if z_dependent else mdot_WR
        return mdot_WR

    # Calculate LBV-like mass loss rate for stars beyond the Humphreys-Davidson limit (Humphreys & Davidson 1994)
    # Hurley+ 2000 Section 7.1 a few equation after Eq. 106 (Equation not labelled)
    def cal_mdot_LBV_Hurley(self):
        HD = 1.0e-5 * self.R * self.L ** 0.5
        if self.L > 6.0e5 and HD > 1.0:
            mdot_LBV_Hurley = 0.1 * (HD - 1.0) ** 3 * (self.L / 6.0e5 - 1.0)
        else:
            mdot_LBV_Hurley = 0
        return mdot_LBV_Hurley

    # Calculate LBV-like mass loss rate for stars beyond the Humphreys-Davidson limit (Humphreys & Davidson 1994)
    # Belczynski et al. 2010, eq 8
    def cal_mdot_LBV_Belczynski(self):
        HD = 1.0e-5 * self.R * self.L ** 0.5
        if self.L > 6.0e5 and HD > 1.0:
            mdot_LBV_Belczynski = f_LBV * 1.0e-4
        else:
            mdot_LBV_Belczynski = 0
        return mdot_LBV_Belczynski

    # 估算零龄主序光度 Lzams （from Tout et al., 1996, MNRAS, 281, 257）
    def lzamsf(self):
        mx = np.sqrt(self.mass0)
        lzams = (self.msp[1] * self.mass0 ** 5 * mx + self.msp[2] * self.mass0 ** 11) / (
                self.msp[3] + self.mass0 ** 3 + self.msp[4] * self.mass0 ** 5 + self.msp[5] * self.mass0 ** 7 +
                self.msp[6] * self.mass0 ** 8 + self.msp[7] * self.mass0 ** 9 * mx)
        return lzams

    # 估算零龄主序半径 Rzams
    def rzamsf(self, m=0):
        mass = self.mass0 if m == 0 else m
        mx = np.sqrt(mass)
        rzams = ((self.msp[8] * mass ** 2 + self.msp[9] * mass ** 6) * mx + self.msp[10] * mass ** 11 + (
                self.msp[11] + self.msp[12] * mx) * mass ** 19) / (self.msp[13] + self.msp[14] * mass ** 2 + (
                self.msp[15] * mass ** 8 + mass ** 18 + self.msp[16] * mass ** 19) * mx)
        return rzams

    # A function to evaluate the lifetime to the BGB or to Helium ignition if no FGB exists. (JH 24/11/97)
    # [已校验] Hurley_2000: equation 5.1(4)
    def tbgbf(self):
        tbgb = (self.msp[17] + self.msp[18] * self.mass0 ** 4 + self.msp[19] * self.mass0 ** (11 / 2) + self.mass0 ** 7) / (
                self.msp[20] * self.mass0 ** 2 + self.msp[21] * self.mass0 ** 7)
        return tbgb

    # A function to evaluate the derivitive of the lifetime to the BGB
    # (or to Helium ignition if no FGB exists) wrt mass. (JH 24/11/97)
    def tbgbdf(self):
        mx = np.sqrt(self.mass0)
        f = self.msp[17] + self.msp[18] * self.mass0 ** 4 + self.msp[19] * self.mass0 ** 5 * mx + self.mass0 ** 7
        df = 4 * self.msp[18] * self.mass0 ** 3 + 5.5 * self.msp[19] * self.mass0 ** 4 * mx + 7 * self.mass0 ** 6
        g = self.msp[20] * self.mass0 ** 2 + self.msp[21] * self.mass0 ** 7
        dg = 2 * self.msp[20] * self.mass0 + 7 * self.msp[21] * self.mass0 ** 6
        tbgbd = (df * g - f * dg) / (g * g)
        return tbgbd

    # A function to evaluate the derivitive of the lifetime to the BGB
    # (or to Helium ignition if no FGB exists) wrt Z. (JH 14/12/98)
    def tbgdzf(self):
        mx = self.mass0 ** 5 * np.sqrt(self.mass0)
        f = self.msp[17] + self.msp[18] * self.mass0 ** 4 + self.msp[19] * mx + self.mass0 ** 7
        df = self.msp[117] + self.msp[118] * self.mass0 ** 4 + self.msp[119] * mx
        g = self.msp[20] * self.mass0 ** 2 + self.msp[21] * self.mass0 ** 7
        dg = self.msp[120] * self.mass0 ** 2
        tbgdz = (df * g - f * dg) / (g * g)
        return tbgdz

    # A function to evaluate the lifetime to the end of the MS hook as a fraction of the lifetime to the BGB
    # (for those models that have one). Note that this function is only valid for self.mass0 > Mhook.
    # [已校验] Hurley_2000: equation 5.1(7)
    def thook_div_tBGB(self):
        term = 1 - 0.01 * max(self.msp[22] / self.mass0 ** self.msp[23], self.msp[24] + self.msp[25] / self.mass0 ** self.msp[26])
        value = max(0.5, term)
        return value

    # 估算主序末尾的光度
    # [已校验] Hurley_2000: equation 5.1(8)
    def ltmsf(self):
        ltms = (self.msp[27] * self.mass0 ** 3 + self.msp[28] * self.mass0 ** 4 + self.msp[29] * self.mass0 ** (self.msp[32] + 1.8)) / (
                self.msp[30] + self.msp[31] * self.mass0 ** 5 + self.mass0 ** self.msp[32])
        return ltms

    # 估算光度 alpha 系数
    # [已校验] Hurley_2000: equation 5.1.1(19)
    def lalphaf(self):
        mcut = 2.0
        if self.mass0 < 0.5:
            lalpha = self.msp[39]
        elif self.mass0 < 0.7:
            lalpha = self.msp[39] + ((0.3 - self.msp[39]) / 0.2) * (self.mass0 - 0.5)
        elif self.mass0 < self.msp[37]:
            lalpha = 0.3 + ((self.msp[40] - 0.3) / (self.msp[37] - 0.7)) * (self.mass0 - 0.7)
        elif self.mass0 < self.msp[38]:
            lalpha = self.msp[40] + ((self.msp[41] - self.msp[40]) / (self.msp[38] - self.msp[37])) * (self.mass0 - self.msp[37])
        elif self.mass0 < mcut:
            lalpha = self.msp[41] + ((self.msp[42] - self.msp[41]) / (mcut - self.msp[38])) * (self.mass0 - self.msp[38])
        else:
            lalpha = (self.msp[33] + self.msp[34] * self.mass0 ** self.msp[36]) / (self.mass0 ** 0.4 + self.msp[35] * self.mass0 ** 1.9)
        return lalpha

    # 估算光度 beta 系数
    # [已校验] Hurley_2000: equation 5.1.1(20)
    def lbetaf(self):
        lbeta = max(0, self.msp[43] - self.msp[44] * self.mass0 ** self.msp[45])
        if self.mass0 > self.msp[46] and lbeta > 0:
            B = self.msp[43] - self.msp[44] * self.msp[46] ** self.msp[45]
            lbeta = max(0, B - 10 * B * (self.mass0 - self.msp[46]))
        return lbeta

    # 估算光度 neta 系数
    # [已校验] Hurley_2000: equation 5.1.1(18)
    def lnetaf(self):
        if self.mass0 <= 1:
            lneta = 10
        elif self.mass0 >= 1.1:
            lneta = 20
        else:
            lneta = 10 + 100 * (self.mass0 - 1)
        lneta = min(lneta, self.msp[97])
        return lneta

    # A function to evaluate the radius at the end of the MS
    # Note that a safety check is added to ensure Rtms > Rzams when extrapolating the function to low masses. (JH 24/11/97)
    # [已校验] Hurley_2000: equation 5.1(9)
    def rtmsf(self, m=0):
        mass = self.mass0 if m == 0 else m

        if mass <= self.msp[62]:
            rtms = (self.msp[52] + self.msp[53] * mass ** self.msp[55]) / (self.msp[54] + mass** self.msp[56])
            # extrapolated to low mass(M < 0.5)
            rtms = max(rtms, 1.5 * self.rzamsf(mass))
        elif mass >= self.msp[62] + 0.1:
            rtms = (self.msp[57] * mass ** 3 + self.msp[58] * mass ** self.msp[61] + self.msp[59] * mass ** (self.msp[61] + 1.5)) / (
                    self.msp[60] + mass ** 5)
        else:
            rtms = self.msp[63] + ((mass - self.msp[62]) / 0.1) * (self.msp[64] - self.msp[63])

        return rtms

    # 估算半径 alpha 系数
    # [已校验] Hurley_2000: equation 5.1.1(21)
    def ralphaf(self):
        if self.mass0 <= 0.5:
            ralpha = self.msp[73]
        elif self.mass0 <= 0.65:
            ralpha = self.msp[73] + ((self.msp[74] - self.msp[73]) / 0.15) * (self.mass0 - 0.5)
        elif self.mass0 <= self.msp[70]:
            ralpha = self.msp[74] + ((self.msp[75] - self.msp[74]) / (self.msp[70] - 0.65)) * (self.mass0 - 0.65)
        elif self.mass0 <= self.msp[71]:
            ralpha = self.msp[75] + ((self.msp[76] - self.msp[75]) / (self.msp[71] - self.msp[70])) * (self.mass0 - self.msp[70])
        elif self.mass0 <= self.msp[72]:
            ralpha = (self.msp[65] * self.mass0 ** self.msp[67]) / (self.msp[66] + self.mass0 ** self.msp[68])
        else:
            a5 = (self.msp[65] * self.msp[72] ** self.msp[67]) / (self.msp[66] + self.msp[72] ** self.msp[68])
            ralpha = a5 + self.msp[69] * (self.mass0 - self.msp[72])
        return ralpha

    # 估算半径 beta 系数
    # [已校验] Hurley_2000: equation 5.1.1(22)
    def rbetaf(self):
        m2 = 2
        m3 = 16
        if self.mass0 <= 1:
            rbeta = 1.06
        elif self.mass0 <= self.msp[82]:
            rbeta = 1.06 + ((self.msp[81] - 1.06) / (self.msp[82] - 1)) * (self.mass0 - 1)
        elif self.mass0 <= m2:
            b2 = (self.msp[77] * m2 ** (7 / 2)) / (self.msp[78] + m2 ** self.msp[79])
            rbeta = self.msp[81] + ((b2 - self.msp[81]) / (m2 - self.msp[82])) * (self.mass0 - self.msp[82])
        elif self.mass0 <= m3:
            rbeta = (self.msp[77] * self.mass0 ** (7 / 2)) / (self.msp[78] + self.mass0 ** self.msp[79])
        else:
            b3 = (self.msp[77] * m3 ** (7 / 2)) / (self.msp[78] + m3 ** self.msp[79])
            rbeta = b3 + self.msp[80] * (self.mass0 - m3)
        rbeta = rbeta - 1
        return rbeta

    # 估算半径 gamma 系数
    # [已校验] Hurley_2000: equation 5.1.1(23)
    def rgammaf(self):
        m1 = 1
        b1 = max(0, self.msp[83] + self.msp[84] * (m1 - self.msp[85]) ** self.msp[86])
        if self.mass0 <= m1:
            rgamma = self.msp[83] + self.msp[84] * abs(self.mass0 - self.msp[85]) ** self.msp[86]
        elif m1 < self.mass0 <= self.msp[88]:
            rgamma = b1 + (self.msp[89] - b1) * ((self.mass0 - m1) / (self.msp[88] - m1)) ** self.msp[87]
        elif self.msp[88] < self.mass0 <= self.msp[88] + 0.1:
            if self.msp[88] > m1:
                b1 = self.msp[89]
            rgamma = b1 - 10 * b1 * (self.mass0 - self.msp[88])
        else:
            rgamma = 0
        rgamma = max(rgamma, 0)
        return rgamma

    # A function to evaluate the luminosity at the base of Giant Branch (for those models that have one)
    # Note that this function is only valid for LM & IM stars
    # [已校验] Hurley_2000: equation 5.1(10)
    def lbgbf(self):
        lbgb = (self.gbp[1] * self.mass0 ** self.gbp[5] + self.gbp[2] * self.mass0 ** self.gbp[8]) / (
                    self.gbp[3] + self.gbp[4] * self.mass0 ** self.gbp[7] + self.mass0 ** self.gbp[6])
        return lbgb

    # A function to evaluate the derivitive of the Lbgb function.
    # Note that this function is only valid for LM & IM stars
    def lbgbdf(self):
        f = self.gbp[1] * self.mass0 ** self.gbp[5] + self.gbp[2] * self.mass0 ** self.gbp[8]
        df = self.gbp[5] * self.gbp[1] * self.mass0 ** (self.gbp[5] - 1) + self.gbp[8] * self.gbp[2] * self.mass0 ** (self.gbp[8] - 1)
        g = self.gbp[3] + self.gbp[4] * self.mass0 ** self.gbp[7] + self.mass0 ** self.gbp[6]
        dg = self.gbp[7] * self.gbp[4] * self.mass0 ** (self.gbp[7] - 1) + self.gbp[6] * self.mass0 ** (self.gbp[6] - 1)
        lbgbd = (df * g - f * dg) / (g * g)
        return lbgbd

    # 估算 He星零龄主序的光度
    # [已校验] Hurley_2000: equation 6.1(77)
    def lzhef(self, m=0):
        mass = self.mass0 if m == 0 else m
        lzhe = 15262 * mass ** 10.25 / (mass ** 9 + 29.54 * mass ** 7.5 + 31.18 * mass ** 6 + 0.0469)
        return lzhe

    # A function to evaluate the ZAHB luminosity for LM stars. (OP 28/01/98)
    # Continuity with LHe, min for IM stars is ensured by setting lx = lHeif(mhefl,z,0.0,1.0)*lHef(mhefl,z,mfgb)
    # and the call to lzhef ensures continuity between the ZAHB and the NHe-ZAMS as Menv -> 0.
    # [已校验] Hurley_2000: equation 5.3(53)
    def lzahbf(self, m, mc, mhefl):
        a5 = self.lzhef(mc)
        a4 = (self.gbp[69] + a5 - self.gbp[74]) / ((self.gbp[74] - a5) * np.exp(self.gbp[71] * mhefl))
        mm = max((m - mc) / (mhefl - mc), 1e-12)
        lzahb = a5 + (1 + self.gbp[72]) * self.gbp[69] * mm ** self.gbp[70] / (
                (1 + self.gbp[72] * mm ** self.gbp[73]) * (1 + a4 * np.exp(m * self.gbp[71])))
        return lzahb

    # A function to evalute the luminosity pertubation on the MS phase for M > Mhook. (JH 24/11/97)【我对这个函数的定义有改动】
    # [已校验] Hurley_2000: equation 5.1.1(16)
    def lpertf(self):
        if self.mass0 <= self.zpars[1]:
            lhook = 0
        elif self.mass0 >= self.msp[51]:
            lhook = min(self.msp[47] / self.mass0 ** self.msp[48], self.msp[49] / self.mass0 ** self.msp[50])
        else:
            B = min(self.msp[47] / self.msp[51] ** self.msp[48], self.msp[49] / self.msp[51] ** self.msp[50])
            lhook = B * ((self.mass0 - self.zpars[1]) / (self.msp[51] - self.zpars[1])) ** 0.4
        return lhook

    # A function to evalute the radius pertubation on the MS phase for M > Mhook. (JH 24/11/97)【我对这个函数的定义有改动】
    # [已校验] Hurley_2000: equation 5.1.1(17)
    def rpertf(self):
        if self.mass0 <= self.zpars[1]:
            rhook = 0
        elif self.mass0 <= self.msp[94]:
            rhook = self.msp[95] * np.sqrt((self.mass0 - self.zpars[1]) / (self.msp[94] - self.zpars[1]))
        elif self.mass0 <= 2:
            m1 = 2
            B = (self.msp[90] + self.msp[91] * m1 ** (7 / 2)) / (self.msp[92] * m1 ** 3 + m1 ** self.msp[93]) - 1
            rhook = self.msp[95] + (B - self.msp[95]) * ((self.mass0 - self.msp[94]) / (m1 - self.msp[94])) ** self.msp[96]
        else:
            rhook = (self.msp[90] + self.msp[91] * self.mass0 ** (7 / 2)) / (self.msp[92] * self.mass0 ** 3 + self.mass0 ** self.msp[93]) - 1
        return rhook

    # A function to evaluate the BAGB luminosity. (OP 21/04/98)
    # Continuity between LM and IM functions is ensured by setting gbp(16) = lbagbf(mhefl,0.0) with gbp(16) = 1.0.
    # [已校验] Hurley_2000: equation 5.3(56) 第三行有出入
    def lbagbf(self, m=0):
        a4 = (self.gbp[9] * self.zpars[2] ** self.gbp[10] - self.gbp[16]) / (
                    np.exp(self.zpars[2] * self.gbp[11]) * self.gbp[16])
        if self.mass0 < self.zpars[2]:
            lbagb = self.gbp[9] * self.mass0 ** self.gbp[10] / (1 + a4 * np.exp(self.mass0 * self.gbp[11]))
        else:
            lbagb = (self.gbp[12] + self.gbp[13] * self.mass0 ** (self.gbp[15] + 1.8)) / (
                        self.gbp[14] + self.mass0 ** self.gbp[15])
        if m > 0:
            lbagb = (self.gbp[12] + self.gbp[13] * m ** (self.gbp[15] + 1.8)) / (self.gbp[14] + m ** self.gbp[15])
        return lbagb

    # A function to evaluate He-ignition luminosity  (OP 24/11/97)
    # Continuity between the LM and IM functions is ensured with a first call setting lhefl = lHeIf(mhefl,0.0)
    # [已校验] Hurley_2000: equation 5.3(49) 第二行有出入
    def lHeIf(self, m=0):
        mass = self.mass0 if m == 0 else m
        if mass < self.zpars[2]:
            lHeI = self.gbp[38] * mass ** self.gbp[39] / (1 + self.gbp[41] * np.exp(mass * self.gbp[40]))
        else:
            lHeI = (self.gbp[42] + self.gbp[43] * mass ** 3.8) / (self.gbp[44] + mass ** 2)
        return lHeI

    # A function to evaluate the ratio LHe,min/LHeI  (OP 20/11/97)
    # Note that this function is everywhere <= 1, and is only valid for IM stars
    # [已校验] Hurley_2000: equation 5.3(51)\
    def lHef(self, m=0):
        mass = self.mass0 if m == 0 else m
        lHe = (self.gbp[45] + self.gbp[46] * mass ** (self.gbp[48] + 0.1)) / (self.gbp[47] + mass ** self.gbp[48])
        return lHe

    # 通过 Mc 估算 GB, AGB and Naked He stars 的光度
    # [已校验] Hurley_2000: equation 5.2(37)
    def mc_to_lum_gb(self, mc, GB):
        if mc <= GB[7]:
            lum = GB[4] * (mc ** GB[5])
        else:
            lum = GB[3] * (mc ** GB[6])
        return lum

    # A function to evaluate the He-burning lifetime.
    # For IM & HM stars, tHef is relative to tBGB.
    # Continuity between LM and IM stars is ensured by setting thefl = tHef(mhefl,0.0,0.0)
    # the call to themsf ensures continuity between HB and NHe stars as Menv -> 0.
    # [已校验] Hurley_2000: equation 5.3(57)
    def tHef(self, m, mc, mhefl):
        if m <= mhefl:
            mm = max((mhefl - m) / (mhefl - mc), 1e-12)
            tHe = (self.gbp[54] + (self.themsf(mc) - self.gbp[54]) * mm ** self.gbp[55]) * (
                        1 + self.gbp[57] * np.exp(m * self.gbp[56]))
        else:
            tHe = (self.gbp[58] * m ** self.gbp[61] + self.gbp[59] * m ** 5) / (self.gbp[60] + m ** 5)
        return tHe

    # 估算 He 星的主序时间
    # [已校验] Hurley_2000: equation 6.1(79)
    def themsf(self, m=0):
        if m == 0:
            thems = (0.4129 + 18.81 * self.mass0 ** 4 + 1.853 * self.mass0 ** 6) / self.mass0 ** 6.5
        else:
            thems = (0.4129 + 18.81 * m ** 4 + 1.853 * m ** 6) / m ** 6.5
        return thems

    # 通过光度估算 GB, AGB and NHe stars 的 Mc
    # [已校验] Hurley_2000: equation 5.2(37)等效
    def lum_to_mc_gb(self, lum):
        if lum <= self.lums[6]:
            mc = (lum / self.GB[4]) ** (1 / self.GB[5])
        else:
            mc = (lum / self.GB[3]) ** (1 / self.GB[6])
        return mc

    # A function to evaluate core mass at the BAGB  (OP 25/11/97)
    # [已校验] Hurley_2000: equation 5.3(66)
    def mcagbf(self, m):
        mcagb = (self.gbp[37] + self.gbp[35] * m ** self.gbp[36]) ** (1 / 4)
        return mcagb

    # 估算渐近巨星分支上的半径
    # [已校验] Hurley_2000: equation 5.4(74)
    def ragbf(self, m, lum, mhef):
        m1 = mhef - 0.2
        if m <= m1:
            b50 = self.gbp[19]
            A = self.gbp[29] + self.gbp[30] * m
        elif m >= mhef:
            b50 = self.gbp[19] * self.gbp[24]
            A = min(self.gbp[25] / m ** self.gbp[26], self.gbp[27] / m ** self.gbp[28])
        else:
            b50 = self.gbp[19] * (1 + (self.gbp[24] - 1) * (m - m1) / 0.2)
            A = self.gbp[31] + (self.gbp[32] - self.gbp[31]) * (m - m1) / 0.2
        ragb = A * (lum ** self.gbp[18] + self.gbp[17] * lum ** b50)
        return ragb

    # A function to evaluate core mass at BGB or He ignition (depending on mchefl) for IM & HM stars
    # [已校验] Hurley_2000: equation 5.2(44)
    def mcheif(self, m, mhefl, mchefl):
        mcbagb = self.mcagbf(m)
        a3 = mchefl ** 4 - self.gbp[33] * mhefl ** self.gbp[34]
        mchei = min(0.95 * mcbagb, (a3 + self.gbp[33] * m ** self.gbp[34]) ** (1 / 4))
        return mchei

    # A function to evaluate Mc given t for GB, AGB and NHe stars
    # [已校验] Hurley_2000: equation 5.2(34、39)
    def mcgbtf(self, t, A, GB, tinf1, tinf2, tx):
        if t <= tx:
            mcgbt = ((GB[5] - 1) * A * GB[4] * (tinf1 - t)) ** (1 / (1 - GB[5]))
        else:
            mcgbt = ((GB[6] - 1) * A * GB[3] * (tinf2 - t)) ** (1 / (1 - GB[6]))
        return mcgbt

    # A function to evaluate the minimum radius during blue loop(He-burning) for IM & HM stars
    # [已校验] Hurley_2000: equation 5.3(55)
    def rminf(self, m):
        rmin = (self.gbp[49] * m + (self.gbp[50] * m) ** self.gbp[52] * m ** self.gbp[53]) / (self.gbp[51] + m ** self.gbp[53])
        return rmin

    # 估算巨星分支上的半径
    # [已校验] Hurley_2000: equation 5.2(46)
    def rgbf(self, m, lum):
        a = min(self.gbp[20] / m ** self.gbp[21], self.gbp[22] / m ** self.gbp[23])
        rgb = a * (lum ** self.gbp[18] + self.gbp[17] * lum ** self.gbp[19])
        return rgb

    # 估算低质量恒星的零龄水平分支(ZAHB)半径
    # Continuity with R(LHe,min) for IM stars is ensured by setting lx = lHeif(mhefl,z,0.0,1.0)*lHef(mhefl,z,mfgb),
    # and the call to rzhef ensures continuity between the ZAHB and the NHe-ZAMS as Menv -> 0.
    # [已校验] Hurley_2000: equation 5.3(54)
    def rzahbf(self, m, mc, mhefl):
        rx = self.rzhef(mc)
        ry = self.rgbf(m, self.lzahbf(m, mc, mhefl))
        mm = max((m - mc) / (mhefl - mc), 1e-12)
        f = (1 + self.gbp[76]) * mm ** self.gbp[75] / (1 + self.gbp[76] * mm ** self.gbp[77])
        rzahb = (1 - f) * rx + f * ry
        return rzahb

    # 估算 He 星零龄主序的半径
    # [已校验] Hurley_2000: equation 6.1(78)
    def rzhef(self, m):
        rzhe = 0.2391 * m ** 4.6 / (m ** 4 + 0.162 * m ** 3 + 0.0065)
        return rzhe

    # A function to evaluate radius derivitive on the GB (as f(L)). [无调用]
    def rgbdf(m, lum, x):
        a1 = min(x.gbp[20] / m ** x.gbp[21], x.gbp[22] / m ** x.gbp[23])
        rgbd = a1 * (x.gbp[18] * lum ** (x.gbp[18] - 1) + x.gbp[17] * x.gbp[19] * lum ** (x.gbp[19] - 1))
        return rgbd

    # A function to evaluate radius derivitive on the AGB (as f(L)).[无调用]
    def ragbdf(m, lum, mhelf, x):
        m1 = mhelf - 0.2
        if m >= mhelf:
            xx = x.gbp[24]
        elif m >= m1:
            xx = 1 + 5 * (x.gbp[24] - 1) * (m - m1)
        else:
            xx = 1
        a4 = xx * x.gbp[19]
        if m <= m1:
            a1 = x.gbp[29] + x.gbp[30] * m
        elif m >= mhelf:
            a1 = min(x.gbp[25] / m ** x.gbp[26], x.gbp[27] / m ** x.gbp[28])
        else:
            a1 = x.gbp[31] + 5 * (x.gbp[32] - x.gbp[31]) * (m - m1)
        ragbd = a1 * (x.gbp[18] * lum ** (x.gbp[18] - 1) + x.gbp[17] * a4 * lum ** (a4 - 1))
        return ragbd

    # A function to evaluate core mass at the end of the MS as a fraction of the BGB value,
    # i.e. this must be multiplied by the BGB value (see below) to give the actual core mass.
    # [已校验] Hurley_2000: equation 5.1.2(29)
    def mctmsf(self):
        mctms = (1.586 + self.mass0 ** 5.25) / (2.434 + 1.02 * self.mass0 ** 5.25)
        return mctms

    # A function to evaluate mass at BGB or He ignition (depending on mchefl) for IM & HM stars by inverting mcheif
    def mheif(self, mc, mhefl, mchefl):
        m1 = self.mbagbf(mc / 0.95)
        a3 = mchefl ** 4 - self.gbp[33] * mhefl ** self.gbp[34]
        m2 = ((mc ** 4 - a3) / self.gbp[33]) ** (1 / self.gbp[34])
        mhei = max(m1, m2)
        return mhei

    # A function to evaluate mass at the BAGB by inverting mcagbf.
    def mbagbf(self, mc):
        mc4 = mc ** 4
        if mc4 > self.gbp[37]:
            mbagb = ((mc4 - self.gbp[37]) / self.gbp[35]) ** (1 / self.gbp[36])
        else:
            mbagb = 0
        return mbagb

    # A function to evaluate L given t for GB, AGB and NHe stars
    # [已校验] Hurley_2000: equation 5.2(35)
    def lgbtf(self, A):
        if self.age <= self.tscls[6]:
            lgbt = self.GB[4] * (((self.GB[5] - 1) * A * self.GB[4] * (self.tscls[4] - self.age)) ** (self.GB[5] / (1 - self.GB[5])))
        else:
            lgbt = self.GB[3] * (((self.GB[6] - 1) * A * self.GB[3] * (self.tscls[5] - self.age)) ** (self.GB[6] / (1 - self.GB[6])))
        return lgbt

    # A function to evaluate the blue-loop fraction of the He-burning lifetime for IM & HM stars  (OP 28/01/98)
    # [已校验] Hurley_2000: equation 5.3(58) 有些不太一样
    def tblf(self):
        mr = self.zpars[2] / self.zpars[3]
        if self.mass0 <= self.zpars[3]:
            m1 = self.mass0 / self.zpars[3]
            m2 = np.log10(m1) / np.log10(mr)
            m2 = max(m2, 1e-12)
            tbl = self.gbp[64] * m1 ** self.gbp[63] + self.gbp[65] * m2 ** self.gbp[62]
        else:
            r1 = 1 - self.rminf(self.mass0) / self.ragbf(self.mass0, self.lHeIf(), self.zpars[2])
            r1 = max(r1, 1e-12)
            tbl = self.gbp[66] * self.mass0 ** self.gbp[67] * r1 ** self.gbp[68]
        tbl = min(1, max(0, tbl))
        if tbl < 1e-10:
            tbl = 0
        return tbl

    # 估算 He 星主序上的光度
    # [已校验] Hurley_2000: equation 6.1(78) [无调用]
    def l_He_MS(self, m):
        lzhe = 15262 * m ** 10.25 / (m ** 9 + 29.54 * m ** 7.5 + 31.18 * m ** 6 + 0.0469)
        return lzhe

    # 根据质量、光度估算赫氏空隙中 He 星的半径
    def rhehgf(self, m, lum, rzhe, lthe):
        Lambda = 500 * (2 + m ** 5) / m ** 2.5
        rhehg = rzhe * (lum / lthe) ** 0.2 + 0.02 * (np.exp(lum / Lambda) - np.exp(lthe / Lambda))
        return rhehg

    # 估算 He 巨星的半径
    def rhegbf(self, lum):
        rhegb = 0.08 * lum ** (3 / 4)
        return rhegb

    # 估算光度扰动的指数(适用于非主序星)
    def lpert1f(self, m, mu):
        b = 0.002 * max(1, 2.5 / m)
        lpert = (1 + b ** 3) * ((mu / b) ** 3) / (1 + (mu / b) ** 3)
        return lpert

    # 估算半径扰动的指数(适用于非主序星)
    def rpert1f(self, m, mu, r, rc):
        if mu <= 0:
            rpert = 0
        else:
            c = 0.006 * max(1, 2.5 / m)
            q = np.log(r / rc)
            fac = 0.1 / q
            facmax = -14 / np.log10(mu)
            fac = min(fac, facmax)
            rpert = ((1 + c ** 3) * ((mu / c) ** 3) * (mu ** fac)) / (1 + (mu / c) ** 3)
        return rpert

    def vrotf(self, m):
        vrot = 330 * m ** 3.3 / (15 + m ** 3.45)
        return vrot

