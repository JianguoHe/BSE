from numba import float64, int64, types
from numba.experimental import jitclass
import numpy as np
from const_new import gamma_mb, mb_model, yearsc, Zsun, neta, bwind, f_WR, f_LBV, Rsun, Teffsun
from zcnst import zcnsts_set


# Single star class
@jitclass([
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
    ('mdot_wind_loss', float64),            # 星风质量损失率
    ('mdot_wind_accrete', float64),         # 星风质量吸积率
    ('jdot_spin_wind', float64),            # 星风提取的自旋角动量
    ('jdot_spin_mb', float64),              # 磁制动提取的自旋角动量
    ('max_time', float64),                  # 最长演化时间        [unit: Myr]
    ('time', float64),                      # 当前的演化时间        [unit: Myr]
    ('max_step', float64),                  # 最大演化步长
    ('step', int64),                        # 当前的演化步长
    ('data', float64[:, :]),                # 存储每个步长的属性
    ('zpars', float64[:]),               # 存储每个步长的属性
    ('msp', float64[:]),                 # 存储每个步长的属性
    ('gbp', float64[:]),                 # 存储每个步长的属性
])
class SingleStar:
    def __init__(self, type, Z, mass, R=0, L=0, dt=1e6, Teff=0, spin=0, jspin=0, rochelobe=0,
                 mass_core=0, mass_he_core=0, mass_c_core=0, mass_o_core=0, mass_co_core=0, mass_envelop=0,
                 radius_core=0, radius_he_core=0, radius_c_core=0, radius_o_core=0, radius_co_core=0,
                 mdot_wind_loss=0, mdot_wind_accrete=0, jdot_spin_wind=0, jdot_spin_mb=0,
                 max_time=10000, time=0, max_step=20000, step=0):
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
        self.mdot_wind_loss = mdot_wind_loss
        self.mdot_wind_accrete = mdot_wind_accrete
        self.jdot_spin_wind = jdot_spin_wind
        self.jdot_spin_mb = jdot_spin_mb
        self.max_time = max_time
        self.time = time
        self.max_step = max_step
        self.step = step
        self.data = np.zeros((max_step, 30))
        self.zpars = np.zeros(20)
        self.msp = np.zeros(200)
        self.gbp = np.zeros(200)

    # 计算金属丰度相关常数
    def _set_jorb(self):
        reduced_mass = self.star1.mass * self.star2.mass / self.totalmass
        jorb = reduced_mass * self.omega * self.sep ** 2 * np.sqrt(1 - self.ecc ** 2)
        return jorb

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

    # def evolve(self, force):
    #     # update velocity and position based on force
    #     self.velocity += force / self.mass * self.dt
    #     self.position += self.velocity * self.dt
    #
    #     # update time
    #     self.time += self.dt
