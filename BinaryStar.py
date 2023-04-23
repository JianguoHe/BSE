from numba import float64, int64, types
from numba.experimental import jitclass
import numpy as np
from const import yearsc, alpha_wind, kick, output, find, SNtype, G, Msun, Rsun, beta_wind, tiny
from const import wind_model, acc1, yeardy, sep_to_period, period_to_sep, mu_wind
from zfuncs import rochelobe
from SingleStar import SingleStar
# from stellerwind import steller_wind


# Binary star class
@jitclass([
    ('star1', SingleStar.class_type.instance_type),
    ('star2', SingleStar.class_type.instance_type),
    ('Z', float64),                             # unit: Zsun
    ('sep', float64),                           # unit: Rsun                     (sep和period任选一输入即可)
    ('period', float64),                        # orbital period, unit: year     (作为变量输入时, 单位是天)
    ('omega', float64),                         # 轨道角频率, unit: /yr
    ('ecc', float64),                           # 偏心率
    ('jorb', float64),                          # 轨道角动量
    ('q1', float64),                            # 轨道角动量
    ('q2', float64),                            # 轨道角动量
    ('dt', float64),                            # evolution timestep
    ('totalmass', float64),                     # total mass of binary
    ('jdot_wind', float64),                     # 星风引起的轨道角动量变化
    ('jdot_gw', float64),                       # 引力波辐射引起轨道角动量变化
    ('jdot_mb', float64),                       # 磁制动引起的轨道角动量变化率
    ('edot_wind', float64),                     # 星风引起的偏心率变化
    ('edot_gw', float64),                       # 引力波辐射引起轨道角动量变化
    ('edot_tide', float64),                     # 引力波辐射引起轨道角动量变化
    ('state', types.string),                    # 双星状态['detached','semi-contacted','contacted','CE']
])
class BinaryStar:
    def __init__(self, star1, star2, separation=None, period=None, eccentricity=0, dt=0,
                 jdot_wind=0, jdot_gw=0, jdot_mb=0, edot_wind=0, edot_gw=0, edot_tide=0, state='detached'):
        self.star1 = star1
        self.star2 = star2
        self.Z = star1.Z
        self.totalmass = star1.mass + star2.mass
        self.ecc = eccentricity
        self.dt = dt
        self.sep, self.period = self._set_orbital_parameter(separation, period)
        self.omega = 2 * np.pi / self.period
        self.jorb = self._set_jorb()
        self.q1 = star1.mass / star2.mass
        self.q2 = star2.mass / star1.mass
        self.jdot_wind = jdot_wind
        self.jdot_gw = jdot_gw
        self.jdot_mb = jdot_mb
        self.edot_wind = edot_wind
        self.edot_gw = edot_gw
        self.edot_tide = edot_tide
        self.state = state
        self.cal_radius_rochelobe()

    # 计算轨道参数 (unit: year)
    def _set_orbital_parameter(self, separation, period):
        if separation is not None:
            period = sep_to_period * (separation ** 3 / self.totalmass) ** 0.5
            return separation, period
        elif period is not None:
            period = period / yeardy
            separation = period_to_sep * (period ** 2 * self.totalmass) ** (1 / 3)
            return separation, period
        else:
            raise ValueError("At least one of 'period' and 'separation' must be provided.")

    def _set_jorb(self):
        reduced_mass = self.star1.mass * self.star2.mass / self.totalmass
        jorb = reduced_mass * self.omega * self.sep ** 2 * np.sqrt(1 - self.ecc ** 2)
        return jorb

    # 计算洛希瓣半径
    def cal_radius_rochelobe(self):
        self.star1.rochelobe = self.sep * rochelobe(self.q1)
        self.star2.rochelobe = self.sep * rochelobe(self.q2)

    # 考虑星风的影响(自旋角动量/轨道角动量/偏心率)
    def steller_wind(self):
        # 星风质量损失率、吸积率
        self.mdot_wind()
        # 自旋角动量的变化率
        for i in range(2):
            star1 = [self.star1, self.star2][i]
            star2 = [self.star1, self.star2][1-i]
            term1 = star1.mdot_wind_loss * star1.spin * star1.R ** 2
            term2 = star1.mdot_wind_accrete * star2.spin * star2.R ** 2 * mu_wind
            star1.jdot_spin_wind = (term1 + term2) * (2 / 3)
        # 轨道角动量的变化率
        ecc4 = np.sqrt(1.0 - self.ecc**2)
        term5 = (self.star1.mdot_wind_loss - self.star1.mdot_wind_accrete * self.q1) * self.star2.mass ** 2
        term6 = (self.star2.mdot_wind_loss - self.star2.mdot_wind_accrete * self.q2) * self.star1.mass ** 2
        self.jdot_wind = (term5 + term6) * self.sep**2 * ecc4 * self.omega / self.totalmass ** 2
        # 偏心率的变化率
        term7 = self.star1.mdot_wind_accrete * (0.5 / self.star1.mass + 1.0 / self.totalmass)
        term8 = self.star2.mdot_wind_accrete * (0.5 / self.star2.mass + 1.0 / self.totalmass)
        self.edot_wind = -self.ecc * (term7 + term8)

    # 星风质量损失/星风吸积
    def mdot_wind(self):
        for i in range(2):
            star1 = [self.star1, self.star2][i]
            star2 = [self.star1, self.star2][1-i]
            # 计算 star1 的星风质量损失率，用 mdot_wind_loss 表示
            if wind_model == 'Hurley':
                star1.mdot_wind_loss = -star1.cal_mdot_wind_Hurley(self.ecc)
            else:
                star1.mdot_wind_loss = -star1.cal_mdot_wind_Belczynski(self.ecc)
            # 计算 star2 从 star1 星风中质量吸积率, 用 mdot_wind_accrete 表示(Boffin & Jorissen, A&A 1988, 205, 155).
            vorb2 = acc1 * (star1.mass + star2.mass) / self.sep
            vwind2 = 2.0 * beta_wind * acc1 * star1.mass / star1.R
            term1 = 1.0 / np.sqrt(1.0 - self.ecc ** 2)
            term2 = (acc1 * star2.mass / vwind2) ** 2
            term3 = 1 / (1.0 + vorb2 / vwind2) ** 1.5
            term4 = alpha_wind * abs(star1.mdot_wind_loss) / (2.0 * self.sep ** 2)
            star2.mdot_wind_accrete = term1 * term2 * term3 * term4
            star2.mdot_wind_accrete = min(star2.mdot_wind_accrete, 0.8 * abs(star1.mdot_wind_loss))

    # 密近双星的引力波辐射导致轨道角动量损失
    def GW_radiation(self):
        if self.sep <= 1000:
            ecc4 = np.sqrt(1.0 - self.ecc ** 2)
            term1 = self.star1.mass * self.star2.mass * self.totalmass / self.sep**4
            term2 = (1.0 + 0.875 * self.ecc**2) / ecc4**5
            term3 = ((19 / 6) + (121 / 96) * self.ecc ** 2) / ecc4**5
            self.jdot_gw = - 8.315e-10 * term1 * term2 * self.jorb
            self.edot_gw = - 8.315e-10 * term1 * term3 * self.ecc

    # 计算潮汐影响
    def tide_effect(self):
        for k in range(2):
            if k == 0:
                star = self.star1
            else:
                star = self.star2

    # 双星的chirp mass
    def chirp_mass(self):
        return (self.star1.mass * self.star2.mass) ** 0.6 / (self.star1.mass + self.star2.mass) ** 0.2

    # 演化双星
    def evolve(self):
        # 考虑星风的影响（质量/自旋角动量/轨道角动量的减少/增加）
        # self.steller_wind()
        # 考虑双星的磁制动影响（自旋角动量的减少）
        self.star1.magnetic_braking()
        self.star2.magnetic_braking()


