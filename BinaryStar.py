from numba import float64, int64, types
from numba.experimental import jitclass
import numpy as np
from const_new import alpha_wind, beta_wind
from const_new import wind_model, acc1, yeardy, sep_to_period, period_to_sep, mu_wind, spin_orbit_resonance
from SingleStar import SingleStar
from utils import conditional_jitclass
from utils import rochelobe
from StellarCal import StellarCal
from StellarProp import StellarProp


# Binary star class
spec = [
    ('star1', SingleStar.class_type.instance_type),
    ('star2', SingleStar.class_type.instance_type),
    ('totalmass', float64),                     # total mass of binary
    ('Z', float64),                             # metallicity      [unit: Zsun]
    ('ecc', float64),                           # eccentricity
    ('sep', float64),                           # semimajor axis   [unit: Rsun]    (sep和period任选一输入即可)
    ('period', float64),                        # orbital period   [unit: year]     (作为变量输入时, 单位是天)
    ('omega', float64),                         # 轨道角频率         [unit: /yr]
    ('jorb', float64),                          # 轨道角动量         [unit: Msun * Rsun2 / yr]
    ('dt', float64),                            # evolution timestep
    ('q1', float64),                            # mass ratio: m1/m2
    ('q2', float64),                            # mass ratio: m2/m1
    ('jdot', float64),                          # 轨道角动量变化率
    ('jdot_wind', float64),                     # 星风引起的轨道角动量变化率
    ('jdot_gr', float64),                       # 引力波辐射引起轨道角动量变化率   [unit: Msun * Rsun2 / yr2]
    ('jdot_mb', float64),                       # 磁制动引起的轨道角动量变化率
    ('edot', float64),                          # 轨道偏心率的变化率
    ('edot_wind', float64),                     # 星风引起的偏心率变化率
    ('edot_gr', float64),                       # 引力波辐射引起轨道角动量变化率
    ('edot_tide', float64),                     # 引力波辐射引起轨道角动量变化率
    ('state', types.string),                    # 双星状态['detached','semi-contacted','contacted','CE']
    ('event', types.optional(types.string)),    # 发生的事件
    ('ktype', int64[:, :]),                     # 计算双星碰撞后的恒星类型
    ('max_time', float64),                      # 最长演化时间        [unit: Myr]
    ('time', float64),                          # 当前的演化时间        [unit: Myr]
    ('max_step', float64),                      # 最大演化步长
    ('step', int64),                            # 当前的演化步长
    ('data', float64[:, :]),                    # 存储每个步长的属性
]


@conditional_jitclass(spec)
class BinaryStar:
    def __init__(self, star1, star2, eccentricity=0, separation=0, period=0, dt=0,
                 jdot=0, jdot_wind=0, jdot_gr=0, jdot_mb=0, edot=0, edot_wind=0, edot_gr=0, edot_tide=0,
                 state='detached', event=None, max_time=10000, time=0, max_step=20000, step=0):
        self.star1 = star1
        self.star2 = star2
        self.totalmass = star1.mass + star2.mass
        self.Z = star1.Z
        self.ecc = eccentricity
        self._set_orbital_parameter(separation, period)
        self.omega = 2 * np.pi / self.period
        self.jorb = self._set_jorb()
        self._set_spin()
        self.dt = dt
        self.q1 = star1.mass / star2.mass
        self.q2 = star2.mass / star1.mass
        self.jdot = jdot
        self.jdot_wind = jdot_wind
        self.jdot_gr = jdot_gr
        self.jdot_mb = jdot_mb
        self.edot = edot
        self.edot_wind = edot_wind
        self.edot_gr = edot_gr
        self.edot_tide = edot_tide
        self.state = state
        self.event = event
        self.cal_radius_rochelobe()
        self.max_time = max_time
        self.time = time
        self.max_step = max_step
        self.step = step
        self.data = np.zeros((max_step, 30))
        self._set_ktype()

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    演化双星
    # ------------------------------------------------------------------------------------------------------------------
    def evolve(self):
        # 首先保存双星的初始属性
        self.save()

        # 确定两颗恒星的不同演化阶段的时标、标志性光度、巨星分支参数
        StellarCal(self.star1)
        StellarCal(self.star2)

        # 确定两颗恒星的光度、半径、核质量、核半径、对流包层质量/半径/转动惯量系数
        StellarProp(self.star1)
        StellarProp(self.star2)

        # 为双星设置合适的自旋值 [待完善]

        # 计算双星自旋角动量
        self.star1.cal_jspin()
        self.star2.cal_jspin()

        # 如果恒星为致密星，设置最小步长为0.01Myrs
        if 10 <= self.star1.type <= 14:
            self.star1.dt = 1e4
        if 10 <= self.star2.type <= 14:
            self.star2.type = 1e4

        # 考虑星风的影响（质量/自旋角动量/轨道角动量的减少/增加）
        # self.steller_wind()
        # 考虑双星的磁制动影响（自旋角动量的减少）
        # self.star1.magnetic_braking()
        # self.star2.magnetic_braking()
        # 考虑引力波辐射的影响(轨道角动量的减少)
        # self.GW_radiation()
        # 考虑潮汐的圆化、轨道收缩和自旋 [待完善]

        # 限制最大 0.2% 的轨道角动量变化
        # self.jdot = self.jdot_wind + self.jdot_gr + self.jdot_mb
        # self.dt = min(0.002 * self.jorb / self.jdot, self.star1.dt, self.star2.dt)

        # 对于非致密星, 每次质量损失不超过包层质量, 且限制 1%
        self.star1.limit_mass_change()
        self.star2.limit_mass_change()
        self.star1.dt = self.star2.dt = self.dt = min(self.star1.dt, self.star2.dt, self.dt)

        # 确保恒星的自旋不会瓦解 [待完善]
        # spin_crit = 2 * np.pi * np.sqrt(mass[k] * aursun ** 3 / rad[k] ** 3)

        # 更新质量和自旋
        self.star1.reset()
        self.star2.reset()

        # 更新轨道的角动量/偏心率/半长轴/周期/角频率

        # 跳过
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #                                                    保存当前属性
    # ------------------------------------------------------------------------------------------------------------------
    def save(self):
        self.star1.save()
        self.star2.save()
        self.data[self.step, 0] = self.time
        self.data[self.step, 1] = self.ecc
        self.data[self.step, 2] = self.period
        self.data[self.step, 3] = self.sep
        self.data[self.step, 4] = self.star1.R / self.star1.rochelobe
        self.data[self.step, 5] = self.star2.R / self.star2.rochelobe
        self.data[self.step, 6] = self.jdot
        self.data[self.step, 7] = self.jdot_wind
        self.data[self.step, 8] = self.jdot_gr
        self.data[self.step, 9] = self.jdot_mb
        self.data[self.step, 10] = self.edot

    # ------------------------------------------------------------------------------------------------------------------
    #                                                  初始化ktype矩阵
    # ------------------------------------------------------------------------------------------------------------------
    # 计算轨道参数 (unit: year)
    def _set_ktype(self):
        self.ktype = np.array([[  1,   1, 102, 103, 104, 105, 106,   4, 106, 106,   3,   6,   6,  13,  14],
                               [  1,   1, 102, 103, 104, 105, 106,   4, 106, 106,   3,   6,   6,  13,  14],
                               [102, 102, 103, 103, 104, 104, 105, 104, 104, 104, 103, 105, 105, 113, 114],
                               [103, 103, 103, 103, 104, 104, 105, 104, 104, 104, 103, 105, 105, 113, 114],
                               [104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 113, 114],
                               [105, 105, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 113, 114],
                               [106, 106, 105, 105, 104, 104, 106, 104, 106, 106, 105, 106, 106, 113, 114],
                               [  4,   4, 104, 104, 104, 104, 104,   1, 108, 109,   7,   9,   9,  13,  14],
                               [106, 106, 104, 104, 104, 104, 106, 108, 108, 109, 107, 109, 109, 113, 114],
                               [106, 106, 104, 104, 104, 104, 106, 109, 109, 109, 107, 109, 109, 113, 114],
                               [  3,   3, 103, 103, 104, 104, 105,   7, 107, 107,  15,   9,   9,  13,  14],
                               [  6,   6, 105, 105, 104, 104, 106,   9, 109, 109,   9,  11,  12,  13,  14],
                               [  6,   6, 105, 105, 104, 104, 106,   9, 109, 109,   9,  12,  12,  13,  14],
                               [ 13,  13, 113, 113, 113, 113, 113,  13, 113, 113,  13,  13,  13,  14,  14],
                               [ 14,  14, 114, 114, 114, 114, 114,  14, 114, 114,  14,  14,  14,  14,  14]])

    # ------------------------------------------------------------------------------------------------------------------
    #                                                  初始化轨道参数
    # ------------------------------------------------------------------------------------------------------------------
    # 计算轨道参数 (unit: year)
    def _set_orbital_parameter(self, separation, period):
        if separation > 0:
            self.sep = separation
            self.period = sep_to_period * (self.sep ** 3 / self.totalmass) ** 0.5
        elif period > 0:
            self.period = period / yeardy
            self.sep = period_to_sep * (self.period ** 2 * self.totalmass) ** (1 / 3)
        else:
            raise ValueError("At least one of 'period' and 'separation' must be provided.")

    # 设置恒星自旋
    def _set_spin(self):
        if spin_orbit_resonance:
            self.star1.spin = self.omega
            self.star2.spin = self.omega

    # 计算轨道角动量
    def _set_jorb(self):
        reduced_mass = self.star1.mass * self.star2.mass / self.totalmass
        jorb = reduced_mass * self.omega * self.sep ** 2 * np.sqrt(1 - self.ecc ** 2)
        return jorb

    # 更新轨道参数(角动量/偏心率/半长轴/周期/角频率)
    def reset_orbital_parameter(self):
        self.jorb = self.jorb + self.jdot * self.dt
        self.edot = self.edot_gr + self.edot_wind + self.edot_tide
        self.ecc = self.ecc + self.edot * self.dt

    # 计算洛希瓣半径
    def cal_radius_rochelobe(self):
        self.star1.rochelobe = self.sep * rochelobe(self.q1)
        self.star2.rochelobe = self.sep * rochelobe(self.q2)

    # ------------------------------------------------------------------------------------------------------------------
    #                                       考虑星风的影响(自旋角动量/轨道角动量/偏心率)
    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    #                                       密近双星的引力波辐射导致轨道角动量损失
    # ------------------------------------------------------------------------------------------------------------------
    def GW_radiation(self):
        if self.sep <= 1000:
            ecc4 = np.sqrt(1.0 - self.ecc ** 2)
            term1 = self.star1.mass * self.star2.mass * self.totalmass / self.sep**4
            term2 = (1.0 + 0.875 * self.ecc**2) / ecc4**5
            term3 = ((19 / 6) + (121 / 96) * self.ecc ** 2) / ecc4**5
            self.jdot_gr = - 8.315e-10 * term1 * term2 * self.jorb
            self.edot_gr = - 8.315e-10 * term1 * term3 * self.ecc

    # 计算潮汐影响
    def tide_effect(self):
        pass

    # 双星的chirp mass
    def chirp_mass(self):
        return (self.star1.mass * self.star2.mass) ** 0.6 / (self.star1.mass + self.star2.mass) ** 0.2


