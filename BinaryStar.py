from numba import float64, int64, types
from numba.experimental import jitclass
import numpy as np
from const import yearsc, alpha_wind, kick, output, find, SNtype, G, Msun, Rsun, beta_wind, tiny
from zfuncs import rochelobe
from SingleStar import SingleStar
from stellerwind import steller_wind


# Binary star class
@jitclass([
    ('star1', SingleStar.class_type.instance_type),
    ('star2', SingleStar.class_type.instance_type),
    ('state', types.string),
    ('Z', float64),  # unit: radius of sun
    ('sep', float64),  # unit: radius of sun
    ('tb', float64),  # 轨道周期, unit: year
    ('ecc', float64),
    ('jorb', float64),          # 轨道角动量
    ('q1', float64),          # 轨道角动量
    ('q2', float64),          # 轨道角动量
    ('totalmass', float64),          # 轨道角动量
    ('djorb_wind', float64),    # 星风引起的轨道角动量变化
    ('djorb_gw', float64),      # 引力波辐射引起轨道角动量变化
    ('decc_wind', float64),     # 星风引起的偏心率变化
    ('decc_gw', float64),       # 引力波辐射引起轨道角动量变化
])
class BinaryStar:
    def __init__(self, star1, star2, separation, eccentricity, state='detached', jorb=0,
                 djorb_wind=0, djorb_gw=0, decc_wind=0, decc_gw=0):
        self.star1 = star1
        self.star2 = star2
        self.Z = star1.Z
        self.sep = separation
        self.tb = self.cal_orbital_period()
        self.omega = 2 * np.pi / self.tb
        self.ecc = eccentricity
        self.jorb = jorb
        self.q1 = star1.mass / star2.mass
        self.q2 = star2.mass / star1.mass
        self.totalmass = star1.mass + star2.mass
        self.djorb_wind = djorb_wind
        self.djorb_gw = djorb_gw
        self.decc_wind = decc_wind
        self.decc_gw = decc_gw
        self.state = state
        self.cal_radius_rochelobe()

    def cal_orbital_period(self):
        sep = self.sep * Rsun
        m1 = self.star1.mass * Msun
        m2 = self.star2.mass * Msun
        tb = 2 * np.pi * sep ** (3 / 2) * (G * (m1 + m2)) ** (-1 / 2) / yearsc    # unit: year
        return tb

    # 计算洛希瓣半径
    def cal_radius_rochelobe(self):
        self.star1.rochelobe = self.sep * rochelobe(self.q1)
        self.star2.rochelobe = self.sep * rochelobe(self.q2)

    # 考虑星风的影响
    def steller_wind(self):
        # 计算恒星星风损失/吸积的质量
        steller_wind(self.star1, self.star2, self.sep, self.ecc, self.Z)
        steller_wind(self.star2, self.star1, self.sep, self.ecc, self.Z)
        # 考虑星风对轨道角动量和偏心率的影响
        ecc4 = np.sqrt(1.0 - self.ecc**2)
        term1 = (self.star1.dml_wind - self.star1.dma_wind * self.q1) * self.star2.mass ** 2
        term2 = (self.star2.dml_wind - self.star2.dma_wind * self.q2) * self.star1.mass ** 2
        self.djorb_wind = (term1 + term2) * self.sep**2 * ecc4 * self.omega / self.totalmass ** 2
        term3 = self.star1.dma_wind * (0.5 / self.star1.mass + 1.0 / self.totalmass)
        term4 = self.star2.dma_wind * (0.5 / self.star2.mass + 1.0 / self.totalmass)
        self.decc_wind = - self.ecc * (term3 + term4)

    # 密近双星的引力波辐射导致轨道角动量损失
    def GW_radiation(self):
        if self.sep <= 1000:
            ecc4 = np.sqrt(1.0 - self.ecc ** 2)
            term1 = self.star1.mass * self.star2.mass * self.totalmass / self.sep**4
            term2 = (1.0 + 0.875 * self.ecc**2) / ecc4**5
            term3 = ((19 / 6) + (121 / 96) * self.ecc ** 2) / ecc4**5
            self.djorb_gw = - 8.315e-10 * term1 * term2 * self.jorb
            self.decc_gw = - 8.315e-10 * term1 * term3 * self.ecc





    def chirp_mass(self):
        return (self.star1.mass * self.star2.mass) ** 0.6 / (self.star1.mass + self.star2.mass) ** 0.2
