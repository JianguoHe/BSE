from numba import float64, int64, types
from numba.experimental import jitclass
import numpy as np
from const import acc1, alpha_wind, kick, output, find, SNtype, G, Msun, Rsun, beta_wind, tiny
from zfuncs import rochelobe
from stellerwind import steller_wind


# Single star class
@jitclass([
    ('type', int64),            # steller type
    ('Z', float64),  # initial mass fraction of metals
    ('mass', float64),  # mass (solar units)
    ('R', float64),  # log10 of radius (solar units)
    ('L', float64),  # log10 luminosity (solar units)
    ('dt', float64),            # 演化步长
    ('Teff', float64),  # effective temperature (K)
    ('spin', float64),  # the dimesionless spin of the star, if it is a compact object,
                        # which is equal to c*J/(GM^2).
    ('jspin', float64),  # 自旋角动量
    ('rochelobe', float64),      # in solar units
    ('mass_core', float64),  # in solar units
    ('mass_he_core', float64),  # in solar units
    ('mass_c_core', float64),  # in solar units
    ('mass_o_core', float64),  # in solar units
    ('mass_co_core', float64),  # in solar units
    ('mass_envelop', float64),  # in solar units
    ('radius_core', float64),  # in solar units
    ('radius_he_core', float64),  # in solar units
    ('radius_c_core', float64),  # in solar units
    ('radius_o_core', float64),  # in solar units
    ('radius_co_core', float64),  # in solar units
    ('dml_wind', float64),      # 星风质量损失
    ('dma_wind', float64),      # 星风质量吸积
    ('djspin_wind', float64),    # 星风提取的自旋角动量
])
class SingleStar:
    def __init__(self, type, Z, mass, R=0, L=0, dt=0, Teff=0, spin=0, jspin=0, rochelobe=0,
                 mass_core=0, mass_he_core=0, mass_c_core=0, mass_o_core=0, mass_co_core=0, mass_envelop=0,
                 radius_core=0, radius_he_core=0, radius_c_core=0, radius_o_core=0, radius_co_core=0,
                 dml_wind=0, dma_wind=0, djspin_wind=0):
        self.type = type
        self.mass = mass
        self.R = R
        self.L = L
        self.Teff = Teff
        self.spin = spin
        self.jspin = jspin
        self.Z = Z
        self.dt = dt
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
        # 星风中的变量
        self.dml_wind = dml_wind
        self.dma_wind = dma_wind
        self.djspin_wind = djspin_wind

    # def massloss_wind(self):
    #     self.massloss_wind = mlwind()

    # 考虑磁制动的影响
    def magnetic_braking(self):
        # 计算有明显对流包层的恒星因磁制动损失的自旋角动量, 包括主序星(M < 1.25)、靠近巨星分支的HG恒星以及巨星, 不包括完全对流主序星
        if self.mass > 0.35 and self.type < 10:
            djspin = -5.83e-16 * self.mass_envelop * (self.R * self.spin) ** 3 / self.mass
            # 限制最大3%的磁制动损失的角动量。这可以保证迭代次数不会超过最大值20000, 当然2%也不会影响演化结果
            if djspin > tiny:
                dtt = 0.03 * self.jspin / abs(djspin)
                self.dt = min(self.dt, dtt)
            self.jspin += djspin * self.dt

    # def evolve(self, force):
    #     # update velocity and position based on force
    #     self.velocity += force / self.mass * self.dt
    #     self.position += self.velocity * self.dt
    #
    #     # update time
    #     self.time += self.dt




