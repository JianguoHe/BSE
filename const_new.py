import numpy as np
from instar import instar
from numba import float64
from numba.experimental import jitclass


# 星族合成参数
num_evolve = int(2e5)                   # 演化的双星数量
m1_min = 5                              # 恒星1的最小质量
m1_max = 50                             # 恒星1的最大质量
m2_min = 0.5                            # 恒星2的最小质量
m2_max = 50                             # 恒星2的最大质量
sep_min = 3                             # 最小轨道间距
sep_max = 1e4                           # 最大轨道间距


# 控制参数
njit_enabled = True                # 是否使用numba.njit修饰器加速程序
jitclass_enabled = njit_enabled    # 是否使用numba.jitclass修饰器加速程序
ecc_scheme = 'zero'             # 初始偏心率分布模型 [option: 'zero', 'uniform', 'thermal']
HG_survive_CE = True            # allow HG to survive CE evolution if HG_survive_CE is True
IMF_scheme = 'Kroupa1993'       # 初始质量函数 [option: 'Kroupa1993', 'Weisz2015']
wind_model = 'Belczynski'       # 星风质量损失模型【option: 'Hurley', 'Belczynski'】
mass_accretion_model = 1        # 物质吸积模型(1/2/3对应于《Shao2014》中的三个物质吸积效率模型)
mb_model = 'Hurley2002'         # 磁制动模型【'Rappaport1983', 'Hurley2002'】
gamma_mb = 4                    # 磁制动指数
max_WD_mass = 0.2               # 中子星白矮星稳定物质转移的最大WD质量


# 演化参数
SNtype = 1                              # 超新星类型(1,2,3分别对应于rapid,delayed,stochastic)
alpha = 1.0                             # 公共包层效率参数(1)
tiny = 1e-14                            # 小量
ceflag = 3                              # ceflag > 0 activates spin-energy correction in common-envelope (0).
                                        # ceflag = 3 activates de Kool common-envelope model
tflag = 1                               # tflag > 0 activates tidal circularisation (1)
wdflag = False                          # wdflag(Ture) uses modified-Mestel cooling for WDs (0).
bhflag = True                           # bhflag(Ture) allows velocity kick at BH formation (0).
mxns = 2.5                              # 最大中子星质量
pts1 = 0.05                             # 演化步长: MS(当前阶段的演化时间 * pts = 当前阶段的演化步长)
pts2 = 0.01                             # 演化步长: GB, CHeB, AGB, HeGB
pts3 = 0.02                             # 演化步长: HG, HeMS
sigma = 265.0                           # 超新星速度踢的麦克斯韦分布（ 190 km/s ）


# 星风吸积相关常数
alpha_wind = 1.5                        # Bondi-Hoyle 星风吸积因子 (3/2)
beta_wind = 0.125                       # 星风速度因子：正比于 vwind**2 (1/8)
mu_wind = 1.0                           # 星风吸积中自旋比角动量的转移效率(1)
acc1 = 3.920659e8                       # 风吸积常数

# 星风质损相关常数
neta = 0.5                              # Reimers 质量损失系数(默认为0.5)
bwind = 0.0                             # Reimers 质量损失潮汐增强参数(默认为0)
f_WR = 0.5                              # 氦星质损定标因子(范围0-1)
f_LBV = 1.5                             # LBV质损定标因子(默认为1.5)


# 随机数(双星初始质量、轨道间距、金属丰度)
rng_m1 = np.random.default_rng(1).uniform(low=np.log(m1_min), high=np.log(m1_max), size=num_evolve)
rng_m2 = np.random.default_rng(2).uniform(low=np.log(m2_min), high=np.log(m2_max), size=num_evolve)
rng_sep = np.random.default_rng(3).uniform(low=np.log(sep_min), high=np.log(sep_max), size=num_evolve)
rng_ecc = np.random.default_rng(4).integers(low=0, high=10, size=num_evolve)
rng_z = np.random.default_rng(5).integers(low=0, high=12, size=num_evolve)


# 随机数(超新星爆发时的 natal kick)
rng = np.random.default_rng(10)
rng_MA = rng.uniform(low=0, high=2 * np.pi, size=num_evolve)      # 平均近点角(mean anomaly)
rng_vkick_maxwell = rng.standard_normal(size=(num_evolve, 3))     # kick在rapid/delay SN model下服从麦克斯韦分布
gaussian_samples = rng.standard_normal(size=(num_evolve, 3))


# 数值常量
mch = 1.44                              # 钱德拉塞卡极限（太阳质量）
M_ECSN = 1.38        # ECSN爆炸极限(太阳质量)
pc = 3.08567758e18                      # 秒差距 → 厘米
yeardy = 365.25                         # 年 → 天
yearsc = 3.1557e7                       # 年 → 秒
Msun = 1.9884e33                        # 太阳质量（单位: g）
Rsun = 6.957e10                         # 太阳半径(单位: cm)
Lsun = 3.83e33                          # 太阳光度(单位: erg/s)
Zsun = 0.02                             # 太阳金属丰度
Teffsun = 5780                          # 太阳表面温度
G = 6.6743e-8                           # 引力常量(单位: cm3 * g-1 * s-2)
clight = 2.99792458e10                  # 光速(单位: cm/s)
aursun = 215.0291                       # 计算双星轨道间距中的一个常数(公式中所有单位转化为太阳单位)
sep_to_period = 0.000317148             # 开普勒定律中常数: 轨道间距(Rsun) → 轨道周期(year)
period_to_sep = 215.0263668             # 开普勒定律中常数: 轨道周期(year) → 轨道间距(Rsun)
tol = 1e-7
epsnov = 0.001
eddfac = 1.0                            # 物质转移的爱丁顿极限因子(1.0)
gamma = -2.0                            # RLOF时损失的物质带走的角动量机制

ktype = instar()


# 单个双星系统演化数据存储数组（以类的形式保存）
spec0 = [('bcm', float64[:, :]), ('bpp', float64[:, :])]
@jitclass(spec0)
class Output(object):
    def __init__(self, bcm, bpp):
        self.bcm = bcm
        self.bpp = bpp


# 依赖金属丰度的 zpars 参数（以类的形式保存）
spec1 = [('z', float64), ('zpars', float64[:]), ('msp', float64[:]), ('gbp', float64[:])]
@jitclass(spec1)
class Zcnsts(object):
    def __init__(self, z, zpars, msp, gbp):
        self.z = z
        self.zpars = zpars
        self.msp = msp
        self.gbp = gbp


# 在速度踢 kick 中用到的参数（以类的形式保存）
spec2 = [('f_fb', float64), ('meanvk', float64), ('sigmavk', float64)]
@jitclass(spec2)
class Kick(object):
    def __init__(self, f_fb, meanvk, sigmavk):
        self.f_fb = f_fb
        self.meanvk = meanvk
        self.sigmavk = sigmavk


# 计算公共包层结合能因子 lamada 所需数据
data002_1 = np.loadtxt("./lambda/z=0.02/m=1/M1.dat")
data002_2 = np.loadtxt("./lambda/z=0.02/m=2/M2.dat")
data002_4 = np.loadtxt("./lambda/z=0.02/m=4/M4.dat")
data002_6 = np.loadtxt("./lambda/z=0.02/m=6/M6.dat")
data002_8 = np.loadtxt("./lambda/z=0.02/m=8/M8.dat")
data002_10 = np.loadtxt("./lambda/z=0.02/m=10/M10.dat")
data002_20 = np.loadtxt("./lambda/z=0.02/m=20/M20.dat")
data002_30 = np.loadtxt("./lambda/z=0.02/m=30/M30.dat")
data002_40 = np.loadtxt("./lambda/z=0.02/m=40/M40.dat")
data002_60 = np.loadtxt("./lambda/z=0.02/m=60/M60.dat")
data0001_1 = np.loadtxt("./lambda/z=0.001/m=1/M1.dat")
data0001_2 = np.loadtxt("./lambda/z=0.001/m=2/M2.dat")
data0001_4 = np.loadtxt("./lambda/z=0.001/m=4/M4.dat")
data0001_6 = np.loadtxt("./lambda/z=0.001/m=6/M6.dat")
data0001_8 = np.loadtxt("./lambda/z=0.001/m=8/M8.dat")
data0001_10 = np.loadtxt("./lambda/z=0.001/m=10/M10.dat")
data0001_20 = np.loadtxt("./lambda/z=0.001/m=20/M20.dat")
data0001_30 = np.loadtxt("./lambda/z=0.001/m=30/M30.dat")
data0001_40 = np.loadtxt("./lambda/z=0.001/m=40/M40.dat")
data0001_60 = np.loadtxt("./lambda/z=0.001/m=60/M60.dat")
data00001_1 = np.loadtxt("./lambda/z=0.0001/m=1/M1.dat")
data00001_2 = np.loadtxt("./lambda/z=0.0001/m=2/M2.dat")
data00001_4 = np.loadtxt("./lambda/z=0.0001/m=4/M4.dat")
data00001_6 = np.loadtxt("./lambda/z=0.0001/m=6/M6.dat")
data00001_8 = np.loadtxt("./lambda/z=0.0001/m=8/M8.dat")
data00001_10 = np.loadtxt("./lambda/z=0.0001/m=10/M10.dat")
data00001_20 = np.loadtxt("./lambda/z=0.0001/m=20/M20.dat")
data00001_30 = np.loadtxt("./lambda/z=0.0001/m=30/M30.dat")
data00001_40 = np.loadtxt("./lambda/z=0.0001/m=40/M40.dat")
data00001_60 = np.loadtxt("./lambda/z=0.0001/m=60/M60.dat")
