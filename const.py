import numpy as np
from instar import instar
from numba import float64
from numba.experimental import jitclass


# 星族合成参数
num_evolve = 1e5                  # 演化的双星数量
m1_min = 5                        # 恒星1的最小质量
m1_max = 50                       # 恒星1的最大质量
m2_min = 0.5                      # 恒星2的最小质量
m2_max = 50                       # 恒星2的最大质量
sep_min = 3                       # 最小轨道间距
sep_max = 1e4                     # 最大轨道间距


# 演化参数
alpha = 1.0                       # 公共包层效率参数(1)
SNtype = 1                        # 超新星类型(1,2,3分别对应于rapid,delayed,stochastic)
tiny = 1e-14                      # 小量
hewind = 1.0                      # 氦星质损因子（通常为 1 ）
ceflag = 3                        # ceflag > 0 activates spin-energy correction in common-envelope (0).
                                  # ceflag = 3 activates de Kool common-envelope model
tflag = 1                         # tflag > 0 activates tidal circularisation (1)
ifflag = False                    # ifflag(Ture) uses WD IFMR of HPE, 1995, MNRAS, 272, 800 (0).
wdflag = False                    # wdflag(Ture) uses modified-Mestel cooling for WDs (0).
bhflag = True                     # bhflag(Ture) allows velocity kick at BH formation (0).
nsflag = True                     # nsflag(Ture) takes NS/BH mass from Fryer et al. 2012, ApJ, 749, 91.
mxns = 2.5                        # 最大中子星质量
pts1 = 0.05                       # 演化步长: MS(当前阶段的演化时间 * pts = 当前阶段的演化步长)
pts2 = 0.01                       # 演化步长: GB, CHeB, AGB, HeGB
pts3 = 0.02                       # 演化步长: HG, HeMS
sigma = 265.0                     # 超新星速度踢的麦克斯韦分布（ 190 km/s ）
mb_model = 'Rappaport1983'        # 磁制动模型【'Hurley2002', 'Rappaport1983'】
mb_gamma = 3                      # 磁制动指数

# 数值常量
mch = 1.44           # 钱德拉塞卡极限（太阳质量）
pc = 3.08567758e18   # 秒差距 → 厘米
yeardy = 365.25
yearsc = 3.1557e7
Msun = 1.9884e33     # 太阳质量（克）
Rsun = 6.957e10      # 太阳半径(单位: 厘米)
G = 6.6743e-8        # 引力常量(厘米克秒制)
c = 2.99792458e10    # 光速(厘米克秒制)
aursun = 215.0291    # 计算双星轨道间距中的一个常数(公式中所有单位转化为太阳单位)
tol = 1e-7
epsnov = 0.001
eddfac = 1.0         # 物质转移的爱丁顿极限因子(1.0)
gamma = -2.0
ktype = instar()


# 星风质量损失相关常数
alpha_wind = 1.5         # Bondi-Hoyle 星风吸积因子 (3/2)
beta_wind = 0.125        # 星风速度因子：正比于 vwind**2 (1/8)
bwind = 0.0              # 星风增强质损参数（由于双星的潮汐作用）
neta = 0.5               # Reimers 质量损失系数，（通常为 0.5 ）
acc1 = 3.920659e8        # 风吸积常数
xi = 1.0                 # 星风吸积中自旋比角动量的转移效率(1)


# 星族合成数据数组（以类的形式保存）
spec = [('BH_BH', float64[:, :]), ('BH_NS', float64[:, :]), ('BH_WD', float64[:, :]),
        ('NS_NS', float64[:, :]), ('NS_WD', float64[:, :]), ('WD_WD', float64[:, :]), ('Merger', float64[:, :])]
@jitclass(spec)
class Find_BH_CS(object):
    def __init__(self, BH_BH, BH_NS, BH_WD, NS_NS, NS_WD, WD_WD, Merger):
        self.BH_BH = BH_BH
        self.BH_NS = BH_NS
        self.BH_WD = BH_WD
        self.NS_NS = NS_NS
        self.NS_WD = NS_WD
        self.WD_WD = WD_WD
        self.Merger = Merger


BH_BH = np.empty(shape=(0, 25))
BH_NS = BH_BH.copy()
BH_WD = BH_BH.copy()
NS_NS = BH_BH.copy()
NS_WD = BH_BH.copy()
WD_WD = BH_BH.copy()
Merger = np.empty(shape=(0, 20))
find = Find_BH_CS(BH_BH, BH_NS, BH_WD, NS_NS, NS_WD, WD_WD, Merger)


# 单个双星系统演化数据存储数组（以类的形式保存）
spec0 = [('bcm', float64[:, :]), ('bpp', float64[:, :])]
@jitclass(spec0)
class Output(object):
    def __init__(self, bcm, bpp):
        self.bcm = bcm
        self.bpp = bpp


bcm = np.zeros((50001, 35))
bpp = np.zeros((81, 11))
output = Output(bcm, bpp)


# 依赖金属丰度的 zpars 参数（以类的形式保存）
spec1 = [('z', float64), ('zpars', float64[:]), ('msp', float64[:]), ('gbp', float64[:])]
@jitclass(spec1)
class Zcnsts(object):
    def __init__(self, z, zpars, msp, gbp):
        self.z = z
        self.zpars = zpars
        self.msp = msp
        self.gbp = gbp


z = 0.0
zpars = np.zeros((1, 20)).flatten()
msp = np.zeros((1, 200)).flatten()
gbp = np.zeros((1, 200)).flatten()
zcnsts = Zcnsts(z, zpars, msp, gbp)


# 在速度踢 kick 中用到的参数（以类的形式保存）
spec2 = [('f_fb', float64), ('meanvk', float64), ('sigmavk', float64)]
@jitclass(spec2)
class Kick(object):
    def __init__(self, f_fb, meanvk, sigmavk):
        self.f_fb = f_fb
        self.meanvk = meanvk
        self.sigmavk = sigmavk


f_fb = 0.0
meanvk = 0.0
sigmavk = 0.0
kick = Kick(f_fb, meanvk, sigmavk)


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



