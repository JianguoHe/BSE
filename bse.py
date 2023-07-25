import numpy as np
import pandas as pd
from time import time
from evolve import evolve
from const import Zcnsts, Kick, Output, yeardy, num_evolve
from const import Msun, Rsun, yearsc
from zcnst import zcnsts_set
from utils import cal_jorb


def bse():
    # 读取初始参数(部分，剩余的参数存储在const文件中供子程序调用)
    data = np.loadtxt("binary_in.txt")
    mass0 = np.array([0, data[0, 0], data[0, 1]])    # 初始双星质量
    tb = data[0, 2]                                  # 初始轨道周期（天）
    z = data[0, 3]                                   # 金属丰度
    tphysf = data[0, 4]                              # 最大演化时间（Myr）
    kstar = np.array([0, data[0, 5], data[0, 6]])    # 恒星初始类型
    ecc = data[0, 7]                                 # 初始偏心率

    # 设置初始参数
    tphys = 0.0
    mass = mass0.copy()                # 当前双星质量
    epoch = np.array([0.0, 0, 0])      #
    ospin = epoch.copy()               # 双星自旋

    # 初始化Zcnsts的实例, 并计算实例参数
    zpars = np.zeros((1, 20)).flatten()
    msp = np.zeros((1, 200)).flatten()
    gbp = np.zeros((1, 200)).flatten()
    zcnsts = Zcnsts(z, zpars, msp, gbp)
    zcnsts_set(zcnsts)

    # 初始化Kick类的实例
    f_fb = 0.0
    meanvk = 0.0
    sigmavk = 0.0
    kick = Kick(f_fb, meanvk, sigmavk)

    # 初始化Output类的实例
    bcm = np.zeros((50001, 40))
    bpp = np.zeros((81, 11))
    output = Output(bcm, bpp)

    # 未初始化参数
    rad = epoch.copy()
    lum = epoch.copy()
    massc = epoch.copy()
    radc = epoch.copy()
    menv = epoch.copy()
    renv = epoch.copy()
    tms = epoch.copy()                 # 主序时间

    # 设置数据存储参数：如果 dtp = 0 , 数据将在每一次迭代都被存储。
    dtp = 0.0

    # SN_index = 13369
    SN_index = np.random.randint(0, num_evolve)

    # 演化双星
    evolve(kstar, mass0, mass, rad, lum, massc, radc, menv, renv, ospin, epoch, tms, tphys, tphysf, dtp, z,
           zcnsts, tb, ecc, kick, output, index=SN_index)

    column_names = ['time', 'kw1', 'mass1_initial', 'mass1', 'lg(L1)', 'lg(r1)', 'lg(T1)', 'mc1', 'rc1',
                    'menv1', 'renv1', 'epoch1', 'spin1', 'mdot1', 'r1/rl1', 'kw2', 'mass2_initial', 'mass2', 'lg(L2)',
                    'lg(r2)', 'lg(T2)', 'mc2', 'rc2', 'menv2', 'renv2', 'epoch2', 'spin2', 'mdot2', 'r2/rl2',
                    'tb', 'sep', 'ecc', 'CE', 'jorb', 'jdot_mb', 'jdot_gr', 'None3', 'None4', 'None5']

    result = output.bcm[:np.where(output.bcm[:, 1] == -1)[0][0], 1:]

    df = pd.DataFrame(result, columns=column_names)
    df['tb'] = df['tb'] * yeardy
    # df['J_orb'] = 1
    df['jorb'] = df['jorb'] * Msun * Rsun ** 2 / yearsc
    df['jdot_mb'] = df['jdot_mb'] * Msun * Rsun ** 2 / yearsc ** 2
    df['jdot_gr'] = df['jdot_gr'] * Msun * Rsun ** 2 / yearsc ** 2
    output_column = ['time', 'tb', 'ecc', 'kw1', 'kw2', 'mass1', 'mass2', 'mc1', 'mc2', 'menv1', 'menv2', 'CE',
                     'r1/rl1', 'r2/rl2', 'lg(r1)', 'lg(r2)', 'lg(L1)', 'lg(L2)',
                     'spin1', 'spin2', 'mdot1', 'mdot2', 'sep', 'jorb', 'jdot_mb', 'jdot_gr']
    df = df[output_column]

    float_list = ['time', 'tb', 'ecc', 'mass1', 'mass2', 'mc1', 'mc2', 'menv1', 'menv2',
                  'lg(r1)', 'lg(r2)', 'r1/rl1', 'r2/rl2', 'lg(L1)', 'lg(L2)']
    int_list = ['kw1', 'kw2', 'CE']
    sci_list = ['spin1', 'spin2', 'mdot1', 'mdot2', 'sep', 'jorb', 'jdot_mb', 'jdot_gr']
    df[float_list] = df[float_list].round(3)
    df[int_list] = df[int_list].astype(int)
    df[sci_list] = df[sci_list].applymap('{:.3e}'.format)
    df.to_csv('./binary_out.csv', index=False)


if __name__ == '__main__':
    start = time()
    for _ in range(1):
        bse()
    end = time()
    print('运行时间: %s 秒' % (end - start))





