import numpy as np
from time import time
from evolve import evolve
from const import zcnsts, kick, output, yeardy
from zcnst import zcnsts_set


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

    # 计算 zcnsts 参数
    zcnsts.z = z
    zcnsts_set(zcnsts)

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

    # print(kstar, mass0, mass, rad, lum, massc, radc, menv, renv, ospin, epoch, tms, tphys, tphysf,
    #       dtp, z, zcnsts.z, zcnsts.zpars, zcnsts.msp, zcnsts.gbp, tb, ecc, kick.f_fb, kick.meanvk, kick.sigmavk)

    # 演化双星
    evolve(kstar, mass0, mass, rad, lum, massc, radc, menv, renv,
           ospin, epoch, tms, tphys, tphysf, dtp, z, zcnsts, tb, ecc, kick, output)
    list = output.bcm[:, 1].tolist()
    num = list.index(-1)
    binary_out = np.zeros((num, 20))
    for i in range(0, num):
        binary_out[i, 0] = output.bcm[i + 1, 1]               # time
        binary_out[i, 1] = output.bcm[i + 1, 30] * yeardy     # tb
        binary_out[i, 2] = output.bcm[i + 1, 32]              # ecc
        binary_out[i, 3] = output.bcm[i + 1, 2]               # kw1
        binary_out[i, 4] = output.bcm[i + 1, 16]              # kw2
        binary_out[i, 5] = output.bcm[i + 1, 4]               # mass1
        binary_out[i, 6] = output.bcm[i + 1, 18]              # mass2
        binary_out[i, 7] = output.bcm[i + 1, 8]               # mc1
        binary_out[i, 8] = output.bcm[i + 1, 22]              # mc2
        binary_out[i, 9] = output.bcm[i + 1, 6]               # log10(r1)
        binary_out[i, 10] = output.bcm[i + 1, 20]              # log10(r2)
        binary_out[i, 11] = output.bcm[i + 1, 15]             # r1/rl1
        binary_out[i, 12] = output.bcm[i + 1, 29]             # r2/rl2
        binary_out[i, 13] = output.bcm[i + 1, 5]              # log10(L1)
        binary_out[i, 14] = output.bcm[i + 1, 19]             # log10(L2)
        binary_out[i, 15] = output.bcm[i + 1, 13]             # spin1
        binary_out[i, 16] = output.bcm[i + 1, 27]             # spin2
        binary_out[i, 17] = output.bcm[i + 1, 14]             # ml_rate1
        binary_out[i, 18] = output.bcm[i + 1, 28]             # ml_rate2
        binary_out[i, 19] = output.bcm[i + 1, 31]             # sep

    # 找到双星并合的时刻
    list_1 = binary_out[:, 2].tolist()
    num_1 = list_1.index(-1)
    # 然后用并合时刻减去演化时刻，即得并合前时间
    binary_out[:, 0] = binary_out[num_1, 0] - binary_out[:, 0]

    fmt_out = '%9.3f %9.3f %9.3f %5d %5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f' \
              ' %9.3f %9.3f %9.3f %12.3e %12.3e %12.3e %12.3e %12.3f'
    header_out = '   time        tb        ecc     kw1   kw2    mass1     mass2      mc1       mc2      lg(r1)   lg(r2)'\
                 '    r1/rl1     r2/rl2     lg(L1)    lg(L2)     spin1      spin2      ml_rate1     ml_rate2        sep'
    with open("./binary_out.txt", 'w') as f:
        np.savetxt(f, binary_out[:, :], fmt=fmt_out, header=header_out, comments='')


if __name__ == '__main__':
    start = time()
    for _ in range(1):
        bse()
    end = time()
    print('运行时间: %s 秒' % (end - start))





