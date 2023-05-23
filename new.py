from const import G, Rsun, Msun, yearsc
import numpy as np
from const import zcnsts
from zcnst import zcnsts_set
from zfuncs import rgbf, lbgbf, lHeIf
import matplotlib.pyplot as plt
from const import RNG1
import concurrent.futures


num = 100
data1 = []
data2 = []


def add():
    data1.append(RNG1.uniform())
    data2.append(np.random.rand())
    print(data1)


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = [executor.submit(add) for _ in range(int(num))]

    data1 = np.array(data1)
    data2 = np.array(data2)

    plt.hist(data1, histtype='step', color='b')
    plt.hist(data2, histtype='step', color='r')
    plt.show()


# zcnsts.z = 0.03
# zcnsts_set(zcnsts)
# print(zcnsts.zpars[3])
# # print(zcnsts.gbp)
# m = np.logspace(-1, 1.5, 100)
# # lum = np.logspace(0, 4, num=100)
# # r = rgbf(m, lum, zcnsts)
# lhei = m.copy()
# lbgb = lbgbf(m, zcnsts)
# for i in range(len(m)):
#     lhei[i] = lHeIf(m[i], zcnsts.zpars[2], zcnsts)
# plt.plot(m, lbgb, c='r')
# plt.plot(m, lhei, c='b')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


# sep_to_period = 2 * np.pi * (Rsun**3 / G / Msun) ** 0.5 / yearsc
# period_to_sep = (G * Msun * (yearsc/2/np.pi)**2)**(1/3) / Rsun
#
# M = 10
# # tb = 2
# # sep = period_to_sep * (tb**2 * M) ** (1 / 3)
# sep = 735.379
# # tb = sep_to_period * (sep ** 3 / M) ** 0.5
# # print(sep)
#
# print(sep_to_period)
# print(period_to_sep)
