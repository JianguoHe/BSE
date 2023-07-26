from const import G, Rsun, Msun, yearsc
import numpy as np
import random
from zcnst import zcnsts_set
from zfuncs import rgbf, lbgbf, lHeIf
import matplotlib.pyplot as plt
import concurrent.futures
from numba.experimental import jitclass
from numba import types
from numba import float64, njit
import time
from utils import rochelobe, add
from numba import float64

jitclass_enabled = True



spec = [
    ('x', float64[:, :]),
    ('y', float64),
]
@conditional_jitclass(spec)
class MyClass:
    def __init__(self, x):
        self.x = x
        self.y = 0

    def go_fast(self):  # Function is compiled and runs in machine code
        trace = 0.0
        for i in range(self.x.shape[0]):
            trace += np.tanh(self.x[i, i])

    def slow(self):
        trace = 0.0
        for i in range(self.x.shape[0]):
            trace += rochelobe(0.1)
        add(self)


x = np.random.rand(10000, 10000)

# 测试不加njit修饰器的compute_sum方法的运算速度
my_class = MyClass(x=x)
start = time.time()
result = my_class.slow()
end = time.time()
print("compute_sum elapsed time: ", end - start)

start = time.time()
result1 = my_class.slow()
end = time.time()
print("compute_sum elapsed time: ", end - start)

print(my_class.y)



# sigma = 1.0  # 麦克斯韦分布的标准差
# n = 1000000
# z1 = np.random.randn(n)  # 生成第一个标准正态分布随机数
# z2 = np.random.randn(n)  # 生成第二个标准正态分布随机数
# x = sigma * np.sqrt(z1**2 + z2**2)  # 计算麦克斯韦分布的随机值
#
# u1 = np.random.rand(n)
# s = sigma * np.sqrt(-2.0 * np.log(1.0 - u1))
#
# plt.hist(x, bins=100, color='blue', label='x', alpha=0.5, histtype='step')
# plt.hist(s, bins=100, color='red', label='s', alpha=0.5, histtype='step')
# plt.legend()
# plt.show()

# rng_z = np.random.default_rng(4).integers(low=0, high=12, size=20)
# z_value = np.array([0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.0125, 0.015, 0.0175, 0.02])
# z = z_value[rng_z[3]]
# print(z)

# num_evolve = int(12e6)
# a = num_evolve // 100
# print(a)
# mu = 5  # 均值
# sigma = 2  # 标准差
# seed = 1  # 设置随机数生成器的种子
#
# # 生成使用 random.gauss 方法的随机数序列
# random.seed(seed)
# random_numbers_gauss = [random.gauss(mu, sigma) for _ in range(500000)]
#
# # 生成使用随机数生成器的随机数序列
# rng = np.random.default_rng(seed)
# random_numbers_rng = rng.normal(mu, sigma, 500000)
#
# # 绘制直方图
# plt.hist(random_numbers_gauss, bins=30, alpha=0.5, label='random.gauss')
# plt.hist(random_numbers_rng, bins=30, alpha=0.5, label='Random number generator')
# # plt.legend(loc='upper right')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Comparison of Random Number Generation')
# plt.show()





#
#
# if __name__ == '__main__':
#     num_processes = 4
#     num_executions = 10000
#
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
#         seeds = range(1, num_executions + 1)
#
#         # 使用map方法执行多次popbin函数，并收集结果
#         results = list(executor.map(popbin, seeds))
#
#     # 绘制直方图
#     plt.hist(results, bins=10)
#     plt.xlabel('Random Number')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Random Numbers')
#     plt.show()

# rng = np.random.default_rng(1)
# data = rng.random(500000)
# plt.hist(data)
# plt.show()

# num = 10
#
# data = np.zeros(shape=(10, ))
# print()
#
#
# def add(i):
#     data[i] = RNG1.random()
#
#
# if __name__ == '__main__':
#
#     # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#     #     results = [executor.submit(add) for _ in range(int(num))]
#     for i in range(num):
#         add(i)
#
#     print(data)

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
