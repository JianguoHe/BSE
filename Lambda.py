import numpy as np
from numba import njit
from const import data002_1, data002_2, data002_4, data002_6, data002_8
from const import data002_10, data002_20, data002_30, data002_40, data002_60
from const import data0001_1, data0001_2, data0001_4, data0001_6, data0001_8
from const import data0001_10, data0001_20, data0001_30, data0001_40, data0001_60
from const import data00001_1, data00001_2, data00001_4, data00001_6, data00001_8
from const import data00001_10, data00001_20, data00001_30, data00001_40, data00001_60


# 公共包层结合能因子λ的计算可以参考文章: 'ON THE BINDING ENERGY PARAMETER λ OF COMMON ENVELOPE EVOLUTION'
@njit
def lambda_cal(m, r, z):
    # 设一些初始值
    Lambda = 1
    Lambda1 = 1
    Lambda2 = 1
    Lambda3 = 1
    r11 = np.zeros((1, 2001)).flatten()
    lg1 = r11.copy()
    lb1 = r11.copy()
    r12 = r11.copy()
    lg2 = r11.copy()
    lb2 = r11.copy()
    r13 = r11.copy()
    lg3 = r11.copy()
    lb3 = r11.copy()

    if 0.1 <= m < 1.5:       # 这里改了一下，原来是0.8 <= m01 < 1.5，后来发现如果 m01<0.8,会使得 Lambda1= 0,后面报错
        for i in range(1, 2001):
            r11[i] = data002_1[i - 1, 0]      # 002_1 表示 z=0.02/m=1，此后类推
            lg1[i] = data002_1[i - 1, 1]
            lb1[i] = data002_1[i - 1, 2]
    elif 1.5 <= m < 3.0:
        for i in range(1, 2001):
            r11[i] = data002_2[i - 1, 0]
            lg1[i] = data002_2[i - 1, 1]
            lb1[i] = data002_2[i - 1, 2]
    elif 3.0 <= m < 5.0:
        for i in range(1, 2001):
            r11[i] = data002_4[i - 1, 0]
            lg1[i] = data002_4[i - 1, 1]
            lb1[i] = data002_4[i - 1, 2]
    elif 5.0 <= m < 7.0:
        for i in range(1, 2001):
            r11[i] = data002_6[i - 1, 0]
            lg1[i] = data002_6[i - 1, 1]
            lb1[i] = data002_6[i - 1, 2]
    elif 7.0 <= m < 9.0:
        for i in range(1, 2001):
            r11[i] = data002_8[i - 1, 0]
            lg1[i] = data002_8[i - 1, 1]
            lb1[i] = data002_8[i - 1, 2]
    elif 9.0 <= m < 15.0:
        for i in range(1, 2001):
            r11[i] = data002_10[i - 1, 0]
            lg1[i] = data002_10[i - 1, 1]
            lb1[i] = data002_10[i - 1, 2]
    elif 15.0 <= m < 25.0:
        for i in range(1, 2001):
            r11[i] = data002_20[i - 1, 0]
            lg1[i] = data002_20[i - 1, 1]
            lb1[i] = data002_20[i - 1, 2]
    elif 25.0 <= m < 35.0:
        for i in range(1, 2001):
            r11[i] = data002_30[i - 1, 0]
            lg1[i] = data002_30[i - 1, 1]
            lb1[i] = data002_30[i - 1, 2]
    elif 35.0 <= m < 50.0:
        for i in range(1, 2001):
            r11[i] = data002_40[i - 1, 0]
            lg1[i] = data002_40[i - 1, 1]
            lb1[i] = data002_40[i - 1, 2]
    elif 50.0 <= m < 100.0:
        for i in range(1, 2001):
            r11[i] = data002_60[i - 1, 0]
            lg1[i] = data002_60[i - 1, 1]
            lb1[i] = data002_60[i - 1, 2]

    for j in range(1, 2000):
        if r11[j] < r <= r11[j + 1]:
            Lambda1 = lb1[j]
        elif r > r11[1999]:
            Lambda1 = lb1[1999]
        elif r <= r11[1]:
            Lambda1 = lb1[1]

    if 0.1 <= m < 1.5:
        for i in range(1, 2001):
            r12[i] = data0001_1[i - 1, 0]
            lg2[i] = data0001_1[i - 1, 1]
            lb2[i] = data0001_1[i - 1, 2]
    elif 1.5 <= m < 3.0:
        for i in range(1, 2001):
            r12[i] = data0001_2[i - 1, 0]
            lg2[i] = data0001_2[i - 1, 1]
            lb2[i] = data0001_2[i - 1, 2]
    elif 3.0 <= m < 5.0:
        for i in range(1, 2001):
            r12[i] = data0001_4[i - 1, 0]
            lg2[i] = data0001_4[i - 1, 1]
            lb2[i] = data0001_4[i - 1, 2]
    elif 5.0 <= m < 7.0:
        for i in range(1, 2001):
            r12[i] = data0001_6[i - 1, 0]
            lg2[i] = data0001_6[i - 1, 1]
            lb2[i] = data0001_6[i - 1, 2]
    elif 7.0 <= m < 9.0:
        for i in range(1, 2001):
            r12[i] = data0001_8[i - 1, 0]
            lg2[i] = data0001_8[i - 1, 1]
            lb2[i] = data0001_8[i - 1, 2]
    elif 9.0 <= m < 15.0:
        for i in range(1, 2001):
            r12[i] = data0001_10[i - 1, 0]
            lg2[i] = data0001_10[i - 1, 1]
            lb2[i] = data0001_10[i - 1, 2]
    elif 15.0 <= m < 25.0:
        for i in range(1, 2001):
            r12[i] = data0001_20[i - 1, 0]
            lg2[i] = data0001_20[i - 1, 1]
            lb2[i] = data0001_20[i - 1, 2]
    elif 25.0 <= m < 35.0:
        for i in range(1, 2001):
            r12[i] = data0001_30[i - 1, 0]
            lg2[i] = data0001_30[i - 1, 1]
            lb2[i] = data0001_30[i - 1, 2]
    elif 35.0 <= m < 50.0:
        for i in range(1, 2001):
            r12[i] = data0001_40[i - 1, 0]
            lg2[i] = data0001_40[i - 1, 1]
            lb2[i] = data0001_40[i - 1, 2]
    elif 50.0 <= m < 100.0:
        for i in range(1, 2001):
            r12[i] = data0001_60[i - 1, 0]
            lg2[i] = data0001_60[i - 1, 1]
            lb2[i] = data0001_60[i - 1, 2]

    for j in range(1, 2000):
        if r12[j] < r <= r12[j + 1]:
            Lambda2 = lb2[j]
        elif r > r12[1999]:
            Lambda2 = lb2[1999]
        elif r <= r12[1]:
            Lambda2 = lb2[1]

    if 0.1 <= m < 1.5:
        for i in range(1, 2001):
            r13[i] = data00001_1[i - 1, 0]
            lg3[i] = data00001_1[i - 1, 1]
            lb3[i] = data00001_1[i - 1, 2]
    elif 1.5 <= m < 3.0:
        for i in range(1, 2001):
            r13[i] = data00001_2[i - 1, 0]
            lg3[i] = data00001_2[i - 1, 1]
            lb3[i] = data00001_2[i - 1, 2]
    elif 3.0 <= m < 5.0:
        for i in range(1, 2001):
            r13[i] = data00001_4[i - 1, 0]
            lg3[i] = data00001_4[i - 1, 1]
            lb3[i] = data00001_4[i - 1, 2]
    elif 5.0 <= m < 7.0:
        for i in range(1, 2001):
            r13[i] = data00001_6[i - 1, 0]
            lg3[i] = data00001_6[i - 1, 1]
            lb3[i] = data00001_6[i - 1, 2]
    elif 7.0 <= m < 9.0:
        for i in range(1, 2001):
            r13[i] = data00001_8[i - 1, 0]
            lg3[i] = data00001_8[i - 1, 1]
            lb3[i] = data00001_8[i - 1, 2]
    elif 9.0 <= m < 15.0:
        for i in range(1, 2001):
            r13[i] = data00001_10[i - 1, 0]
            lg3[i] = data00001_10[i - 1, 1]
            lb3[i] = data00001_10[i - 1, 2]
    elif 15.0 <= m < 25.0:
        for i in range(1, 2001):
            r13[i] = data00001_20[i - 1, 0]
            lg3[i] = data00001_20[i - 1, 1]
            lb3[i] = data00001_20[i - 1, 2]
    elif 25.0 <= m < 35.0:
        for i in range(1, 2001):
            r13[i] = data00001_30[i - 1, 0]
            lg3[i] = data00001_30[i - 1, 1]
            lb3[i] = data00001_30[i - 1, 2]
    elif 35.0 <= m < 50.0:
        for i in range(1, 2001):
            r13[i] = data00001_40[i - 1, 0]
            lg3[i] = data00001_40[i - 1, 1]
            lb3[i] = data00001_40[i - 1, 2]
    elif 50.0 <= m < 100.0:
        for i in range(1, 2001):
            r13[i] = data00001_60[i - 1, 0]
            lg3[i] = data00001_60[i - 1, 1]
            lb3[i] = data00001_60[i - 1, 2]

    for j in range(1, 2000):
        if r13[j] < r <= r13[j + 1]:
            Lambda3 = lb3[j]
        elif r > r13[1999]:
            Lambda3 = lb3[1999]
        elif r <= r13[1]:
            Lambda3 = lb3[1]

    if z > 0.02:
        Lambda = Lambda1
    elif 0.001 < z <= 0.02:
        Lambda = Lambda1 + (0.02 - z) / (0.02 - 0.001) * (Lambda2 - Lambda1)
    elif 0.0001 < z <= 0.001:
        Lambda = Lambda2 + (0.001 - z) / (0.001 - 0.0001) * (Lambda3 - Lambda2)

    return Lambda
