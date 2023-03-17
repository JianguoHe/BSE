import numpy as np
from numba import njit


# 设置碰撞矩阵, 也就是设置双星碰撞后合并产生的新星类型, 对于有类巨星参与的碰撞, 新星类型会加100表示双星会在公共包层演化中发生碰撞
@njit
def instar():
    ktype = np.zeros((15, 15))
    ktype[0, 0] = 1
    for j in range(1, 7):
        ktype[0, j] = j
        ktype[1, j] = j

    ktype[0, 7] = 4
    ktype[1, 7] = 4

    for j in range(8, 13):
        if j != 10:
            ktype[0, j] = 6
        else:
            ktype[0, j] = 3
        ktype[1, j] = ktype[0, j]

    ktype[2, 2] = 3
    for i in range(3, 15):
        ktype[i, i] = i
    ktype[5, 5] = 4
    ktype[7, 7] = 1
    ktype[10, 10] = 15
    ktype[13, 13] = 14
    for i in range(2, 6):
        for j in range(i + 1, 13):
            ktype[i, j] = 4

    ktype[2, 3] = 3
    ktype[2, 6] = 5
    ktype[2, 10] = 3
    ktype[2, 11] = 5
    ktype[2, 12] = 5
    ktype[3, 6] = 5
    ktype[3, 10] = 3
    ktype[3, 11] = 5
    ktype[3, 12] = 5
    ktype[6, 7] = 4
    ktype[6, 8] = 6
    ktype[6, 9] = 6
    ktype[6, 10] = 5
    ktype[6, 11] = 6
    ktype[6, 12] = 6
    ktype[7, 8] = 8
    ktype[7, 9] = 9
    ktype[7, 10] = 7
    ktype[7, 11] = 9
    ktype[7, 12] = 9
    ktype[8, 9] = 9
    ktype[8, 10] = 7
    ktype[8, 11] = 9
    ktype[8, 12] = 9
    ktype[9, 10] = 7
    ktype[9, 11] = 9
    ktype[9, 12] = 9
    ktype[10, 11] = 9
    ktype[10, 12] = 9
    ktype[11, 12] = 12
    for i in range(0, 13):
        ktype[i, 13] = 13
        ktype[i, 14] = 14
    ktype[13, 14] = 14

    # Increase common-envelope cases by 100.
    for i in range(0, 10):
        for j in range(i, 15):
            if i <= 1 or i == 7:
                if 2 <= j <= 9 and j != 7:
                    ktype[i, j] = ktype[i, j] + 100
            else:
                ktype[i, j] = ktype[i, j] + 100

    # Assign the remaining values by symmetry.
    for i in range(1, 15):
        for j in range(0, i):
            ktype[i, j] = ktype[j, i]

    return ktype


