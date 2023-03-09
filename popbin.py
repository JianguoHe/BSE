import random
import numpy as np
from numba import njit
from time import time, strftime, localtime
from evolve import evolve
from zcnst import zcnsts_set
from zfuncs import weight, SFR
from const import yeardy, zcnsts, kick, output, find, SNtype, G, Msun, Rsun, num_evolve, alpha
import concurrent.futures
import pysnooper


@njit
def Find(zcnsts, kick, output, find):
    # 设置初始参数(双星质量、轨道间距)
    a = random.uniform(np.log(0.5), np.log(100))
    b = random.uniform(np.log(0.5), np.log(100))
    c = random.uniform(np.log(3), np.log(10000))
    m1 = np.exp(a)  # 范围是(0.5, 100)
    m2 = np.exp(b)  # 范围是(0.5, 100)
    sep = np.exp(c)  # 双星间距，范围是(3, 10000)

    # 初始轨道周期(单位: 天), 范围为 0.04 - 49393
    tb = 2 * np.pi * (sep * Rsun) ** (3 / 2) * (G * Msun * (m1 + m2)) ** (-1 / 2) / (3600 * 24)

    # 排除 m1 小于 m2 的情况
    if m1 < m2:
        return

    # 应用【Alex. J. Kemp，21】中的 12-grid, 在固定值中随机取金属丰度, 同时记录金属丰度所在区间的起止时间以及间隔
    z_value = np.array([0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.0125, 0.015, 0.0175, 0.02])
    z = np.random.choice(z_value)
    ind = np.abs(z - z_value).argmin()
    time_start = np.array([0.0, 80, 200, 450, 700, 1200, 2000, 3333, 5000, 6666, 8333, 10000])[ind]
    time_end = np.array([80.0, 200, 450, 700, 1200, 2000, 3333, 5000, 6666, 8333, 10000, 14000])[ind]
    time_interval = time_end - time_start

    # 计算双星的权重
    Wb = weight(m1, m2)

    # 计算 zcnsts 参数
    zcnsts.z = z
    zcnsts_set(zcnsts)

    # 最大演化时间为 M31 星系年龄减去诞生时刻(Myr)
    tphysf = 14000 - time_start
    tphys = 0.0

    # 初始偏心率
    ecc = 0.0

    # 初始化参数
    kstar = np.array([0.0, 1, 1])
    mass0 = np.array([0.0, m1, m2])  # 初始双星质量
    mass = np.array([0.0, m1, m2])  # 当前双星质量

    # 设置双星初始自旋(若在零时刻为0, 则evolve会自动设置合适的值; 若自旋值大于0, 则会在任意时刻采用该值; 若自旋值小于0, 则与轨道公转)
    ospin = np.array([0.0, 0, 0])

    # 设置其他一些值
    epoch = ospin.copy()
    tms = ospin.copy()  # 主序时间
    rad = ospin.copy()
    lum = ospin.copy()
    massc = ospin.copy()
    radc = ospin.copy()
    menv = ospin.copy()
    renv = ospin.copy()

    # 设置数据存储参数：如果 dtp = 0 , 数据将在每一次迭代都被存储
    dtp = 0.0

    # 演化双星
    evolve(kstar, mass0, mass, rad, lum, massc, radc, menv, renv,
           ospin, epoch, tms, tphys, tphysf, dtp, z, zcnsts, tb, ecc, kick, output)

    # 首先排除原恒星在进入主序之前就开始物质转移（通常出现在分子云盘中）
    if 0.0 < output.bcm[1, 15] < 1.0 and 0.0 < output.bcm[1, 29] < 1.0:
        jj = 0
        t1 = -1.0
        t2 = -1.0
        t3 = -1.0

        while True:
            jj = jj + 1
            if output.bcm[jj, 1] < 0.0:
                return
            kw = int(output.bcm[jj, 2])
            kw2 = int(output.bcm[jj, 16])

            # 改变轨道周期的单位(年 → 天)
            output.bcm[jj, 30] = output.bcm[jj, 30] * yeardy

            # 寻找双星(致密星 + 非简并星)
            if (kw >= 10 or kw2 >= 10) and (kw <= 6 or kw2 <= 6) and output.bcm[jj, 30] <= 1e5 and \
                    0.0 <= output.bcm[jj, 32] < 1.0:
                # 只记录一次(演化到致密星 + 非简并星时的初始状态)
                if t1 < 0.0:
                    t1 = output.bcm[jj, 1]
                    kwx1 = kw
                    kwx2 = kw2
                    mx1 = output.bcm[jj, 4]
                    mx2 = output.bcm[jj, 18]
                    tbx = output.bcm[jj, 30]
                    eccx = output.bcm[jj, 32]

            # 寻找双致密星
            if 10 <= kw <= 14 and 10 <= kw2 <= 14 and output.bcm[jj, 30] < 1e4 and 0.0 <= output.bcm[jj, 32] < 1.0:
                # 记录多次(演化到黑洞 + 致密星后的每一步状态)
                if t1 > 0.0:
                    # 这里记录一下刚成为双致密星的时间, 用于比对Fortran的输出结果以验证程序的正确性
                    if t2 < 0:
                        t1 = output.bcm[jj, 1]
                    t2 = output.bcm[jj, 1]
                    # 由于很小概率下kw会小于kw2, 且有时双白矮星也会出现m1小于m2, 不方便后面统计数量，所以这里作一个调整
                    if kw > kw2:
                        mx11 = output.bcm[jj, 4]
                        mx22 = output.bcm[jj, 18]
                    elif kw == kw2:
                        if output.bcm[jj, 4] >= output.bcm[jj, 18]:
                            mx11 = output.bcm[jj, 4]
                            mx22 = output.bcm[jj, 18]
                        else:
                            mx11 = output.bcm[jj, 18]
                            mx22 = output.bcm[jj, 4]
                    else:
                        kw, kw2 = kw2, kw
                        mx11 = output.bcm[jj, 18]
                        mx22 = output.bcm[jj, 4]
                    kwx11 = kw
                    kwx22 = kw2
                    tbx1 = output.bcm[jj, 30]
                    eccx1 = output.bcm[jj, 32]
                    dtt = output.bcm[jj, 1] - output.bcm[jj - 1, 1]
                    if 14000 - t2 + dtt > time_end:
                        dtt = time_end + t2 - 14000

                    # 若 t2 = now(即此时看到双星), 则其对应的诞生时间(星系时间)为
                    Time = 14000 - t2

                    # 计算诞生率(每百万年)和数量
                    rate = Wb * SFR(Time)
                    num = rate * dtt

                    # 初步计算信噪比，在后面的数据处理中可根据情况需要重新计算
                    (fgw, hc, SNR) = (1.0, 1.0, 1.0)

                    # 保存致密双星的初始参数及典型演化轨迹
                    CS_CS = np.array([m1, m2, tb, z, tphysf, mx1, mx2, tbx, eccx, mx11, mx22, tbx1, eccx1,
                                      kwx1, kwx2, kw, kw2, fgw, hc, SNR, t1, t2, dtt, Wb, num]).reshape(1, 25)

                    # 确保找到的目标双星可以在特定环境(金属丰度)中形成
                    if t2 > tphysf - time_interval:
                        # 寻找双黑洞
                        if kw == 14 and kw2 == 14:
                            find.BH_BH = np.append(find.BH_BH, CS_CS, axis=0)
                        # 寻找黑洞-中子星
                        elif kw == 14 and kw2 == 13:
                            find.BH_NS = np.append(find.BH_NS, CS_CS, axis=0)
                        # 寻找黑洞-白矮星(未排除吸积致坍缩AIC黑洞)
                        elif kw == 14 and 10 <= kw2 <= 12:
                            find.BH_WD = np.append(find.BH_WD, CS_CS, axis=0)
                        # 寻找双中子星
                        elif kw == 13 and kw2 == 13:
                            find.NS_NS = np.append(find.NS_NS, CS_CS, axis=0)
                        # 寻找中子星-白矮星
                        elif kw == 13 and 10 <= kw2 <= 12:
                            find.NS_WD = np.append(find.NS_WD, CS_CS, axis=0)
                        # 寻找双白矮星
                        elif 10 <= kw <= 12 and 10 <= kw2 <= 12:
                            find.WD_WD = np.append(find.WD_WD, CS_CS, axis=0)

            # 寻找并合的黑洞-致密星双星系统
            if kw == 15 or kw2 == 15:
                if t2 > 0.0 and t3 < 0.0:
                    t3 = output.bcm[jj, 1]
                    Merger = np.array([m1, m2, tb, z, tphysf, mx1, mx2, tbx, eccx, mx11, mx22, tbx1, eccx1,
                                       kwx1, kwx2, kwx11, kwx22, t1, t3, Wb]).reshape(1, 20)
                    find.Merger = np.append(find.Merger, Merger, axis=0)
                return
    else:
        return


count = 0  # 计数用
SN = 'rapid' if SNtype == 1 else ('delayed' if SNtype == 2 else 'stochastic')
alpha_value = 'alpha_low' if alpha < 1 else ('alpha' if alpha == 1 else 'alpha_high')
path = './data/' + SN + '/' + alpha_value + '/'


# @pysnooper.snoop()
def popbin():
    # 用于打印执行次数
    global count
    count = count + 1
    if count == 2:
        print("开始啦！")
    if count % 10000 == 0.0:
        print(count)
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))

    # 开始寻找目标源
    Find(zcnsts, kick, output, find)

    # 为了方便查看结果，这里先对输出格式做一些控制，例如9.3表示占九个字符且有三位有效数字，e表示科学计数法，f表示浮点数，d为整数
    fmt_CS_CS = '%9.3f %9.3f %11.3f %9.4f %12.3f %9.3f %9.3f %12.3e %12.3e %9.3f %9.3f %12.3e %15.6e' \
                '%5d %5d %4d %5d %12.3e %12.3e %9.3f %12.3f %12.3f %12.3e %12.3e %12.3e'
    fmt_Merger = '%9.3f %9.3f %11.3f %9.4f %12.3f %9.3f %9.3f %11.3f %9.3f %9.3f %9.3f %12.3e %12.3e' \
                 '%6d %5d %6d %5d %12.3f %12.3f %12.3e'
    CS_CS = ['BH_BH', 'BH_NS', 'BH_WD', 'NS_NS', 'NS_WD', 'WD_WD']

    # 如果找到，记录相关初始参数
    for i in range(6):
        if len(list(getattr(find, CS_CS[i]))) != 0:
            with open(path + CS_CS[i] + ".txt", 'ab') as f:
                np.savetxt(f, getattr(find, CS_CS[i]), fmt=fmt_CS_CS)

    if len(list(find.Merger)) != 0:
        with open(path + "Merger.txt", 'ab') as f:
            np.savetxt(f, find.Merger[:, :], fmt=fmt_Merger)

    # 初始化find实例(说明: 由于popbin函数未实用修饰器njit, 因此这里可以直接改变类的实例而无需作为popbin的变量)
    find.BH_BH = np.empty(shape=(0, 25))
    find.BH_NS = find.BH_BH.copy()
    find.BH_WD = find.BH_BH.copy()
    find.NS_NS = find.BH_BH.copy()
    find.NS_WD = find.BH_BH.copy()
    find.WD_WD = find.BH_BH.copy()
    find.Merger = np.empty(shape=(0, 20))

    # 初始化output实例
    output.bcm = np.zeros((50001, 35))
    output.bpp = np.zeros((81, 11))

    # 初始化zcnsts实例
    zcnsts.z = 0.0
    zcnsts.zpars = np.zeros((1, 20)).flatten()
    zcnsts.msp = np.zeros((1, 200)).flatten()
    zcnsts.gbp = np.zeros((1, 200)).flatten()

    # 初始化kick实例
    kick.f_fb = 0.0
    kick.meanvk = 0.0
    kick.sigmavk = 0.0


def main():
    # 首先清除之前的文本记录并写入标题
    CS = ['BH_BH', 'BH_NS', 'BH_WD', 'NS_NS', 'NS_WD', 'WD_WD', 'Merger']
    for i in range(7):
        file = open(path + CS[i] + ".txt", 'w')
        if i < 6:
            file.write(
                'm1 m2 tb z tphysf mx mx2 tbx eccx mx11 mx22 tbx1 eccx1 kwx1 kwx2 kw kw2 fgw hc SNR t1 t2 dtt Wb num \n')
        else:
            file.write('m1 m2 tb z tphysf mx mx2 tbx eccx mx11 mx22 tbx1 eccx1 kwx1 kwx2 kwx11 kwx22 t1 t3 Wb \n')
        file.close()

    # 并行计算演化大数量的双星系统
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        results = [executor.submit(popbin) for _ in range(int(num_evolve))]


if __name__ == '__main__':
    start = time()

    main()

    end = time()
    print('运行时间: %s 秒' % (end - start))


