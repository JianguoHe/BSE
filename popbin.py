import os
import time
import random
import pysnooper
import numpy as np
import pandas as pd
import concurrent.futures
from evolve import evolve
from zcnst import zcnsts_set
from utils import Wb_binary, SFR_Galaxy, select
from const import ecc_scheme
from const import rng_m1, rng_m2, rng_sep, rng_ecc, rng_z
from const import yeardy, Zcnsts, Kick, Output, G, Msun, Rsun, num_evolve
from const import mb_model, gamma_mb, mass_accretion_model, max_WD_mass


MA = 'MA1_' if mass_accretion_model == 1 else ('MA2_' if mass_accretion_model == 2 else 'MA3_')
MB = 'MB1_' if mb_model == 'Hurley2002' else ('MB2_' if gamma_mb == 4 else 'MB3_')
critical_mass = '02' if max_WD_mass == 0.2 else ('04' if max_WD_mass == 0.4 else '125')
target_file = './data/' + MA + MB + critical_mass + '.csv'


def Find(zcnsts, kick, output, index):
    # 设置初始参数(双星质量、轨道间距)
    a = rng_m1[index]
    b = rng_m2[index]
    c = rng_sep[index]

    m1 = np.exp(a)          # 范围是(5, 50)
    m2 = np.exp(b)          # 范围是(0.5, 50)
    sep = np.exp(c)         # 双星间距，范围是(3, 10000)

    # 初始轨道周期(单位: 天), 范围为 0.04 - 49393
    tb = 2 * np.pi * (sep * Rsun) ** (3 / 2) * (G * Msun * (m1 + m2)) ** (-1 / 2) / (3600 * 24)

    # 单个特殊值
    # m1 = 10
    # m2 = 4
    # tb = 5000

    # 排除 m1 小于 m2 的情况
    if m1 < m2:
        return

    # 计算双星的权重
    Wb = Wb_binary(m1, m2)

    # 计算 zcnsts 参数
    z = 0.02
    zcnsts.z = z
    zcnsts_set(zcnsts)

    # 最大演化时间
    tphysf = 10000
    tphys = 0.0

    # 初始偏心率
    if ecc_scheme == 'zero':
        ecc = 0.0
    elif ecc_scheme == 'uniform':
        ecc_value = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ecc = ecc_value[rng_ecc[index]]
    elif ecc_scheme == 'thermal':
        ecc = np.sqrt(random.random())
    else:
        raise ValueError("Please provide an allowed scheme of initial eccentricity.")

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
           ospin, epoch, tms, tphys, tphysf, dtp, z, zcnsts, tb, ecc, kick, output, index)

    result = output.bcm[:np.where(output.bcm[:, 1] == -1)[0][0], :]
    result[:, 0] = index
    result[:, 30] = result[:, 30] * yeardy

    # 定义列名
    column_names = ['index', 'time', 'kw1', 'mass1_initial', 'mass1', 'lg(L1)', 'lg(r1)', 'lg(T1)', 'mc1', 'rc1',
                    'menv1', 'renv1', 'epoch1', 'spin1', 'mdot1', 'r1/rl1', 'kw2', 'mass2_initial', 'mass2', 'lg(L2)',
                    'lg(r2)', 'lg(T2)', 'mc2', 'rc2', 'menv2', 'renv2', 'epoch2', 'spin2', 'mdot2', 'r2/rl2',
                    'tb', 'sep', 'ecc', 'CE_channel', 'None']

    # 将二维数组转换为pandas数据
    df = pd.DataFrame(result, columns=column_names)

    # 挑选目标源
    condition_NS_WD = select(df)[4]
    if np.any(condition_NS_WD):
        # 计算数量
        rate = SFR_Galaxy() * Wb                            # 原初双星的诞生率(每百万年)
        df['num'] = df['time'].diff().fillna(df['time'][0]) * rate      # 原初双星在每个步长内的诞生数量

        # 符合目标源条件的行标
        df['NS_WD'] = np.where(condition_NS_WD, True, False)

        # 记录每次双星类型发生变化的行标
        df['kw_change'] = (df['kw1'] != df['kw1'].shift().fillna(df['kw1'].iloc[-1])
                           ) | (df['kw2'] != df['kw2'].shift().fillna(df['kw2'].iloc[-1]))

        # 创建SNR和CE列
        df['SNR'] = 0
        df['CE'] = 0

        # 获取每次类型变化的行的索引位置
        rows = df.index[df['kw_change']]

        # 检查在每次类型变换期间, 是否存在CE演化
        for j in range(len(rows) - 1):
            if 1 in df.loc[rows[j]:rows[j + 1] - 1, 'CE_channel'].values:
                df.loc[rows[j + 1], 'CE'] = 1

        # 筛选类型变化以及双致密星的记录
        df = df[df['NS_WD'] | df['kw_change']]

        # 选择需要输出的列
        df = df[['index', 'time', 'mass1', 'mass2', 'tb', 'ecc', 'kw1', 'kw2', 'CE', 'SNR',
                 'r1/rl1', 'r2/rl2', 'num']]

        # 存储到文本中
        float_list_NS_WD = ['time', 'mass1', 'mass2', 'r1/rl1', 'r2/rl2', 'SNR']
        int_list_NS_WD = ['index', 'kw1', 'kw2', 'CE']
        sci_list_NS_WD = ['tb', 'ecc', 'num']

        df[float_list_NS_WD] = df[float_list_NS_WD].round(3)
        df[int_list_NS_WD] = df[int_list_NS_WD].astype(int)
        df[sci_list_NS_WD] = df[sci_list_NS_WD].applymap('{:.3e}'.format)

        if os.path.getsize(target_file) == 0:
            df.to_csv(target_file, header=True, index=False)
        else:
            df.to_csv(target_file, header=False, index=False, mode='a')


# @pysnooper.snoop()
def popbin(index):
    # 用于打印执行次数
    if index == 100:
        print("开始啦！")
    if index % (num_evolve // 100) == 0.0 and index > 0:
        print(index)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # 初始化Zcnsts的实例
    z = 0.02
    zpars = np.zeros((1, 20)).flatten()
    msp = np.zeros((1, 200)).flatten()
    gbp = np.zeros((1, 200)).flatten()
    zcnsts = Zcnsts(z, zpars, msp, gbp)

    # 初始化Kick类的实例
    f_fb = 0.0
    meanvk = 0.0
    sigmavk = 0.0
    kick = Kick(f_fb, meanvk, sigmavk)

    # 初始化Output类的实例
    bcm = np.zeros((50001, 35))
    bpp = np.zeros((81, 11))
    output = Output(bcm, bpp)

    # 开始寻找目标源
    Find(zcnsts, kick, output, index)


def main():
    # 首先清除之前的文本记录
    open(target_file, 'w').close()

    # 并行计算演化大数量的双星系统
    # with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    #     results = [executor.submit(popbin, index) for index in range(num_evolve)]

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            results = [executor.submit(popbin, index) for index in range(num_evolve)]
            for future in concurrent.futures.as_completed(results):
                try:
                    result = future.result()
                except Exception as e:
                    print("Exception occurred:", e)
    except Exception as e:
        print("Exception occurred:", e)


if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    print('运行时间: %s 秒' % (end - start))


