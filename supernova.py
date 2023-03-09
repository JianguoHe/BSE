from numba import njit
from const import mxns
import numpy as np
import random


# 对于各种超新星模型, 输出不同的致密星类型(NS/BH)和当前质量
# 输入变量: mt 为当前质量, mc 为CO核质量, mc_CO_he 为CO核+He包层质量
@njit
def supernova(mt, mc, mc_CO_he, SNtype, kick):
    if SNtype == 1:     # rapid SN, origin from Fryer et al. 2012, ApJ, 749, 91
        mproto = 1.0
        if mc < 2.5:
            mfb = 0.2
        elif 2.5 <= mc < 6:
            mfb = 0.286 * mc - 0.514
        elif 6 <= mc < 7:
            mfb = mt - mproto
        elif 7 <= mc < 11:
            a1 = 0.25 - 1.275 / (mt - mproto)
            b1 = -11.0 * a1 + 1.0
            mfb = (mt - mproto) * (a1 * mc + b1)
        else:
            mfb = mt - mproto
        kick.f_fb = mfb / (mt - mproto)
        mrem_bar = mfb + mproto                                         # 遗迹重子质量
        mrem1 = -6.6667 + 0.6667 * (100 + 30 * mrem_bar) ** 0.5         # 中子星引力质量
        mrem2 = 0.9 * mrem_bar                                          # 黑洞引力质量
        # 中子星
        if mrem1 <= mxns:
            kw = 13
            mt = mrem1
        # 黑洞
        else:
            kw = 14
            mt = mrem2
    elif SNtype == 2:     # delayed SN, origin from Fryer et al. 2012, ApJ, 749, 91
        if mc <= 3.5:
            mproto = 1.2
        elif 3.5 < mc <= 6.0:
            mproto = 1.3
        elif 6.0 < mc <= 11.0:
            mproto = 1.4
        else:
            mproto = 1.6
        if mc <= 2.5:
            mfb = 0.2
        elif 2.5 < mc <= 3.5:
            mfb = 0.5 * mc - 1.05
        elif 3.5 < mc <= 11.0:
            a2 = 0.133 - 0.093 / (mt - mproto)
            b2 = -11.0 * a2 + 1.0
            mfb = (mt - mproto) * (a2 * mc + b2)
        else:
            mfb = mt - mproto
        kick.f_fb = mfb / (mt - mproto)
        mrem_bar = mfb + mproto                                         # 遗迹重子质量
        mrem1 = -6.6667 + 0.6667 * (100 + 30 * mrem_bar) ** 0.5         # 中子星引力质量
        mrem2 = 0.9 * mrem_bar                                          # 黑洞引力质量
        # 中子星
        if mrem1 <= mxns:
            kw = 13
            mt = mrem1
        # 黑洞
        else:
            kw = 14
            mt = mrem2
    else:               # stochastic SN, origin from Mandel et al. 2020, MNRAS 499, 3214–3221
        m11 = 2.0
        m22 = 3.0
        m33 = 7.0
        m44 = 8.0
        meanbh = 0.8
        sigmabh = 0.5
        p1 = random.random()
        p2 = random.random()
        # 计算黑洞形成时物质完全回落(complete fallback)的概率
        if m11 <= mc < m44:
            pcf = (mc - m11) / (m44 - m11)
        else:
            pcf = 1.0
        # 中子星
        if mc < m11:
            mean0 = 1.2
            sigma0 = 0.02
            kw = 13
            mt = max(1.13, random.gauss(mean0, sigma0))
        # 中子星或黑洞
        elif m11 <= mc < m33:
            # 计算遗迹是黑洞的概率
            pbh = (mc - m11) / (m33 - m11)
            # 黑洞
            if p1 <= pbh:
                kw = 14
                # 完全回落
                if p2 <= pcf:
                    mt = mc_CO_he
                # 不完全回落
                else:
                    mt = max(2.0, random.gauss(meanbh * mc, sigmabh))
            # 中子星
            else:
                kw = 13
                if m11 <= mc < m22:
                    mean0 = 1.4 + 0.5 * (mc - m11) / (m22 - m11)
                    sigma0 = 0.05
                else:
                    mean0 = 1.4 + 0.4 * (mc - m22) / (m33 - m22)
                    sigma0 = 0.05
                mt = max(1.13, random.gauss(mean0, sigma0))
        # 黑洞
        else:
            kw = 14
            # 完全回落
            if p2 <= pcf:
                mt = mc_CO_he
            # 不完全回落
            else:
                mt = max(2.0, random.gauss(meanbh * mc, sigmabh))
        # 对于 stochastic SN, 速度踢服从一定的正态分布(高斯分布)
        if kw == 13:
            kick.meanvk = 520.0 * (mc - mt) / mt
            kick.sigmavk = 0.3 * kick.meanvk
        elif kw == 14:
            kick.meanvk = 200.0 * max((mc - mt) / mt, 0.0)
            kick.sigmavk = 0.3 * kick.meanvk
    return kw, mt



