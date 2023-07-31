import numpy as np
from const import ktype, mch, mxns
from StellarCal import StellarCal
from gntage import gntage
from utils import conditional_njit


# 模拟恒星碰撞
@conditional_njit()
def mix(star1, star2):   #mass0, mass, aj, kstar, zcnsts):
    # # 设置初始值
    # tm1 = 0   # 不同恒星的主序时间
    # tm2 = 0
    # tm3 = 0
    # # 以下几个变量只用于本程序内的过程计算，不影响外部程序
    # tn = 0
    # tscls = np.zeros((1, 21)).flatten()
    # lums = np.zeros((1, 11)).flatten()
    # # GB = lums.copy()
    #
    # # Define global indices with body j1 being most evolved.(演化历程较深的, 即 kw 较大)
    # if kstar[1] >= kstar[2]:
    #     j1 = 1
    #     j2 = 2
    # else:
    #     j1 = 2
    #     j2 = 1
    #
    # # Specify case index for collision treatment.
    # k1 = kstar[j1]
    # k2 = kstar[j2]
    # icase = ktype[int(k1), int(k2)]
    #
    # # Determine evolution time scales for first star.
    # mass01 = mass0[j1]
    # mass1 = mass[j1]
    # age1 = aj[j1]
    # (tm1, tn, tscls, lums, GB) = star(k1, mass01, mass1, zcnsts)
    #
    # # Obtain time scales for second star.
    # mass02 = mass0[j2]
    # mass2 = mass[j2]
    # age2 = aj[j2]
    # (tm2, tn, tscls, lums, GB) = star(k2, mass02, mass2, zcnsts)
    #
    # star(self1)
    # star(self2)
    #
    # # Check for planetary systems - which is defined as HeWDs and low-mass WDs!
    # if k1 == 10 and mass1 < 0.05:
    #     icase = k2
    #     if k2 <= 1:
    #         icase = 1
    #         age1 = 0.0
    # elif k1 >= 11 and mass1 < 0.5 and icase == 6:
    #     icase = 9
    #
    # if k2 == 10 and mass2 < 0.05:
    #     icase = k1
    #     if k1 <= 1:
    #         icase = 1
    #         age2 = 0.0
    #
    # # Specify total mass.
    # mass3 = mass1 + mass2
    # mass03 = mass01 + mass02
    # kw = icase
    # age3 = 0.0
    #
    # # Restrict merged stars to mass less than 100 Msun.
    # if mass3 >= 100.0:
    #     mass3 = 99.0
    #     mass03 = min(mass03, mass3)
    #
    # # Evaluate apparent age and other parameters.
    # if icase == 1:
    #     # Specify new age based on complete mixing.
    #     if k1 == 7:
    #         kw = 7
    #     (tm3, tn, tscls, lums, GB) = star(kw, mass03, mass3, zcnsts)
    #     age3 = 0.1 * tm3 * (age1 * mass1 / tm1 + age2 * mass2 / tm2) / mass3
    # elif icase == 3 or icase == 6 or icase == 9:
    #     mc3 = mass1
    #     (mc3, mass3, kw, mass03, age3) = gntage(mc3, mass3, kw, zcnsts, mass03, age3)
    # elif icase == 4:
    #     mc3 = mass1
    #     age3 = age1 / tm1
    #     (mc3, mass3, kw, mass03, age3) = gntage(mc3, mass3, kw, zcnsts, mass03, age3)
    # elif icase == 7:
    #     (tm3, tn, tscls, lums, GB) = star(kw, mass03, mass3, zcnsts)
    #     age3 = tm3 * (age2 * mass2 / tm2) / mass3
    # elif icase <= 12:
    #     # Ensure that a new WD has the initial mass set correctly.
    #     mass03 = mass3
    #     if icase < 12 and mass3 >= mch:
    #         mass3 = 0.0
    #         kw = 15
    # elif icase == 13 or icase == 14:
    #     # Set unstable Thorne-Zytkow object with fast mass loss of envelope
    #     # unless the less evolved star is a WD, NS or BH.
    #     # Thorne-Zytkow object 是一种假设存在的恒星，是指核心有中子星存在的红巨星或红超巨星.
    #     if k2 < 10:
    #         mass03 = mass1
    #         mass3 = mass1
    #     if icase == 13 and mass3 > mxns:
    #         kw = 14
    # elif icase == 15:
    #     mass3 = 0.0
    # elif icase > 100:
    #     # Common envelope case which should only be used after COMENV.
    #     kw = k1
    #     age3 = age1
    #     mass3 = mass1
    #     mass03 = mass01
    # else:
    #     # This should not be reached.
    #     kw = 1
    #     mass03 = mass3
    #
    # # Put the result in *1.
    # kstar[1] = kw
    # mass0[1] = mass03
    # mass[1] = mass3
    # aj[1] = age3
    # kstar[2] = 15
    # mass[2] = 0.0
    #
    # return mass0, mass, aj, kstar





    icase = ktype[int(star1.type), int(star2.type)]
    StellarCal(star1)
    StellarCal(star2)
    if star1.type == 10 and star1.mass < 0.05:
        icase = star2.type
        if star2.type <= 1:
            icase = 1
            age1 = 0.0
    elif star1.type >= 11 and star1.mass < 0.5 and icase == 6:
        icase = 9

    if star1.type == 10 and star2.mass < 0.05:
        icase = star1.type
        if star1.type <= 1:
            icase = 1
            age2 = 0.0

    # Specify total mass.
    mass3 = star1.mass + star2.mass
    mass03 = star1.mass0 + star2.mass0
    kw = icase
    age3 = 0.0

    # Restrict merged stars to mass less than 100 Msun.
    if mass3 >= 100.0:
        mass3 = 99.0
        mass03 = min(mass03, mass3)

    type_temp, mass0_temp, mass_temp, masscore_temp, age_temp = star1.type, star1.mass0, star1.mass, star1.mass_core, star1.age
    # Evaluate apparent age and other parameters.
    if icase == 1:
        # Specify new age based on complete mixing.
        if star1.type == 7:
            kw = 7
        star1.type, star1.mass0, star1.mass = kw, mass03, mass3
        StellarCal(star1)
        tm3 = star1.tm
        tn = star1.tn
        tscls = star1.tscls
        # (tm3, tn, tscls, lums, GB) = star(kw, mass03, mass3, zcnsts)
        age3 = 0.1 * star1.tm * (star1.age * star1.mass / star1.tm + star2.age * star2.mass / star2.tm) / mass3
    elif icase == 3 or icase == 6 or icase == 9:
        mc3 = star1.mass
        star1.mass_core, star1.mass, star1.type, star1.mass0, star1.age = mc3, mass3, kw, mass03, age3
        gntage(star1)
        mc3, mass3, kw, mass03, age3 = star1.mass_core, star1.mass, star1.type, star1.mass0, star1.age
        # (mc3, mass3, kw, mass03, age3) = gntage(mc3, mass3, kw, zcnsts, mass03, age3)
    elif icase == 4:
        mc3 = star1.mass
        age3 = star1.age / star1.tm
        star1.mass_core, star1.mass, star1.type, star1.mass0, star1.age = mc3, mass3, kw, mass03, age3
        gntage(star1)
        mc3, mass3, kw, mass03, age3 = star1.mass_core, star1.mass, star1.type, star1.mass0, star1.age
        # (mc3, mass3, kw, mass03, age3) = gntage(mc3, mass3, kw, zcnsts, mass03, age3)
    elif icase == 7:
        # (tm3, tn, tscls, lums, GB) = star(kw, mass03, mass3, zcnsts)
        StellarCal(star1)
        tm3 = star1.tm
        age3 = tm3 * (star2.age * star2.mass / star2.tm) / mass3
    elif icase <= 12:
        # Ensure that a new WD has the initial mass set correctly.
        mass03 = mass3
        if icase < 12 and mass3 >= mch:
            mass3 = 0.0
            kw = 15
    elif icase == 13 or icase == 14:
        # Set unstable Thorne-Zytkow object with fast mass loss of envelope
        # unless the less evolved star is a WD, NS or BH.
        # Thorne-Zytkow object 是一种假设存在的恒星，是指核心有中子星存在的红巨星或红超巨星.
        if star2.type < 10:
            mass03 = star1.mass
            mass3 = star1.mass
        if icase == 13 and mass3 > mxns:
            kw = 14
    elif icase == 15:
        mass3 = 0.0
    elif icase > 100:
        # Common envelope case which should only be used after COMENV.
        kw = star1.type
        age3 = age1
        mass3 = star1.mass
        mass03 = star1.mass0
    else:
        # This should not be reached.
        kw = 1
        mass03 = mass3
    star1.type, star1.mass0, star1.mass, star1.mass_core, star1.age = type_temp, mass0_temp, mass_temp, masscore_temp, age_temp
    StellarCal(star1)
    star1.type, star1.mass0, star1.mass, star1.age = kw, mass03, mass3, age3
    star2.type, star2.mass = 15, 0.0

    return 0