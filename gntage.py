import numpy as np
from StellarCal import StellarCal
# from zfuncs import mcheif,mcagbf,mheif,mbagbf,lum_to_mc_gb,mc_to_lum_gb,lbgbf,lbgbdf
from utils import conditional_njit
from const import tiny


@conditional_njit()
def gntage(self):   #mc, self.mass, self.type, self.zcnsts, self.mass0, self.age):   #mass_core, mass, self.type, self.zcnsts, mass0, age):
    # 设置常数
    macc = 0.00001
    lacc = 0.0001
    flag1 = True    # 仅供程序中跳转功能用（用来实现 Fortran 中的 goto 功能）
    flag2 = True    # 仅供程序中跳转功能用
    flag3 = True    # 仅供程序中跳转功能用

    jmax = 30

    if self.type == 4:
        mcy = self.mcheif(self.zpars[2], self.zpars[2], self.zpars[10], self.zcnsts)
        if self.mass_core <= mcy:
            self.type = 3

    if self.type == 3:
        mcy = self.mcheif(self.zpars[3], self.zpars[2], self.zpars[9], self.zcnsts)
        if self.mass_core >= mcy:
            self.type = 4
            self.age = 0

    if self.type == 6:
        mcy = 0.440 * 2.250 + 0.4480
        if self.mass_core > mcy:
            mcx = (self.mass_core + 0.350) / 0.7730
        elif self.mass_core >= 0.8:
            mcx = (self.mass_core - 0.4480) / 0.440
        else:
            mcx = self.mass_core
        self.mass0 = self.mbagbf(mcx)
        if self.mass0 < tiny:
            self.type = 14
        else:
            # (tm, tn, tscls, lums, GB) = star(self.type, self.mass0, self.mass, self.zcnsts)
            StellarCal(self)
            self.age = self.tscls[13]

    if self.type == 5:
        self.mass0 = self.mbagbf(self.mass_core)
        if self.mass0 < tiny:
            self.type = 14
        else:
            # (tm, tn, tscls, lums, GB) = star(self.type, self.mass0, self.mass, self.zcnsts)
            StellarCal(self)
            self.age = self.tscls[2] + self.tscls[3]

    while self.type == 4:
        if self.age < 0.0 or self.age > 1.0:
            self.age = 0.0
        mcy = self.mcagbf(self.zpars[2])
        if self.mass_core >= mcy:
            mmin = self.mbagbf(self.mass_core)
        else:
            mmin = self.zpars[2]
        mmax = self.mheif(self.mass_core, self.zpars[2], self.zpars[10], self.zcnsts)
        if self.age < tiny:
            self.mass0 = mmax
            flag1 = False
        elif self.age >= 1.0:
            self.mass0 = mmin
            flag1 = False
        if flag1:
            fmid = (1.0 - self.age) * self.mcheif(mmax, self.zpars[2], self.zpars[10]) + self.age * self.mcagbf(mmax) - self.mass_core
            f = (1.0 - self.age) * self.mcheif(mmin, self.zpars[2], self.zpars[10]) + self.age * self.mcagbf(mmin) - self.mass_core
            if f * fmid >= 0.0:
                self.type = 3
                break
            self.mass0 = mmin
            dm = mmax - mmin
            for j in range(1, jmax+1):
                dm = 0.50 * dm
                mmid = self.mass0 + dm
                fmid = (1.0 - self.age) * self.mcheif(mmid, self.zpars[2], self.zpars[10]) + self.age * self.mcagbf(mmid) - self.mass_core
                if fmid < 0.0:
                    self.mass0 = mmid
                if abs(dm) < macc or abs(fmid) < tiny:
                    break
                if j == jmax:
                    self.mass0 = self.mass
                    self.age = 0.0
        # (tm, tn, tscls, lums, GB) = star(self.type, self.mass0, self.mass, self.zcnsts)
        StellarCal(self)
        self.age = self.tscls[2] + self.age * self.tscls[3]
        break

    if self.type == 3:
        mcy = self.mcheif(self.zpars[3],self.zpars[2],self.zpars[9])
        if self.mass_core >= mcy:
            self.mass_core = 0.99 * mcy
        mcx = self.mcheif(self.zpars[2],self.zpars[2],self.zpars[9])
        if self.mass_core > mcx:
            self.mass0 = self.mheif(self.mass_core,self.zpars[2],self.zpars[9])
        else:
            self.mass0 = self.zpars[2]
            # (tm, tn, tscls, lums, GB) = star(self.type, self.mass0, self.mass, self.zcnsts)
            StellarCal(self)
            lum = self.mc_to_lum_gb(self.mass_core,self.GB)
            j = 0
            while True:
                dell = self.lbgbf() - lum
                if abs(dell / lum) <= lacc:
                    break
                derl = self.lbgbdf()
                self.mass0 = self.mass0 - dell / derl
                j = j + 1
                if j == jmax:
                    self.mass0 = self.zpars[2]
                    self.mass0 = max(self.mass0, self.mass)
                    break
        # (tm, tn, tscls, lums, self.GB) = star(self.type, self.mass0, self.mass, self.zcnsts)
        StellarCal(self)
        self.age = self.tscls[1] + 1.0e-6 * (self.tscls[2] - self.tscls[1])

    if self.type == 8 or self.type == 9:
        self.type = 8
        mmin = self.mass_core
        StellarCal(self)
        mcx = self.lum_to_mc_gb(self.lums[2])
        if mcx >= self.mass_core:
            self.mass0 = self.mass
            flag2 = False
        if flag2:
            f = mcx - self.mass_core
            mmax = self.mass
            for j in range(1, jmax+1):
                StellarCal(self)
                mcy = self.lum_to_mc_gb(self.lums[2])
                if mcy > self.mass_core:
                    break
                mmax = 2.0 * mmax
                if j == jmax:
                    self.mass0 = self.mass
                    flag3 = False
        while flag2 and flag3:
            fmid = mcy - self.mass_core
            if f * fmid >= 0.0:
                self.mass0 = self.mass
                break
            self.mass0 = mmin
            dm = mmax - mmin
            for j in range(1, jmax+1):
                dm = 0.5 * dm
                mmid = self.mass0 + dm
                StellarCal(self)
                mcy = self.lum_to_mc_gb(self.lums[2])
                fmid = mcy - self.mass_core
                if fmid < 0.0:
                    self.mass0 = mmid
                if abs(dm) < macc or abs(fmid) < tiny:
                    break
                if j == jmax:
                    self.mass0 = self.mass
                    break
            break
        # (tm, tn, tscls, lums, self.GB) = star(self.type, self.mass0, self.mass, self.zcnsts)
        StellarCal(self)
        self.age = self.tm + 1e-10 * self.tm

    if self.type == 14:
        self.type = 4
        self.mass0 = self.mass
        mcy = self.mcagbf(self.mass0)
        self.age = self.mass_core / mcy
        (tm, tn, tscls, lums, self.GB) = StellarCal(self.type, self.mass0, self.mass, self.zcnsts)
        if self.mass0 <= self.zpars[2]:
            mcx = self.lum_to_mc_gb(lums[4])
        else:
            mcx = self.mcheif(self.mass0, self.zpars[2], self.zpars[10])
        self.mass_core = mcx + (mcy - mcx) * self.age
        self.age = tscls[2] + self.age * tscls[3]

    # return self.mass_core, self.mass, self.type, self.mass0, self.age
    return 0