from const import pts1, pts2, pts3
from utils import conditional_njit


# 确定恒星演化的更新步长
@conditional_njit()
def timestep(self):   #self.type, self.age, self.tm, self.tn, self.tscls, dt):
    # 输入: self.type, self.age, self.tm, self.tn, self.tscls, dt
    # 输出: dt, dtr(只有当确定致密星的下一步演化时长需要用到旧的步长)

    # pts1 = 0.05      MS
    # pts2 = 0.01      CHeB, GB, AGB, HeGB
    # pts3 = 0.02      HG, HeMS
    if self.type <= 1:
        self.dt = pts1 * self.tm
        dtr = self.tm - self.age
    elif self.type == 2:
        self.dt = pts3 * (self.tscls[1] - self.tm)      # 【更改】把这里的 pts1 改成 pts3, 缩短 HG 的演化步长
        dtr = self.tscls[1] - self.age
    elif self.type == 3:
        if self.age < self.tscls[6]:
            self.dt = pts2 * (self.tscls[4] - self.age)
        else:
            self.dt = pts2 * (self.tscls[5] - self.age)
        dtr = min(self.tscls[2], self.tn) - self.age
    elif self.type == 4:
        self.dt = pts2 * self.tscls[3]
        dtr = min(self.tn, self.tscls[2] + self.tscls[3]) - self.age
    elif self.type == 5:
        if self.age < self.tscls[9]:
            self.dt = pts3 * (self.tscls[7] - self.age)
        else:
            self.dt = pts3 * (self.tscls[8] - self.age)
        dtr = min(self.tn, self.tscls[13]) - self.age
    elif self.type == 6:
        if self.age < self.tscls[12]:
            self.dt = pts3 * (self.tscls[10] - self.age)
        else:
            self.dt = pts3 * (self.tscls[11] - self.age)
        self.dt = min(self.dt, 0.005)
        dtr = self.tn - self.age
    elif self.type == 7:
        self.dt = pts1 * self.tm
        dtr = self.tm - self.age
    elif self.type == 8 or self.type == 9:
        if self.age < self.tscls[6]:
            self.dt = pts2 * (self.tscls[4] - self.age)
        else:
            self.dt = pts2 * (self.tscls[5] - self.age)
        dtr = self.tn - self.age
    else:
        self.dt = max(0.1, self.dt * 10.0)
        self.dt = min(self.dt, 5.0e2)
        dtr = self.dtdt

    return dtr
