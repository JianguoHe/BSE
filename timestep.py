from const import pts1, pts2, pts3
from utils import conditional_njit


# 确定恒星演化的更新步长
@conditional_njit()
def timestep(kw, age, tm, tn, tscls, dt):
    # 输入: kw, age, tm, tn, tscls, dt
    # 输出: dt, dtr(只有当确定致密星的下一步演化时长需要用到旧的步长)

    # pts1 = 0.05      MS
    # pts2 = 0.01      CHeB, GB, AGB, HeGB
    # pts3 = 0.02      HG, HeMS
    if kw <= 1:
        dt = pts1 * tm
        dtr = tm - age
    elif kw == 2:
        dt = pts3 * (tscls[1] - tm)      # 【更改】把这里的 pts1 改成 pts3, 缩短 HG 的演化步长
        dtr = tscls[1] - age
    elif kw == 3:
        if age < tscls[6]:
            dt = pts2 * (tscls[4] - age)
        else:
            dt = pts2 * (tscls[5] - age)
        dtr = min(tscls[2], tn) - age
    elif kw == 4:
        dt = pts2 * tscls[3]
        dtr = min(tn, tscls[2] + tscls[3]) - age
    elif kw == 5:
        if age < tscls[9]:
            dt = pts3 * (tscls[7] - age)
        else:
            dt = pts3 * (tscls[8] - age)
        dtr = min(tn, tscls[13]) - age
    elif kw == 6:
        if age < tscls[12]:
            dt = pts3 * (tscls[10] - age)
        else:
            dt = pts3 * (tscls[11] - age)
        dt = min(dt, 0.005)
        dtr = tn - age
    elif kw == 7:
        dt = pts1 * tm
        dtr = tm - age
    elif kw == 8 or kw == 9:
        if age < tscls[6]:
            dt = pts2 * (tscls[4] - age)
        else:
            dt = pts2 * (tscls[5] - age)
        dtr = tn - age
    else:
        dt = max(0.1, dt * 10.0)
        dt = min(dt, 5.0e2)
        dtr = dt

    return dt, dtr
