from SingleStar import SingleStar
from BinaryStar import BinaryStar



def evolve(star1, star2, binary):
    # 考虑星风的影响（质量/自旋角动量/轨道角动量/偏心率的减少/增加）
    binary.steller_wind()
    # 考虑引力波辐射
    binary.GW_radiation()
    # 考虑双星的磁制动影响
    star1.magnetic_braking()
    star2.magnetic_braking()
    # 考虑限制一下
