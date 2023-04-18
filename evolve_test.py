from SingleStar import SingleStar
from BinaryStar import BinaryStar



def evolve(star1, star2, binary):
    # 考虑星风的影响（质量/自旋角动量/轨道角动量的减少/增加）
    binary.steller_wind()
    # 考虑双星的磁制动影响（自旋角动量的减少）
    star1.magnetic_braking()
    # 计算有明显对流包层的恒星因磁制动损失的自旋角动量, 包括主序星(M < 1.25)、靠近巨星分支的HG恒星以及巨星, 不包括完全对流主序星
