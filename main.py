from SingleStar import SingleStar
from BinaryStar import BinaryStar


# 类实例化
star1 = SingleStar(type=1, Z=0.02, mass=1.4)
star2 = SingleStar(type=1, Z=0.02, mass=0.45)
binary = BinaryStar(star1, star2, separation=20, eccentricity=0)

# 测试函数
M_chrip = binary.chirp_mass()

print(M_chrip)

