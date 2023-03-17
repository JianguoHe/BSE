from SingleStar import SingleStar
from BinaryStar import BinaryStar


# 类实例化
star1 = SingleStar(type=1, Z=0.02, mass=2)
star2 = SingleStar(type=1, Z=0.02, mass=1)
binary = BinaryStar(star1, star2, separation=20, eccentricity=0)

# 测试函数
binary.steller_wind()
print(star1.R)
print(star2.R)
