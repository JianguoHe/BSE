from SingleStar import SingleStar
from BinaryStar import BinaryStar


star1 = SingleStar(type=1, mass=2, Z=0.02)
print(star1.Z)
print(star1.zpars)
print(star1.msp)
print(star1.gbp)



# star2 = SingleStar(type=1, mass=1, Z=0.02)
# binary = BinaryStar(star1=star1, star2=star2, period=66.8, eccentricity=0)
# binary.evolve()


# print(binary.sep, binary.period, binary.jorb, binary.jdot_gr)
# print(star1.rochelobe, star2.rochelobe)
# print(binary.state, binary.event)
# print(star1.tm)
# print(star2.tm)





