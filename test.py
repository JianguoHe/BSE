import numpy as np

from SingleStar import SingleStar
from BinaryStar import BinaryStar
from star import star
a = [[1,0.8],[1,1.5],[1,5],[7,0.6],[7,2],[11,0.8]]
# b = np.linspace(0.2,100,20)
# print(b)
# exit(0)
for i in a:
    star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
    star(star1)
    # print("---")
    print(i[0])
    print(i[1])
    print(star1.tm)
    print(star1.tn)
    print(star1.tscls)
    print(star1.lums)
    print(star1.GB)

# star2 = SingleStar(type=1, mass=1, Z=0.02)
# binary = BinaryStar(star1=star1, star2=star2, period=66.8, eccentricity=0)
# binary.evolve()


# print(binary.sep, binary.period, binary.jorb, binary.jdot_gr)
# print(star1.rochelobe, star2.rochelobe)
# print(binary.state, binary.event)
# print(star1.tm)
# print(star2.tm)
