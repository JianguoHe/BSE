import numpy as np

from SingleStar import SingleStar
from BinaryStar import BinaryStar
from StellarCal import StellarCal
from StellarProp import StellarProp

star1 = SingleStar(type=1, mass=1, Z=0.02)
star2 = SingleStar(type=1, mass=1, Z=0.02)
binary = BinaryStar(star1=star1, star2=star2, period=1000, eccentricity=0)
binary.evolve()
print(binary.sep, binary.period)





# def pr(star1, type, mass, age):
#     print("---------", type, mass, age)
#     # print(star1.type, star1.mass, star1.L, star1.R, star1.mass_core, star1.mass_envelop, star1.radius_core, star1.radius_envelop, star1.k2)
#     # print(type, mass, age, "lums:              ", str(star1.lums))
#     # print(type, mass, age, "tscls:           ", str(star1.tscls))
#     print(type, mass, age, "type:            ", str(star1.type))
#     print(type, mass, age, "mass:            ", str(star1.mass))
#     print(type, mass, age, "L:               ", str(star1.L))
#     print(type, mass, age, "R:               ", str(star1.R))
#     print(type, mass, age, "mass_core:       ", str(star1.mass_core))
#     print(type, mass, age, "mass_envelop:    ", str(star1.mass_envelop))
#     print(type, mass, age, "radius_core:     ", str(star1.radius_core))
#     print(type, mass, age, "radius_envelop:  ", str(star1.radius_envelop))
#     print(type, mass, age, "k2:              ", str(star1.k2))

# for i in[[1,1],[1,3],[1,10],[7,1],[7,2],[7,3]]:
#     star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
#     star(star1)
#     star1.age = star1.tm * 0.9
#     hrdiag(star1)
#     pr(star1,i[0],i[1],0.9)
#
# for i in[[1,1],[1,3],[1,10]]:
#     star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
#     star(star1)
#     star1.age = star1.tm + 0.5 * (star1.tscls[1] - star1.tm)
#     hrdiag(star1)
#     pr(star1,i[0],i[1],2)

# for i in[[1,1],[1,3],[1,10]]:
#     star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
#     star(star1)
#     star1.age = star1.tscls[1]+0.5*(star1.tscls[2]-star1.tscls[1])
#     hrdiag(star1)
#     pr(star1,i[0],i[1],3)
#     star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
#     star(star1)
#     star1.age = star1.tscls[2]+0.5*star1.tscls[3]
#     hrdiag(star1)
#     pr(star1,i[0],i[1],4)


# for i in[[7,1],[7,2],[7,3]]:
#     star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
#     star(star1)
#     star1.age = star1.tm * 1.1
#     hrdiag(star1)
#     pr(star1,i[0],i[1],1.1)
    # del star1
    # star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
    # star(star1)
    # star1.age = star1.tm * 3
    # hrdiag(star1)
    # pr(star1,i[0],i[1],3)
    # del star1

# for i in [[11, 0.5], [13, 1.4]]:
#     star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
#     star(star1)
#     star1.age = 0.1
#     hrdiag(star1)
#     pr(star1, i[0], i[1], 0.1)



# print(star1.)
# print(star1.)
# print(star1.)
# print(star1.)
# from BinaryStar import BinaryStar
# a = [[1,0.8],[1,1.5],[1,5],[7,0.6],[7,2],[11,0.8]]
# # b = np.linspace(0.2,100,20)
# # print(b)
# # exit(0)
# for i in a:
#     star1 = SingleStar(type=i[0], mass=i[1], Z=0.02)
#     star(star1)
#     # print("---")
#     print(i[0])
#     print(i[1])
#     print(star1.tm)
#     print(star1.tn)
#     print(star1.tscls)
#     print(star1.lums)
#     print(star1.GB)

