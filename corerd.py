import numpy as np
from const import mch
from numba import njit


@njit
def corerd(kw, mc, m0, mflash):
    if kw == 14:
        corerd = 4.24e-06*mc
    elif kw == 13:
        corerd = 1.4e-05
    elif kw <= 1 or kw == 7:
        corerd = 0.0
    elif kw == 4 or kw == 5 or (kw <= 3 and m0 > mflash):
        corerd = 0.22390*mc**0.620
    else:
        corerd = 0.01150*np.sqrt(max(1.48204e-06,(mch/mc)**(2.0/3.0)- (mc/mch)**(2.0/3.0)))
        if kw <= 9:
            corerd = 5.0*corerd
    return corerd
