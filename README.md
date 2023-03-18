# BSE
The binary star evolution code originated from the Hurley et.al 2000, we rewritte it in python and advance the operation speed through numba.njit decorator and parallel computing.


【popbin】
通过演化大量的双星系统，来研究这些双星的总体表征特性。
主要调用了evolve及其子程序，筛选出符合条件的结果并记录下来，通常演化100万个（样本空间100*100*100）。
为了加快Python运行速度，使用了两种加速方法：njit修饰器和并行计算。
njit修饰器需要在每一次运行的时候首先编译程序，耗时大概2-3分钟。编译完成即开始演化第1个双星，此后每个双星的演化都以近乎c语言的速度运行，
直到100万个演化完成。但在这种环境下，必须要将jitclass实例作为形参传递到各个函数中，目前尚未找到更好的替换办法。
并行计算用了多进程并行，因为是计算密集型任务，所以多进程相比多线程要快不少，目前是在陈老师的服务器上以十个核并行。
通过这两种方法，如果服务器不拥挤的话，演化100万个双星大概不到6分钟，减去编译时间，大概4分钟/百万个。
演化小样本的（20万个以内）可以尝试在自己电脑上跑，当然速度很慢，大样本的（100万个以上）在自己电脑上容易卡死，最好在服务器上跑。

************************************************************************

Definitions of the evolution types for the stars:
    STELLAR TYPES - KW
        0 - deeply or fully convective low mass MS star（<0.7）  /深对流层/完全对流的低质量主序星
        1 - Main Sequence star（>0.7） / 主序星
        2 - Hertzsprung Gap / 中等质量恒星离开主序演化到红巨星的时间很短，很难观测，称为赫罗图的空隙区
        3 - First Giant Branch   / 巨星分支，通常中低质量的恒星会在这个阶段点燃氦核
        4 - Core Helium Burning / 水平分支，氦平稳燃烧阶段
        5 - Early Asymptotic Giant Branch / 早期渐近巨星分支，双壳层产能阶段
        6 - Thermally Pulsing Asymptotic Giant Branch   / 热脉冲渐近巨星
        7 - Main Sequence Naked Helium star   /主序上的裸氦星
        8 - Hertzsprung Gap Naked Helium star  /赫氏空隙上的裸氦星
        9 - Giant Branch Naked Helium star  /巨星分支上的裸氦星
        10 - Helium White Dwarf  / 氦白矮星
        11 - Carbon/Oxygen White Dwarf  / 碳氧白矮星
        12 - Oxygen/Neon White Dwarf  / 氧氖白矮星
        13 - Neutron Star  / 中子星
        14 - Black Hole  / 黑洞
        15 - Massless Supernova  / 并合超新星

************************************************************************

evolve 文件中数据存储数组 bcm：
    bcm[,1:5] = Time, kw1, mass01, mass1, log10(L1)
    bcm[,6:10] = log10(r1), log10(Teff1), mc1, rc1, menv1
    bcm[,11:15] = renv1, epoch1, spin1, ml_rate1, r1/rl1
    bcm[,16:20] = kw2, mass02, mass2, log10(L2), log10(r2)
    bcm[,21:25] = log10(Teff2), mc2, rc2, menv2, renv2
    bcm[,26:30] = epoch2, spin2, ml_rate2, r2/rl2, tb
    bcm[,31:32] = sep, ecc

************************************************************************

Information on the BSE package.

Created by Jarrod Hurley at the Institute of Astronomy, Cambridge, UK 
in collaboration with Onno Pols and Christopher Tout. 
12th February 2000. 

IMPORTANT: This package must be used in conjunction with the SSE 
           package for evolving single stars. 

Information on the BSE package can be found in the paper: 

"Evolution of binary stars and the effect of tides on binary 
 populations" 
 Hurley J.R., Tout C.A., & Pols O.R., 2002, MNRAS, 329, 897. 

and more information on the SSE package can be found in the paper: 

"Comprehensive analytic formulae for stellar evolution as a function 
 of mass and metallicity" 
 Hurley J.R., Pols O.R., & Tout C.A., 2000, MNRAS, 315, 543. 

Any queries that are not answered by referring to these texts, or by 
reading the comments in the programs, can be addressed to: 
  jhurley@astro.swin.edu.au  

************************************************************************

Stellar mass range: 0.1 -> 100 Msun 
Metallicity range:  0.0001 -> 0.03 (0.02 is solar) 
Period range:       all 
Eccentricity Range: 0.0 -> 1.0

************************************************************************

The BSE package contains the following python files:

bse.py                - 主程序: 演化一个双星
popbin.py             - 主程序: 演化大样本的双星
const.py              - 存放一些常量以及类
evolve.py             - 控制双星演化
binary_in.txt         - bse 的输入文件
binary_out.txt        - bse 的输出文件
comenv.py             - 公共包层演化
corerd.py             - 估算类巨星的核半径
timestep.py           - 确定恒星演化的更新步长
dgcore.py             - 确定两个简并核并合的结果
gntage.py             - 计算并合中诞生的新恒星的参数
hrdiag.py             - 确定恒星目前处于哪一个演化阶段(kw, age), 然后计算光度、半径、质量、核质量
instar.py             - 设置碰撞矩阵
kick.py               - 产生超新星速度踢, 调整轨道参数
mix.py                - 模拟恒星碰撞
mlwind.py             - 包括质量损失公式
mrenv.py              - 估算对流包层的质量、半径, 以及包层的 gyration radius
star.py               - 推导不同演化阶段的时标、标志性光度、巨星分支参数
supernova.py          - 确定超新星爆炸后的致密星类型、质量
zcnsts.py             - 设置所有公式中依赖金属丰度的常数
zdata.py              - 包含 zcnsts 中的所有系数(实际上也是用来计算zcnsts的)
zfuncs.py             - 所有函数集合

************************************************************************

The main routine bse.f contains comments that should (hopefully?) make everything self-explanatory.
It is simply an example to show how EVOLV2 should be used.
Information is also contained in the headers of some routines.
In particular, see bse.f for an explanation of the input.

In the case of EVOLV2 being called in a loop over many stars 
be careful to initialize each new star, i.e. 

mass(i) = mass0(i)
kstar(i) = 1
epoch(i) = 0.0
ospin(i) = 0.0
tphys = 0.0 

as well as setting the masses (mass0), period (tb) and eccentricity (ecc). 

However, the routine ZCNSTS only needs to be called each time you change metallicity.

程序中有三个与年龄相关的变量，分别是age/tphys/epoch，下面是这几个的区别
Note that the epoch is required because the physical time and the age of the star are not always the same.
For example, when a star becomes a WD, at say time tWD, its effective age becomes zero because the WD evolution formulae
start from t=0.0. Therefore, we set epoch = tWD and from then on the age of the star is age = t - epoch.
The same happens when the star becomes a naked helium star, NS or BH. The epoch parameter is also very useful
in binary evolution when a star needs to be rejuvenated or aged as a result of mass transfer.


You may not want to use bse.f at all and instead, for example, prefer to use evolve directly as the main routine.
Also, you may want to utilize the individual subroutines in different ways. 

PROBLEMS: if the program stops with a 'FATAL ERROR' then more information should be contained in the fort.99 file.
          When you have identified the initial conditions that produce the error then please contact me, and
          I may help to fix the bug - assuming it can be reproduced!



#       ------------------------------------------------------------
#
#       [tscls] 1: BGB               2: He ignition         3: He burning      (BGB is the base of giant branch.)
#               4: Giant t(inf1)     5: Giant t(inf2)       6: Giant t(Mx)
#               7: EAGB t(inf1)      8: EAGB t(inf2)        9: EAGB  t(Mx)
#               10: TPAGB t(inf1)    11: TPAGB t(inf2)      12: TPAGB  t(Mx)
#               13: TP               14: t(Mcmax)                              (TP is thermally-pulsing AGB)
#
#       [lums]  1: ZAMS              2: End MS              3: BGB
#               4: He ignition       5: He burning          6: L(Mx)
#               7: BAGB              8: TP
#
#       [GB]    1: effective A(H)    2: A(H,He)             3: B
#               4: D                 5: p                   6: q
#               7: Mx                8: A(He)               9: Mc,BGB
#
#       ------------------------------------------------------------








