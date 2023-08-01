from numba import float64, int64, types
from numba.experimental import jitclass

# 定义jitclass类的规范
spec = [
    ('property1', float64),  # 第一个属性的名称和类型
    ('property2', float64),  # 第二个属性的名称和类型
]

# 创建jitclass类
@jitclass(spec)
class MyClass:
    def __init__(self, value1, value2):
        self.property1 = value1
        self.property2 = value2

# 创建一个实例
my_instance = MyClass(1.5, 2.3)

# 访问属性
print("property1:", my_instance.property1)  # 输出: property1: 1.5
print("property2:", my_instance.property2)  # 输出: property2: 2.3

my_instance.property1 = my_instance.property2 = 3
my_instance.property2 = 4

# 访问属性
print("property1:", my_instance.property1)  # 输出: property1: 1.5
print("property2:", my_instance.property2)  # 输出: property2: 2.3
my_instance.property1 = 5

# 访问属性
print("property1:", my_instance.property1)  # 输出: property1: 1.5
print("property2:", my_instance.property2)  # 输出: property2: 2.3