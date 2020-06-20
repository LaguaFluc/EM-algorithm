"""
目标-步骤分解：
1. 给出初始条件 完成。
2. 读出数据，完成。
3. 设计函数，E步，完成。
4. 设计函数，M步，完成。
5. 设计函数，终止条件，完成。-2020/6/10/10:00,功能实现。
TODO 使用类，在类中定义方法来写代码
- 主要随机生成
- 测定对初值的敏感性

- 已生成于CSV文件中，2020/6/5/15：43
"""
import numpy as np
import csv
import timeit

# ----------------------1、给定初始条件 ----------------------
# 正态总体的均值、方差
mu = [165, 175]
sigma = [6, 5]

# 两类的比例，分别是男性：女性
# 后面的默认参数，定义为不变对象--元组
alpha_init = (0.5, 0.5) # 实际数据是6：4

# 类别的个数
J = len(alpha_init)

# ----------------------2、读出数据 ----------------------
path_male = r"D:\lagua\study\coding\PythonStudy\EM_data\male_1.csv"
path_female = r"D:\lagua\study\coding\PythonStudy\EM_data\female_1.csv"
# 获取数据。
def get_data(path_in):
    height_data = []
    with open(path_in,'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            # 从CSV文件中获取的，是字符串数据，需要自己转化为浮点型数据。
            height_data.append(float(line[1])) # 每一行是一个列表，从列表中取出数据。
    print('Height Data collected.')
    return height_data

# TODO 可以设置一个函数，直接获取数据，而不用分两步。
male_height = get_data(path_male)
female_height = get_data(path_female)
Mixed_height = male_height + female_height # 列表相加/合并

# 样本数据个数
N = len(Mixed_height)


# 返回一个数 正态分布函数的密度
def normal_pdf(x, mu_num, sigma_num):
    temp = np.exp(-((x-mu_num)**2)/(2*sigma_num**2)) / (sigma_num * np.sqrt(2*np.pi))
    return temp


def EucliDis(x, y):
    '''
    作用：计算二范数，即两个列表/向量之间的距离
    '''
    d = np.sqrt(sum([(argu1-argu2)**2 for argu1,argu2 in zip(x,y)]))
    return d
def alpha_update(gamma):
    '''
    作用：更新响应度
    原理：根据上一步的响应度，来计算新的迭代值，依照书上公式打的
    '''
    alpha_new = gamma.mean(axis=0)
    return alpha_new

# -------------------------EM 主要程序-------------------
def EM(mu, sigma, alpha=alpha_init, EPSILON=0.001):
    """
    mu：各总体的均值
    sigma：各总体的标准差
    alpha：隐含变量的初值
    EPSILON: 两次迭代，距离小于这个数，跳出循环
    """
    # 每个身高，每一个种类的概率密度函数值，直接根据身高数据全部算出来了，没有设定函数。
    PDF = np.zeros((N,J))
    for i,height in enumerate(Mixed_height):
        for j in range(J):
            PDF[i,j] = normal_pdf(height,mu[j],sigma[j])
    # ----------------------3、E步，求期望。 ----------------------
    # 不同类，不同参数，具体的概率密度函数
    def gammajk(i, j, alpha):
        '''
        作用：计算分模型j对观测数据height[i]的响应度
        i: 第i个身高数据
        j: 假设这个身高数据所属的模型类别j
        '''
        p_multi = 0
        for jj in range(J):
            p_multi += alpha[jj]*PDF[i,jj]
        temp = alpha[j]*PDF[i,j]
        return temp/p_multi

    # 迭代，第t步和第t+1步 之间是怎么继承上一步的？
    alpha_cur = list(alpha_init) # 将之前的元组 --> 列表
    alpha_new = [0.4, 0.6] # 随便设置的跟上一步不同的数

    # ----------------------5、中止条件 ----------------------
    while 1:
        '''
        作用：设置迭代循环，找到最优的参数
        停止条件：两次迭代之间的距离小于EPSILON
        '''
        # 计算所有的响应度，方便以后计算，以及下次的更新
        gamma = np.zeros((N,J))
        for i in range(N):
            for j in range(J):
                gamma[i,j] = gammajk(i,j,alpha_cur)

        # ----------------------4、M步 ----------------------
        alpha_new = alpha_update(gamma) # 更新参数值

        # 判断条件
        if EucliDis(alpha_cur,alpha_new) < EPSILON:
            break
        alpha_cur = alpha_new # 将新的参数传到alpha_cur中，方便进行下次迭代
    
    return alpha_new

# 算法耗费的时间
# 算法的参数估计结果
t_start = timeit.default_timer()
print('last result is',EM(mu,sigma))
t_end = timeit.default_timer()

cost = t_end - t_start
print('Time cost of EM is %f'%cost)