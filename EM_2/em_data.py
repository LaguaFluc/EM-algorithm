'''
1. 给定数据，需要生成样本的正态分布参数
2. 使用numpy中的random方法生成数据
3. 存储数据，写进CSV文件中。
'''

import numpy as np
import csv


# -------------------1. 给定数据，需要生成样本的正态分布参数-------------------------
# 初始化试验数据。
# normal distribution with mu and sigma
mu = [165, 175]
sigma = [6, 5]
# sample_num_0, sample_num_1 = sample_num
sample_num = [600, 400]
N = sum(sample_num) # 样本容量

# -------------------2. 使用numpy中的random方法生成数据-------------------------
# 生成size_in个服从正态分布的数据
def create_normal(mu, sigma, size_in):
    """
    params: one-dimensional data.
    """
    temp = np.random.normal(loc=mu, scale=sigma, size=size_in)
    return temp # 返回的是一个列表
# 生成数据
def create_data(mu, sigma, size_in):
    """
    paras:
    all the parameters have the same dimension, 
    not one-dimensional,
    all of them are list.
    mu: the mean of the normal distribution.
    sigma: the standard deviation of the normal distribution.
    size_in: the number of each category.
    """
    # 依次按照不同的分布，生成数据
    tag_data = [] # 带有标签的数据
    for i in range(len(sample_num)):
        tag_data.append(list(create_normal(mu[i],sigma[i],sample_num[i]))) # list(): np.ndarray -> list      
    return tag_data # 一个列表，里面的元素为列表。

male, female = create_data(mu,sigma,sample_num)
# print(type(male)) # 列表

# 数据存储地址：
path_male = r"D:\lagua\study\coding\PythonStudy\EM_data\male_1.csv"
path_female = r"D:\lagua\study\coding\PythonStudy\EM_data\female_1.csv"


# -------------------3. 存储数据，写进CSV文件中。-------------------------
# 将生成的数据放到CSV文件中。
def write_data(li, path_li):
    # 将以上生成的数据写入到CSV文件中。
    for li_i in li:
        with open(path_li, "w+", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            to_be_write = []
            # 判断应该写入哪个文件。
            if path_li == path_male:
                tag_ = [0 for i in range(len(li))]
            elif path_li == path_female:
                tag_ = [1 for i in range(len(li))]
            else:
                print('The path that you have entered does not exist.')            
            # 将生成的数据，按照列表打包，写入到CSV文件中。
            # 添加进CSV文件中的数据：第一列为标签，第二列为身高数据
            for tag_i, li_h in zip(tag_, li):
                to_be_write.append([tag_i,li_h])
            writer.writerows(to_be_write)    
    print('The data has been wirtten to the csv file.')

# 执行函数，写入数据。
write_data(male,path_male)
write_data(female,path_female)
