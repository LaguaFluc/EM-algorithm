'''
1. 给定数据，需要生成样本的正态分布参数
2. 使用numpy中的random方法生成数据
3. 存储数据，写进CSV文件中。
'''
# 两步
# 三步
# 1 给定参数
# 2 随机生成数据
# 3 存储数据，写进CSV文件中。

import numpy as np
from scipy.stats import norm
import json
import csv


class Init_data(object):
    def __init__(self, mus, sigmas, sample_num, alpha):
        self.mus = mus # 均值列表
        self.sigmas = sigmas # 标准差列表
        self.sample_num = sample_num # 总的样本个数
        self.alpha = alpha # 样本比重列表
        self.sample_num_class = [alpha_i*sample_num for alpha_i in alpha] # 每类样本个数，列表
        self.class_num = len(mus) # 多少类，类别的个数
    
    def create_norm(self):
        # , mus=self.mus, sigmas=self.sigmas, sample_size=self.sample_num_class
        temp =[] # 是列表，里面的元素是列表，列表元素的长度不一定相等
        for j in range(self.class_num):
            temp.append(list(norm.rvs(loc=self.mus[j], scale=self.sigmas[j], size=int(self.sample_num_class[j]))))
        data = []
        for temp_i in temp:
            for j in temp_i:
                data.append(j)
        print('Data created')
        return data
    
    def write_data(self, data, path):
        with open(path, 'w+', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for data_i in data:
                writer.writerow([data_i])
        print('Data collected.')

with open("./config.json", encoding="utf-8") as j:
    configs = json.load(j) # 对数据解码 将json -> dict

mus = configs["mu"]
sigmas = configs['sigma']
sample_num = configs['sample_num']
alpha = configs['alpha']

data_collection = Init_data(mus, sigmas, sample_num, alpha)
data_collection.write_data(data_collection.create_norm(), '.\em_data_multi.csv')