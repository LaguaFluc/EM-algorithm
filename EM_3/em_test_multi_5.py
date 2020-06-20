'''
给定各类的比例，估计总体均值和方差
'''
import numpy as np
import csv
import random
from scipy.stats import norm
import json
# import timeit

class EM_method(object):
    def __init__(self, path, class_num, sample_num):
        self.path = path
        self.sample_num = sample_num
        self.class_num = class_num
        self.data = self.get_data()
        self.alpha = [0.4, 0.4, 0.2]

    def get_data(self):
        data_temp = []
        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                data_temp.append(float(line[0]))
        print('You have drawn the observed data from the CSV file.')
        return np.array(data_temp) # np.array

    def initial_paras(self):
        mus_init = [1, 3, 2]
        sigmas_init = [1, 3, 2]
        print('Data initialized.')
        print(mus_init, sigmas_init)
        return mus_init, sigmas_init

    # 样本，对应所有类的，概率密度函数值
    def calc_pdf(self, paras_cur):
        mus_cur, sigmas_cur = paras_cur
        temp = np.zeros((self.sample_num, self.class_num))
        for sample_i in range(self.sample_num):
            for class_i in range(self.class_num):
                temp[sample_i, class_i] = norm.pdf(
                    self.data[sample_i],
                    loc=mus_cur[class_i],
                    scale=sigmas_cur[class_i]
                    )
                # temp[sample_i, class_i] = self.normal_pdf(self.data[sample_i], mus_cur[class_i], sigmas_cur[class_i])
                temp[sample_i, class_i] = norm.pdf(self.data[sample_i], loc=mus_cur[class_i], scale=sigmas_cur[class_i])

        return temp # np.array
    
    # ---------------------E-step------------------------
    def gamma_j_k(self, j, k, paras_cur):
        mus_cur, sigmas_cur = paras_cur
        numerator = self.alpha[k] * self.pdf_val[j,k]
        # denominator = np.dot(pdf_val, np.array(alpha))[j]
        denominator = (self.pdf_val @ np.array(self.alpha))[j]
        return numerator / denominator

    # 计算所有的gamma值，构成一个二维数组，numpy
    def calc_gamma(self, paras_cur):
        gamma = np.zeros((self.sample_num, self.class_num))
        self.pdf_val = self.calc_pdf(paras_cur)
        for j in range(self.sample_num):
            for k in range(self.class_num):
                gamma[j,k] = self.gamma_j_k(j,k,paras_cur)

        return gamma


    def update_paras(self, paras_cur):
        # 继承上一步的参数值
        mus_cur, sigmas_cur = paras_cur
        # 先计算gamma的值
        gamma = self.calc_gamma(paras_cur)
        # 计算分母，对列求和
        denominator = gamma.sum(axis=0) # numpy array [3, ]
        mus_new = np.zeros(self.class_num)
        for k in range(self.class_num):
            mus_new[k] = (self.data @ gamma[:,k]) / denominator[k]
        # mus_new = (self.data @ gamma) / denominator # [N,] @ [N, J]

        def y_mu_k(data, mus_new, k):
            return np.array(data - mus_cur[k]) # np.array
        # sigmas_new = [0 for i in range(self.class_num)]
        # 初始化sigmas_new
        sigmas_new = np.zeros(self.class_num)
        for k in range(self.class_num):
            y_new = y_mu_k(self.data, mus_cur, k)
            sigmas_new[k] = np.sqrt((y_new**2 @ gamma[:,k]) / denominator[k])
        print('Data updated.')
        return mus_new, sigmas_new
  
    def EucliDis(self, x, y):
        temp = np.sqrt(sum([(arg1-arg2)**2 for arg1,arg2 in zip(x,y)]))
        return temp

    def iteration(self, EPSILON=0.01, MAX_ITERATION=300):
        print('Function excuted.')
        paras_cur = self.initial_paras()
        mus_cur, sigmas_cur = paras_cur
        i = 0
        while 1:
            i += 1
            print(i)
            # mus_c = np.array([1, 3, 5])
            # sigmas_c = [1, 5, 10]
            # alpha_c = [0.4, 0.3, 0.3]
            mus_new, sigmas_new = self.update_paras(paras_cur)
            condition_1 = self.EucliDis(mus_cur, mus_new) < EPSILON
            condition_2 = self.EucliDis(sigmas_cur, sigmas_new) < EPSILON
            # print('sigmas_cur ',self.EucliDis(sigmas_cur,sigmas_new))
            # print('sigmas_c ',self.EucliDis(sigmas_c,sigmas_new))
            # condition_3 = self.EucliDis(alpha, alpha_new) < EPSILON
            # print('alpha_c ',self.EucliDis(alpha_c, alpha_new))
            condition = [condition_1, condition_2]
            condition_4 = i > MAX_ITERATION
            if all(condition) or condition_4:
                break
            # mus_cur, sigmas_cur, alpha = mus_new, sigmas_new, alpha_new
            paras_cur = (mus_new, sigmas_new)
            print('mus_new',mus_new)
            print('sigmas_new',sigmas_new)
        return mus_new, sigmas_new


path_data = r'./em_data_multi.csv'
path_paras = r'./config.json'
with open(path_paras, 'r', encoding='utf-8') as f:
    configs = json.load(f) # 对数据解码 将json -> dict

sample_num = configs["sample_num"]
class_num = 3

# 实例化
Sample_EM = EM_method(path_data, class_num, sample_num)
print("------------------")
mus_new, sigmas_new = Sample_EM.iteration()

print(mus_new)
print(sigmas_new)
