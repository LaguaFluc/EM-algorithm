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

    def get_data(self):
        data_temp = []
        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                data_temp.append(float(line[0]))
        print('You have drawn the observed data from the CSV file.')
        # print('data_temp[999]',data_temp[999])
        return np.array(data_temp) # np.array

    def initial_paras(self):
        # mus_init = random.sample(range(4, 10), self.class_num)
        mus_init = [3, 2, 1]
        # mus_init = [170, 170, 175]
        # sigmas_init = random.sample(range(1,10), self.class_num)
        # sigmas_init = [1. for i in range(self.class_num)]
        sigmas_init = [1., 1., 1.]
        # alpha_init = [1/class_num for i in range(self.class_num)]
        alpha_init = [0.4, 0.3, 0.3]
        print(mus_init, sigmas_init, alpha_init)
        return mus_init, sigmas_init, alpha_init

    # def normal_pdf(self, x, mu_num, sigma_num):
    #     temp = np.exp(-((x-mu_num)**2)/(2*sigma_num**2)) / (sigma_num * np.sqrt(2*np.pi))
    #     return temp
    # 样本，对应所有类的，概率密度函数值
    def calc_pdf(self, paras_cur):
        mus_cur, sigmas_cur, alpha_cur = paras_cur
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
        mus_cur, sigmas_cur, alpha_cur = paras_cur
        numerator = alpha_cur[k] * self.pdf_val[j,k]
        # denominator = np.dot(pdf_val, np.array(alpha))[j]
        denominator = (self.pdf_val @ np.array(alpha_cur))[j]
        return numerator/denominator

    # 计算所有的gamma值，构成一个二维数组，numpy
    def calc_gamma(self, paras_cur):
        gamma = np.zeros((self.sample_num, self.class_num))
        self.pdf_val = self.calc_pdf(paras_cur)
        for j in range(self.sample_num):
            for k in range(self.class_num):
                gamma[j,k] = self.gamma_j_k(j,k,paras_cur)
        # gamma = self.calc_pdf(paras_cur)
        # mus, sigmas, alpha = paras_cur
        # gamma = gamma @ np.array(alpha)
        # temp = gamma.sum(axis=1) # 对行
        # gamma = (gamma.T / temp).T
        return gamma


    def update_paras(self, paras_cur):
        mus_cur, sigmas_cur, alpha_cur = paras_cur

        gamma = self.calc_gamma(paras_cur)
        denominator = gamma.sum(axis=0) # numpy array [3, ]
        # print('denominator.shape',denominator.shape)
        mus_new = (self.data @ gamma) / denominator # [N,] @ [N, J]

        def y_mu_k(data, mus_new, k):
            return np.array(data - mus_new[k]) # np.array
        sigmas_new = [0 for i in range(self.class_num)]
        for k in range(self.class_num):
            y_new = y_mu_k(self.data, mus_new, k)
            sigmas_new[k] = np.sqrt((y_new**2 @ gamma[:,k]) / denominator[k])
        alpha_new = list(denominator / self.sample_num)
        # alpha_new = gamma.mean(axis=0)

        return mus_new, sigmas_new, alpha_new
    
    def EucliDis(self, x, y):
        temp = np.sqrt(sum([(arg1-arg2)**2 for arg1,arg2 in zip(x,y)]))
        return temp

    def iteration(self, EPSILON=1, MAX_ITERATION=300):
        paras_cur = self.initial_paras()
        mus_cur, sigmas_cur, alpha_cur = paras_cur
        i = 0
        while 1:
            i += 1
            print(i)
            # 真实值
            # mus_c = np.array([1, 3, 5])
            # sigmas_c = [1, 5, 10]
            # alpha_c = [0.4, 0.3, 0.3]
            mus_new, sigmas_new, alpha_new = self.update_paras(paras_cur)
            condition_1 = self.EucliDis(mus_cur, mus_new) < EPSILON
            # print('mus_c',self.EucliDis(mus_c, mus_new))
            condition_2 = self.EucliDis(sigmas_cur, sigmas_new) < EPSILON
            # print('sigmas_cur ',self.EucliDis(sigmas_cur,sigmas_new))
            # print('sigmas_c ',self.EucliDis(sigmas_c,sigmas_new))
            condition_3 = self.EucliDis(alpha_cur, alpha_new) < EPSILON
            # print('alpha_c ',self.EucliDis(alpha_c, alpha_new))
            condition = [condition_1, condition_2, condition_3]
            condition_4 = i > MAX_ITERATION
            if all(condition) or condition_4:
                break
            paras_cur = (mus_new, sigmas_new, alpha_new)
            print(mus_new)
            print(sigmas_new)
            print(alpha_new)
        return mus_new, sigmas_new, alpha_new


path_data = r'./em_data_multi.csv'
path_paras = r'./config.json'
with open(path_paras, 'r', encoding='utf-8') as f:
    configs = json.load(f)
    # configs = json.load(f) # 对数据解码 将json -> dict

sample_num = configs["sample_num"]
class_num = 3

# 实例化
Sample_EM = EM_method(path_data, class_num, sample_num)

# print(pdf_val)
# print(temp_gamma_jk)
# t_start = timeit.default_timer()
# gamma = Sample_EM.calc_gamma(paras_cur)[1,1]
# t_end = timeit.default_timer()
# cost = t_end - t_start
# print('Time cost of calculating gamma is %f'%cost)
# print(gamma)

mus_new, sigmas_new, alpha_new = Sample_EM.iteration()
print(mus_new)
print(sigmas_new)
print(alpha_new)
