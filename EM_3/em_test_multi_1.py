import numpy as np
import csv
import random
from scipy.stats import norm
import json

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
        return data_temp

    def initial_paras(self):
        mus_init = random.sample(range(160,191), self.class_num)
        sigmas_init = random.sample(range(1,10), self.class_num)
        alpha_init = [1/class_num for i in range(self.class_num)]
        return mus_init, sigmas_init, alpha_init

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
                # temp[sample_i, class_i] = norm.pdf(self.data[sample_i], loc=mus_cur[class_i], scale=sigmas_cur[class_i])

        return temp # np.array
    
    # ---------------------E-step------------------------
    def gamma_j_k(self, j, k, paras_cur):
        mus_cur, sigmas_cur, alpha_cur = paras_cur
        # numerator = 0 # 分子
        # denominator = 0 # 分母
        pdf_val = self.calc_pdf(paras_cur)
        numerator = alpha_cur[k] * pdf_val[j,k]
        # denominator = np.dot(pdf_val, np.array(alpha))[j]
        denominator = (pdf_val @ np.array(alpha_cur))[j]
        return numerator/denominator

    def gamma_jk(self, j, k, paras_cur):
        mus_cur, sigmas_cur, alpha_cur = paras_cur
        pdf_val = self.calc_pdf(paras_cur)
        
        return alpha_cur[k] * pdf_val[j,k]

    def calc_gamma(self, paras_cur):
        mus_cur, sigmas_cur, alpha_cur = paras_cur
        gamma = np.zeros((self.sample_num, self.class_num))
        for j in range(self.sample_num):
            for k in range(self.class_num):
                gamma[j, k] = self.gamma_jk(j, k, paras_cur)

        pdf_val = self.calc_pdf(paras_cur) # [N, K]
        numerator = pdf_val @ np.array(alpha_cur)
        denominator = numerator.sum(axis=1) # [N, 1]
        return gamma / denominator




    # def update_paras(self, paras_cur):
    #     mus_cur, sigmas_cur, alpha_cur = paras_cur
    #     gamma = self.calc_gamma(paras_cur)

    #     denominator = gamma.sum(axis=0) # numpy array
    #     mus_new = (self.data @ gamma) / denominator

    #     def y_mu_k(data, mu_new, k):
    #         return data - mu_new[k]
    #     sigmas_new = [0 for i in range(len(sigma_cur))]
    #     for k in range(len(sigmas_new)):
    #         sigmas_new[k] = update_sigma(self.data, mus_new, k)
    #     alpha_new = list(denominator / self.sample_num)

    #     return mus_new, sigmas_new, alpha_new
    
    # def EucliDis(self, x, y):
    #     temp = np.sqrt(sum([(arg1-arg2)**2 for arg1,arg2 in zip(x,y)]))
    #     return temp

    # def iteration(self, data, EPSILON=0.01, MAX_ITERATION=1000):
    #     paras_cur = self.initial_paras()
    #     mus_cur, sigmas_cur, alpha_cur = paras_cur
    #     i = 0
    #     while 1:
    #         i += 1
    #         print(i)
    #         mus_new, sigmas_new, alpha_new = self.update_paras(paras_cur)
    #         print('I have come to condition_1')
    #         condition_1 = self.EucliDis(mus_cur, mus_new) < EPSILON
    #         print('condition_2')
    #         condition_2 = self.EucliDis(sigmas_new, sigmas_new) < EPSILON
    #         condition_3 = self.EucliDis(alpha_cur, alpha_new)
    #         condition = [condition_1, condition_2, condition_3]
    #         condition_4 = i > MAX_ITERATION
    #         if all(condition) or condition_4:
    #             break
    #         mus_cur, sigmas_cur, alpha_cur = mus_new, sigmas_new, alpha_new
    #     return mus_new, sigmas_new, alpha_new


path_data = r'./em_data_multi.csv'
path_paras = r'./config.json'
with open(path_paras, 'r', encoding='utf-8') as f:
    configs = json.load(f)
    # configs = json.load(f) # 对数据解码 将json -> dict


# mus = configs["mu"]
# sigmas = configs['sigma']
sample_num = configs['sample_num']
# alpha = configs['alpha']
# class_num = len(alpha)
class_num = 3

Sample_EM = EM_method(path_data, class_num, sample_num)

# Sample_EM.iteration(Sample_EM.data)
paras_cur = Sample_EM.initial_paras()
mus_init, sigmas_init, alpha_init = paras_cur
# print(mus_init)
# print(sigmas_init)
# print(alpha_init)
pdf_val = Sample_EM.calc_pdf(paras_cur)
temp_gamma_jk = Sample_EM.gamma_j_k(1,1,paras_cur)

# print(pdf_val)
print(temp_gamma_jk)
gamma = Sample_EM.calc_gamma(paras_cur)[1,1]
print(gamma)

