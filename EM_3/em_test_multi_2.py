import numpy as np
import random
import csv
import json
from scipy.stats import norm
import timeit

class EM_method(object):
    def __init__(self, sample_num, class_num, path):
        self.sample_num = sample_num
        self.class_num = class_num
        self.path = path
        self.data = self.get_data()

    def get_data(self):
        data_temp = []
        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                data_temp.append(float(line[0]))
        print('Data has been drawn from the CSV file.')
        return np.array(data_temp)
    
    def init_paras(self):
        mu_init = np.array(random.sample(range(160, 200), self.class_num))
        sigma_init = np.ones(self.class_num)
        alpha_init = np.array([1/class_num for i in range(self.class_num)])
        paras_init = (mu_init, sigma_init, alpha_init)
        return paras_init

    def clac_pdf(self, paras_cur):
        mu_cur, sigma_cur, alpha_cur = paras_cur
        pdf_val = np.zeros((self.sample_num, self.class_num))
        # pdf_val = norm.pdf(self.data, mu_cur, sigma_cur)
        for j in range(self.sample_num):
            for k in range(self.class_num):
                pdf_val[j, k] = norm.pdf(self.data[j], loc=mu_cur[k], scale=sigma_cur[k])
        
        return pdf_val
    
    def gamma_jk(self, j, k, paras_cur):
        pass

path_configs = r".\config.json"
path_data = r".\em_data_multi.csv"


with open(path_configs, 'r', encoding='utf-8') as f:
    configs = json.load(f)
sample_num = configs["sample_num"]
class_num = configs["class_num"]

concrete_sample = EM_method(sample_num, class_num, path_data)

paras_init = concrete_sample.init_paras()
pdf_val = concrete_sample.clac_pdf(paras_init)

print(pdf_val[0,0])


