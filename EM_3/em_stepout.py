import numpy as np
from scipy.stats import norm
import json
import random

# path = r'./em_data_multi.csv'
# path = r'./config.json'
# with open(path, 'r') as f:
#     configs = json.load(f)

# li = random.sample(range(1,11), 3)

# print(li)

# print(norm(165,5).pdf(165))
# print(norm.pdf(165,loc=165,scale=5))

# # 传入参数
# def log(text):
#     # 传入函数参数
#     def decorator(func):
#         # 传入任意参数
#         def wrapper(*args, **kw):
#             # 打印最外层函数传入的参数，次外层函数的名称
#             print('%s %s():'%(text,func.__name__))
#             # 返回次外层函数，并且给带上里面函数的参数
#             return func(*args, **kw)

#         return wrapper # 返回最里面的函数
#     return decorator # 返回次外层函数


# @log('execute')
# def now():
#     print('2020-6-13')

# now()
# now = log('execute')(now)
# now()

arr1 = np.zeros(3)
print(arr1)
# print(arr1**2)


