#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: zhangyuhao
@file: driver_info_generate.py
@time: 2022/11/1 上午11:24
@email: yuhaozhang76@gmail.com
@desc: 
"""
import pickle
from tqdm import tqdm
order = pickle.load(open('./input/taxiweek_20090802.pickle', 'rb'))
for time in tqdm(order.keys()):
    orders = order[time]
    for ord in orders:
        if ord[11][0] != ord[1] or ord[11][-1] != ord[4]:
            print(ord[11][0])
            print(ord[1])
            print(ord)
