#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:50:00 2019

@author: kushaldeb
"""

import os
import shutil
import random
from tqdm import tqdm

dir_list = ['Training', 'Validation',  'Testing']
subdir_list = ['0', '1']
dir_dict = {'Training':50000,
            'Validation':5000,
            'Testing':500}

src_base = '/home/kushaldeb/D/8th Semester/Breast Cancer Classification/Dataset/'
dst_base = '/home/kushaldeb/D/8th Semester/Breast Cancer Classification/sample_dataset/'

for i in tqdm(dir_list):
    for j in subdir_list:
        src = src_base + i + '/' + j
        dst = dst_base + i + '/' + j
        files = [file for file in os.listdir(src) if os.path.isfile(os.path.join(src, file))]
        amount = dir_dict[i]
        for x in range(amount):
            file = random.choice(files)
            shutil.copyfile(os.path.join(src, file), os.path.join(dst, file))
            files.remove(file)
