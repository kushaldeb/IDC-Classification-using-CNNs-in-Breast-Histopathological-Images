#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 6 12:04:25 2019

@author: kushaldeb

"""

import shutil
import os
import numpy as np
from tqdm import tqdm

base_dir = '/home/kushaldeb/Desktop/D/8th-Semester/Breast-Cancer-Classification/Dataset/'

train_0 = base_dir + "Training/0"
train_1 = base_dir + "Training/1"

val_0 = base_dir + "Validation/0"
val_1 = base_dir + "Validation/1"

test_0 = base_dir + "Testing/0"
test_1 = base_dir + "Testing/1"

files_0 = os.listdir(train_0)
files_1 = os.listdir(train_1)

for f_v in tqdm(files_0):
   if np.random.rand(1) < 0.2:
       shutil.move(train_0  + "/" + f_v, val_0 + "/" + f_v)

for i_v in tqdm(files_1):
   if np.random.rand(1) < 0.2:
       shutil.move(train_1  + "/" + i_v, val_1 + "/" + i_v)
  
filesv_0 = os.listdir(val_0)
filesv_1 = os.listdir(val_1)
    
for f_t in tqdm(filesv_0):
    if np.random.rand(1) < 0.25:
        shutil.move(val_0  + "/" + f_t, test_0 + "/" + f_t)

for i_t in tqdm(filesv_1):
    if np.random.rand(1) < 0.25:
        shutil.move(val_1  + "/" + i_t, test_1 + "/" + i_t)
        
print(len(os.listdir(train_0)))
print(len(os.listdir(train_1)))

print(len(os.listdir(val_0)))
print(len(os.listdir(val_1)))

print(len(os.listdir(test_0)))
print(len(os.listdir(test_1)))
