#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 4 13:19:51 2019

@author: kushaldeb

"""

import glob
import shutil
import os
from tqdm import tqdm


dir_list = [folder for folder in os.listdir('/home/kushaldeb/D/B.Tech/8th-Semester/Breast-Cancer-Classification/Original-Dataset')]
subdir_list = [0, 1]

for j in tqdm(subdir_list):
    for i in dir_list:
        src_path = '/home/kushaldeb/D/8th-Semester/Breast-Cancer-Classification/Original-Dataset/'+str(i)+'/'+str(j)
        dst_path = '/home/kushaldeb/D/8th-Semester/Breast-Cancer-Classification/Dataset/'+str(j)
        
        for file in glob.iglob(os.path.join(src_path, "*.png")):
            shutil.move(file, dst_path)
