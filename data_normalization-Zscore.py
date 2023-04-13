# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 21:07:16 2022

@author: wu535
"""
import pandas as pd
import numpy as np
import os
import h5py

#读入车辆信息，one_hot embedding
indicator = 'bi'
df_info = pd.read_excel(r'/home/zhanzxy5/data/data-Zhaopei/车辆信息.xlsx', sheet_name = 'info')

if indicator == 'fuel':
    input_col = ['速度（km/h）', '发动机转速（rpm）', '发动机净输出扭矩（%）',
                 '摩擦扭矩（%）','经度（°）','纬度（°）','distance(m)',
                 'weekday', 'month', '排量', '总质量','最大输出功率', 'hour',
                 '发动机燃料流量（L/h）']
else:
    input_col = ['速度（km/h）', '发动机转速（rpm）', '发动机净输出扭矩（%）',
                 '摩擦扭矩（%）','经度（°）','纬度（°）','distance(m)','weekday',
                 'month', '排量', '总质量','最大输出功率', 'hour','nox',
                 '发动机燃料流量（L/h）']
  

datapath = r'/home/zhanzxy5/data/data-Zhaopei/z-score/raw'
readfolder1 = datapath + '/' +'train' +'/'+indicator
files1 = os.listdir(readfolder1)
readfolder2 = datapath + '/' +'test' +'/'+indicator
files2 = os.listdir(readfolder2)

writefolder = r'/home/zhanzxy5/data/data-Zhaopei/z-score/normalized'
writefolder1 = writefolder + '/' +'train' +'/'+indicator
if not os.path.exists(writefolder1):
    os.mkdir(writefolder1)
writefolder2 = writefolder + '/' +'test' +'/'+indicator
if not os.path.exists(writefolder2):
    os.mkdir(writefolder2)

mean_value = []
std_value = []  
print(indicator)
carnum = 0
for file in files1:
    if (file.find('._') != -1) | (file.find('$') != -1):
        continue
    carnum += 1  
    if carnum % 1000 == 0:
        print(carnum)
        break
    readfile = readfolder1 + '/' + file    
    f = pd.read_csv(readfile,encoding = 'gbk')
    #print(f.shape)
    if carnum == 1:
        df = f
    else:
        df = df.append(f)
df.index = range(df.shape[0])
    
for c in input_col:
    values = []
    v = df[c].values.tolist()
    null = df[c].isnull()
    for i in range(len(v)):
    	
        if (null[i] == True):
            continue
        values.append(v[i])
    mean_value.append(np.mean(values))
    std_value.append(np.std(values))
        
writefile = writefolder + '/' +'Mean-std_'+ indicator + '.xlsx'
ff1 = pd.DataFrame(dict(zip(input_col,zip(mean_value,std_value))))
ff1.to_excel(writefile)


print('Mean_std done!')
print('')
carnum = 0
for file in files1:
    if (file.find('._') != -1) | (file.find('$') != -1):
        continue
    carnum += 1  
    if carnum % 10000 == 0:
        print(carnum)
    readfile = readfolder1 + '/' + file   
    df = pd.read_csv(readfile,encoding = 'gbk')
    for c in input_col:
        values = df[c].values.tolist()
        k = input_col.index(c)
        vmean = mean_value[k]
        vstd = std_value[k]
        for i in range(len(values)):
            values[i] = (values[i]) / (vstd) + 1e-8
        df[c] = values
    writefile = writefolder1 + '/' +file
    df.to_csv(writefile, encoding = 'gbk')
    
    
for file in files2:
    if (file.find('._') != -1) | (file.find('$') != -1):
        continue
    carnum += 1  
    if carnum % 10000 == 0:
        print(carnum)
    readfile = readfolder2 + '/' + file    
    df = pd.read_csv(readfile,encoding = 'gbk')
    for c in input_col:
        values = df[c].values.tolist()
        k = input_col.index(c)
        vmean = mean_value[k]
        vstd = std_value[k]
        for i in range(len(values)):
            values[i] = (values[i]) / (vstd) + 1e-8
        df[c] = values
    writefile = writefolder2 + '/' +file
    df.to_csv(writefile, encoding = 'gbk')
