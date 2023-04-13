# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
import h5py

indicator = 'NOx'
folder = 'test18'
indexes = ['bi']

#处理input embedding + concate

#读入车辆信息，one_hot embedding
readpath = r'/home/zhanzxy5/data/data-Zhaopei/车辆信息.xlsx'
df_info = pd.read_excel(readpath, sheet_name = 'info')
df_profile = pd.read_excel(readpath, sheet_name = 'speed-acce')
pro_id_list = df_profile['ID'].values.tolist()

engine_type_list = list(set(df_info['发动机型号'].values.tolist()))
vehicle_type_list = list(set(df_info['车辆型号'].values.tolist()))
utility_list = list(set(df_info['车辆用途'].values.tolist()))
std_list = list(set(df_info['排放标准'].values.tolist()))
length = [len(engine_type_list),len(vehicle_type_list),
         len(utility_list),len(std_list)]
carid = df_info['车辆编号'].values.tolist()
vid = df_info['车牌号'].values.tolist()
names = ['发动机型号','车辆型号','车辆用途','排放标准']

for indicator in indexes:
    print(indicator)
    if indicator == 'fuel':
        input_col = ['速度（km/h）', '发动机转速（rpm）', '发动机净输出扭矩（%）',
                     '摩擦扭矩（%）','distance(m)','经度（°）','纬度（°）',
                     'weekday', 'month', 'hour','排量', '总质量','最大输出功率',
                     '牵引车','注册地']
        #static_col = []
        output_col = ['发动机燃料流量（L/h）']
    else:
        if indicator == 'NOx':
            input_col = ['速度（km/h）', '发动机转速（rpm）', '发动机净输出扭矩（%）',
                     '摩擦扭矩（%）','distance(m)','经度（°）','纬度（°）',
                     'weekday', 'month', 'hour','排量', '总质量','最大输出功率',
                     '牵引车','注册地']
            #static_col = ['经度（°）','纬度（°）']
            output_col = ['nox']
        else:
            input_col = ['速度（km/h）', '发动机转速（rpm）', '发动机净输出扭矩（%）',
                     '摩擦扭矩（%）','distance(m)','经度（°）','纬度（°）',
                     'weekday', 'month', 'hour','排量', '总质量','最大输出功率',
                     '牵引车','注册地']
            #static_col = []
            output_col_fuel = ['发动机燃料流量（L/h）']
            output_col_NOx = ['nox']
        
    datapath = r'/home/zhanzxy5/data/data-Zhaopei/'+folder+'/normalized/train'
    readfolder = datapath + '/' + indicator
    files = os.listdir(readfolder)
    
    validation_ratio = 0.1
    valid_num = set(np.random.randint(len(files), size = int(validation_ratio*len(files))))
    carnum = 0
    
    #区分静态、动态信息
    train_input_static = []
    train_input_dynamic = []
    train_output_fuel = []
    train_output_NOx = [] 
    train_input_ids = []
    train_input_profile = []
    
    val_input_static = []
    val_input_dynamic = []
    val_input_ids = []
    val_output_fuel = []
    val_output_NOx = []
    val_input_profile = []
    
    for file in files:
        if (file.find('._') != -1) | (file.find('$') != -1):
            continue
        carnum += 1  
        if carnum % 10000 == 0:
            print(carnum)
          
        readfile = readfolder + '/' + file    
        df = pd.read_csv(readfile,encoding = 'gbk')
        if df.shape[0] != df.dropna().shape[0]:
            continue
        one_hot_embedding = []
        one_hot_embedding_std = []
        for i in range(len(names)):
            s = np.zeros(length[i])
            if names[i] == '发动机型号':
                k = engine_type_list.index(df[names[i]][0])
            if names[i] == '车辆型号':
                k = vehicle_type_list.index(df[names[i]][0])
            if names[i] == '车辆用途':
                k = utility_list.index(df[names[i]][0])
            if names[i] == '排放标准':
                k = std_list.index(df[names[i]][0])
            s[k] = 1
            if names[i] == '排放标准':
                one_hot_embedding_std += list(s)
            else:
                one_hot_embedding += list(s)    
        f = df[input_col].values.tolist() #dynamic info-input
        ff_fuel = df[output_col_fuel] #output
        ff_NOx = df[output_col_NOx]
               
        #静态信息
        static = []  #static info
        for i in range(df.shape[0]):
            static.append(one_hot_embedding)
            f[i] =  f[i] + one_hot_embedding_std
      
        idlist = [] #idlist-output
        k = carid[vid.index(df['车牌号'][0])]
        #for i in range(f.shape[0]):
            #static.append(one_hot_embedding)
        for i in range(df.shape[0]):
            idlist.append(k)
        
        #加速度、加速度profile
        profile = []
        k = pro_id_list.index(df['车牌号'][0])
        l = df_profile.loc[k].values.tolist()
        for i in range(df.shape[0]):
            profile.append(l[2:])
        
        
        if carnum in valid_num:
            val_input_static.append(static) 
            val_input_dynamic.append(f)
            val_input_profile.append(profile)
            val_output_fuel.append(ff_fuel.values.tolist())
            val_output_NOx.append(ff_NOx.values.tolist())
            val_input_ids.append(idlist)
        else:
            train_input_static.append(static) 
            train_input_dynamic.append(f)
            train_input_profile.append(profile)
            train_input_ids.append(idlist)
            train_output_fuel.append(ff_fuel.values.tolist())
            train_output_NOx.append(ff_NOx.values.tolist())
    
    #print(train_input.shape)
    #sprint(test_input.shape)
    print('carnum', carnum)
    print('fuel mean',np.mean(train_output_fuel),np.mean(val_output_fuel))
    print('NOx mean',np.mean(train_output_NOx),np.mean(val_output_NOx))
       
    filename = r'/home/zhanzxy5/data/data-Zhaopei/'+folder + '/h5/include profile/' +'DataSet_'+indicator+'_'+folder+'_0512.h5'
    file = h5py.File(filename,'w')
    file.create_dataset('train_input_static', data = train_input_static)
    file.create_dataset('train_input_dynamic', data = train_input_dynamic)
    file.create_dataset('train_input_profile', data = train_input_profile)
    file.create_dataset('train_output_fuel', data = train_output_fuel)
    file.create_dataset('train_output_NOx', data = train_output_NOx)
    file.create_dataset('train_input_ids', data =  train_input_ids)
    
    file.create_dataset('val_input_static', data = val_input_static)
    file.create_dataset('val_input_dynamic', data = val_input_dynamic)
    file.create_dataset('val_output_fuel', data = val_output_fuel)   
    file.create_dataset('val_input_ids', data =  val_input_ids)
    file.create_dataset('val_output_NOx', data = val_output_NOx)
    file.create_dataset('val_input_profile', data = val_input_profile)
    file.close()
        
