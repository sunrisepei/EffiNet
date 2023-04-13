# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:22:39 2022

@author: wu535

"""
import torch
import torch.nn as nn
import numpy as np
import time
import math
import h5py
import torch.nn.functional as F
import pandas as pd
import time
import random
import torch_scatter
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

batch_size = 10 
bsize = 50
epochs = 600 # The number of epochs
epoch_size = 30000
folder = 'test18'
indicator = 'bi'
epoch_start_time = time.time()
print('CUDA', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dimension
dim_static_fuel = 5
dim_static_NOx = 20
dim_profile_size = 10
f_size = 64
pred_feature_size = 64
dim_fusion = f_size
dim_pooling = 25




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,feature_size =f_size,num_layers=8,dropout=0.1):
        super(TransAm, self).__init__()
        #self.model_type = 'Transformer'
        #for fusion transformer 
        self.fc_static_dense_fuel = torch.nn.Linear(raw_static_size, dim_static_fuel)
        self.fc_static_dense_NOx = torch.nn.Linear(raw_static_size, dim_static_NOx)
        self.fc_profile = torch.nn.Linear(train_input_profile.size()[2], dim_profile_size)
        self.fc1 = torch.nn.Linear(ksize,f_size)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=5)      
        self.prepooling = nn.Linear(dim_prepooling,dim_pooling)
        #self.decoder_layer2 = nn.Linear(int(f_size/2),dim_fusion)
        
        #for prediction transformer-fuel
        self.fuel_mask = None
        self.fcfuel = torch.nn.Linear(dim_pred_fuel,pred_feature_size)
        self.fuel_pred_pos_encoder = PositionalEncoding(pred_feature_size)
        self.fuel_pred_encoder_layer = nn.TransformerEncoderLayer(d_model=pred_feature_size, nhead=8, dropout=dropout)
        self.fuel_pred_transformer_encoder = nn.TransformerEncoder(self.fuel_pred_encoder_layer, num_layers=num_layers)
        self.fuel_pred_fc1 = nn.Linear(pred_feature_size,int(pred_feature_size/2))
        self.fuel_pred_fc2 = nn.Linear(int(pred_feature_size/2),int(pred_feature_size/4))
        
        self.NOx_mask = None
        self.fcNOx = torch.nn.Linear(dim_pred_NOx,pred_feature_size)
        self.NOx_pred_pos_encoder = PositionalEncoding(pred_feature_size)
        self.NOx_pred_encoder_layer = nn.TransformerEncoderLayer(d_model=pred_feature_size, nhead=8, dropout=dropout)
        self.NOx_pred_transformer_encoder = nn.TransformerEncoder(self.NOx_pred_encoder_layer, num_layers=num_layers)
        self.NOx_pred_fc1 = nn.Linear(pred_feature_size,int(pred_feature_size/2))
        self.NOx_pred_fc2 = nn.Linear(int(pred_feature_size/2),int(pred_feature_size/4))

        #output FCN
        self.output_fc1 = nn.Linear(int(pred_feature_size/2),int(pred_feature_size/4))
        self.output_fc2 = nn.Linear(int(pred_feature_size/4),2)       
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    

    def pooling(self, src_pooling, src_ids):
        device = src_pooling.device
        res_pooling = torch.zeros(src_pooling.shape)
        #pooling_index = torch.zeros(src_ids.shape[0]).to(device).type(torch.int64)
        #print('src_pooling dim',src_pooling.shape)
        #pooling_index = src_ids.gather(1,pooling_index)
        index = src_ids.type(torch.int64)
        pooling_status = np.zeros(src_ids.shape[0])
        pooling_temp_mean = torch_scatter.scatter_mean(src_pooling, index, dim = 0)
        pooling_temp_std = torch_scatter.scatter_std(src_pooling, index, dim = 0)

        eps = torch.randn_like(src_ids) / 100
        
        eps = eps.repeat(dim_pooling, 1, 1).permute(1,2,0)
        
        index = index.unsqueeze(dim=2).repeat(1,1,dim_pooling)

        res_pooling_mean = pooling_temp_mean.gather(dim = 0, index = index)
        
        #assert 0
        res_pooling_std = pooling_temp_std.gather(dim = 0, index = index)

        return res_pooling_mean + eps * res_pooling_std
        
    def forward(self,src_dynamic, src_static,src_profile, src_ids):
        src_static_fuel = self.fc_static_dense_fuel(src_static)
        src_static_NOx = self.fc_static_dense_NOx(src_static)         
        
        src_profile = self.fc_profile(src_profile)
        src_fuel = torch.cat((src_static_fuel, src_profile), dim = 2)
        src_NOx = torch.cat((src_static_NOx, src_profile), dim = 2) 
        
        #fusion transformer
        #src = torch.cat((src_fuel, src_NOx), dim = 2)      
        src = src_dynamic
        src = self.fc1(src)    
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
            
        src = self.pos_encoder(src)
        fusion_output = self.transformer_encoder(src)#, self.src_mask)
    
        #pooling
        device = src.device
        src_pooling = torch.cat((src_fuel, fusion_output), dim = 2)
        src_pooling = self.prepooling(src_pooling)
        src_pooling = self.pooling(src_pooling, src_ids).to(device)     
    
        #predict Transformer
        pred_fuel = torch.cat((src_fuel, fusion_output,src_pooling), dim = 2) 
        pred_fuel = self.fcfuel(pred_fuel)
                
        if self.fuel_mask is None or self.fuel_mask.size(0) != len(pred_fuel):
            device = pred_fuel.device
            mask = self._generate_square_subsequent_mask(len(pred_fuel)).to(device)
            self.fuel_mask = mask        
        pred_fuel = self.pos_encoder(pred_fuel)
        output_fuel = self.fuel_pred_transformer_encoder(pred_fuel)
        output_fuel = F.relu(self.fuel_pred_fc1(output_fuel))
        output_fuel = self.fuel_pred_fc2(output_fuel)
        pred_NOx = torch.cat((src_NOx, fusion_output,src_pooling), dim = 2) 
        pred_NOx = self.fcNOx(pred_NOx)
        if self.NOx_mask is None or self.NOx_mask.size(0) != len(pred_NOx):
            device = pred_NOx.device
            mask = self._generate_square_subsequent_mask(len(pred_NOx)).to(device)
            self.NOx_mask = mask        
        pred_NOx = self.pos_encoder(pred_NOx)
        output_NOx = self.NOx_pred_transformer_encoder(pred_NOx)
        output_NOx = F.relu(self.NOx_pred_fc1(output_NOx))
        output_NOx = self.NOx_pred_fc2(output_NOx)
        
        #output
        output = torch.cat((output_fuel,output_NOx), dim = 2)
        output = F.relu(self.output_fc1(output))
        output = self.output_fc2(output)
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



def train(train_input_dynamic,train_input_static,train_input_profile, train_output_fuel,train_output_NOx, train_input_ids):
    model.train() # Turn on the train mode \o/
    total_loss = 0
    start_time = time.time()
    torch_dataset = torch.utils.data.TensorDataset(train_input_dynamic,train_input_static,train_input_profile,train_output_fuel,train_output_NOx, train_input_ids)
    loader = torch.utils.data.DataLoader(dataset = torch_dataset, batch_size = bsize,shuffle = True)

    for i, (batch_x_dynamic,batch_x_static, batch_x_profile,batch_y_fuel, batch_y_NOx,batch_x_ids) in enumerate(loader):
        optimizer.zero_grad()
        output = model(batch_x_dynamic,batch_x_static,batch_x_profile,batch_x_ids)
    
        batch_y = torch.cat((batch_y_fuel, batch_y_NOx), dim=2)
        output_fuel, output_NOx = output.chunk(2, dim = 2)
        loss = criterion(output_fuel, batch_y_fuel)*1000 + criterion(output_NOx, batch_y_NOx)*1000
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()



def evaluate(eval_model, val_input_dynamic,val_input_static, val_input_profile,val_output_fuel, val_output_NOx,val_input_ids):
    eval_model.eval() # Turn on the evaluation mode
    total_loss_fuel = 0.
    total_loss_NOx = 0.
    eval_batch_size = bsize
    torch_dataset = torch.utils.data.TensorDataset(val_input_dynamic,val_input_static,val_input_profile, val_output_fuel, val_output_NOx,val_input_ids)
    loader = torch.utils.data.DataLoader(dataset = torch_dataset, batch_size = eval_batch_size,shuffle = True)

    with torch.no_grad():
        for i, (batch_x_dynamic,batch_x_static,batch_x_profile, batch_y_fuel, batch_y_NOx,batch_x_ids) in enumerate(loader):            
            output = eval_model(batch_x_dynamic,batch_x_static,batch_x_profile,batch_x_ids) 
            batch_y = torch.cat((batch_y_fuel, batch_y_NOx), dim=2)
            
            output_fuel, output_NOx = output.chunk(2, dim = 2)
            total_loss_fuel +=len(batch_x_static) * criterion(output_fuel, batch_y_fuel).item()
            total_loss_NOx +=len(batch_x_static) * criterion(output_NOx, batch_y_NOx).item()
            #print(criterion(output_fuel, batch_y_fuel).item(),criterion(output_NOx, batch_y_NOx).item())
    return total_loss_fuel / len(val_output_fuel),total_loss_NOx / len(val_output_fuel)



##MAIN

filename = r'data.h5'
Dataset = h5py.File(filename,'r')
carnum = 0
#read validation and train
train_input_static = torch.FloatTensor(Dataset['train_input_static'][:2]).to(device)
train_input_dynamic = torch.FloatTensor(Dataset['train_input_dynamic'][:2]).to(device)
train_input_profile = torch.FloatTensor(Dataset['train_input_profile'][:2]).to(device)
train_output_fuel = torch.FloatTensor(Dataset['train_output_fuel'][:2]).to(device)
train_output_NOx = torch.FloatTensor(Dataset['train_output_NOx'][:2]).to(device)
val_input_static = torch.FloatTensor(Dataset['val_input_static']).to(device)
val_input_dynamic =  torch.FloatTensor(Dataset['val_input_dynamic']).to(device)
val_input_profile = torch.FloatTensor(Dataset['val_input_profile']).to(device)
val_output_fuel = torch.FloatTensor(Dataset['val_output_fuel']).to(device)
val_output_NOx = torch.FloatTensor(Dataset['val_output_NOx']).to(device)
val_input_ids = torch.FloatTensor(Dataset['val_input_ids']).to(device)  

ksize = train_input_dynamic.shape[2]
raw_static_size = train_input_static.shape[2]
dim_prepooling = dim_profile_size + dim_static_fuel + dim_fusion 
dim_pred_fuel = dim_profile_size + dim_static_fuel + dim_fusion + dim_pooling
print('dim_pred_fuel',dim_pred_fuel)
dim_pred_NOx = dim_profile_size + dim_static_NOx + dim_fusion + dim_pooling
print('INPUT DONE')

model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.75)


best_val_loss = float("inf")
save_loss = float("inf")

best_model = None
loss11 = []
loss21 = []
loss12 = []
loss22 = []


s = [x for x in range(train_input_dynamic.shape[0])]
random.shuffle(s)
print(Dataset['train_input_static'].shape[0])
#s = s[:epochs]
random_status = False
stl = 0
enl = -1
for epoch in range(1, epochs + 1):
    
    stl = 0
    enl = 10000
    while stl < Dataset['train_input_static'].shape[0]:
   
	    train_input_static = torch.FloatTensor(Dataset['train_input_static'][stl:enl]).to(device)
	    train_input_dynamic = torch.FloatTensor(Dataset['train_input_dynamic'][stl:enl]).to(device)
	    train_input_profile = torch.FloatTensor(Dataset['train_input_profile'][stl:enl]).to(device)
        
	    train_output_fuel = torch.FloatTensor(Dataset['train_output_fuel'][stl:enl]).to(device)
	    train_output_NOx = torch.FloatTensor(Dataset['train_output_NOx'][stl:enl]).to(device)
	    train_input_ids = torch.FloatTensor(Dataset['train_input_ids'][stl:enl]).to(device)

	    train(train_input_dynamic,train_input_static, train_input_profile,train_output_fuel,train_output_NOx,train_input_ids)
	    stl = enl + 1
	    enl = min(enl+10000,Dataset['train_input_static'].shape[0])
     
    train_loss_fuel,train_loss_NOx = evaluate(model,train_input_dynamic,train_input_static,train_input_profile, train_output_fuel,train_output_NOx,train_input_ids)
    val_loss_fuel, val_loss_NOx = evaluate(model,val_input_dynamic,val_input_static,val_input_profile, val_output_fuel, val_output_NOx,val_input_ids)
    #test_loss = evaluate(model, test_input, test_output)
   
    print('-' * 80)
    print('|epoch{:3d}|time:{:5.2f}s|train loss {:5.6f} {:5.6f} |val loss {:5.6f} {:5.6f}'.format(epoch, (time.time() - epoch_start_time),
                                     train_loss_fuel,train_loss_NOx,val_loss_fuel, val_loss_NOx))
    #print('-' * 89)
    loss11.append(train_loss_fuel)
    loss21.append(val_loss_fuel)
    loss12.append(train_loss_NOx)
    loss22.append(val_loss_NOx)
    if val_loss_fuel+val_loss_NOx < best_val_loss:
        best_val_loss = val_loss_fuel+val_loss_NOx
        best_model = model
    if epoch % 20 == 0:
    	#string = 'epoch:' +str(epoch)+'_loss:'+str(val_loss_fuel+val_loss_NOx)[:8]
        k = time.localtime(time.time())
        hour = k.tm_hour
        minute = k.tm_min
        day = k.tm_mday
        month = k.tm_mon
        string = 'epoch:' +str(epoch) + '_'+str(month) + str(day)+' '+ str(hour) + ':' +str(minute)
        savepath = string+'.pt'    
        torch.save(best_model,savepath)

    scheduler.step(val_loss_fuel+val_loss_NOx) 


#save the records
k = time.localtime(time.time())
hour = k.tm_hour
minute = k.tm_min
day = k.tm_mday
month = k.tm_mon
string = indicator + '_'+str(month) + str(day)+' '+ str(hour) + ':' +str(minute)
savepath = r'EffiNet.pt'    
torch.save(best_model,savepath)

