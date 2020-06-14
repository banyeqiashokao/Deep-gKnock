#import time
#import math
#import sys
#import os

import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_normal, Constant
from keras import regularizers
from keras import initializers
from keras import optimizers as optimizers
from GKlayer import Glayer
from knockoff import knockoff_threshold
from LossHistory import LossHistory 



def deep_gKnock(y, X , Xk, group, lasso_init=None,coeff1=0.05,coeff2=0.05,num_epochs=500,batch_size = 50,use_bias=True,fdr=0.2,offset=0,num_layer=2,loss='MSE',verbose=False):
    # coeff is the l1 penalty tuning parameter in the dense layer
    #n=X.shape[0]
    p=X.shape[1]
    #num_epochs = 200;
        #filterNum = 1;
    #bias = True;
    #activation='relu';
    #iterNum = 10;
    group_label=np.unique(group)
    g=len(group_label)
    
    newX=np.concatenate((X,Xk),axis=1) # n by 2p
    
    group_stru=np.array(pd.get_dummies(group))
    group_size=np.sum(group_stru,0).reshape(g)
    group2=np.tile(group,2)  #  2p 
    group_stru2=np.array(pd.get_dummies(group2))
    group_stru2.shape  # 2p by g
    def my_init(shape,dtype):
        return initializers.get(glorot_normal(seed=1))(shape) 
    use_bias=True 
#%%    
    model = Sequential()
    model.add(Glayer(group_stru=group_stru2,input_shape=(2*p,),use_bias=True,kernel_regularizer=regularizers.l1(coeff1),name='lay1',kernel_initializer=my_init))
    model.add(Glayer(group_stru=np.eye(g),kernel_regularizer=regularizers.l1(coeff2),use_bias=False,name='lay2',kernel_initializer=my_init))
    # begin dense layer
    model.add(Dense(units=g, activation='relu',use_bias=use_bias,kernel_regularizer=regularizers.l1(coeff2),name='lay31',kernel_initializer=my_init))    
    if num_layer>=2:
        model.add(Dense(units=g, activation='relu',use_bias=use_bias,kernel_regularizer=regularizers.l1(coeff2),name='lay32',kernel_initializer=my_init))    
    if num_layer>=3:
        model.add(Dense(units=g, activation='relu',use_bias=use_bias,kernel_regularizer=regularizers.l1(coeff2),name='lay33',kernel_initializer=my_init))    
    if num_layer>=4:
        model.add(Dense(units=g, activation='relu',use_bias=use_bias,kernel_regularizer=regularizers.l1(coeff2),name='lay34',kernel_initializer=my_init))    
    
    if loss=='logit': 
        model.add(Dense(units=1,name='lay4',activation='sigmoid',use_bias=False,kernel_initializer=my_init))    
    else:
        model.add(Dense(units=1,name='lay4',use_bias=False,kernel_initializer=my_init))    
    
    history = LossHistory() 
    model.summary() 
    #adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #model.compile(loss='mean_squared_error', optimizer=adam)
    if loss=='logit':
        print('loss = binary_crossentropy')
        model.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        print('loss = mean_squared_error')
        model.compile(loss='mean_squared_error', optimizer='adam')
 
    
    if not lasso_init is None:
        print('set lasso_init\n')
        tmp=lasso_init.reshape(p*2,1,order='F')
        tmp2=list()      
        tmp2.append(tmp)
        #tmp2.append(np.zeros((p,)))
        model.layers[0].set_weights(tmp2)
        
    model.fit(newX, y, epochs=num_epochs, batch_size=batch_size,callbacks=[history],verbose=verbose)
    loss=history.losses[-1]
    print('Deep_gKnock_loss:'+str(loss)+'\n')
#%%       
    lay1=model.get_layer('lay1')
    tmp=lay1.get_weights()[0]
    tmp=tmp.reshape(p,2,order='F')
    #z=np.dot(group_stru.T,np.abs(tmp)) #g by 2
    #z=np.dot(group_stru.T,(tmp)**2) #g by 2
    groupsize2=np.tile(group_size,2).reshape(g,2)   #g by 2
    z=np.dot(group_stru.T,np.abs(tmp))/groupsize2 #g by 2
    
    lay2=model.get_layer('lay2')
    W0=lay2.get_weights()[0]
    W0=W0.reshape(g,1)
    W0.shape
    W0    # g by 1
    
    
    lay31=model.get_layer('lay31')
    W31=lay31.get_weights()
    W31=W31[0].reshape(g,g)    
    tmp=W31
    
    if num_layer>=2:
        lay32=model.get_layer('lay32')
        W32=lay32.get_weights()
        W32=W32[0].reshape(g,g)
        tmp=np.matmul(tmp,W32)  # g by g
    
    if num_layer>=3:
        lay33=model.get_layer('lay33')
        W33=lay33.get_weights()
        W33=W33[0].reshape(g,g)
        tmp=np.matmul(tmp,W33)  # g by g
    if num_layer>=4:
        lay34=model.get_layer('lay34')
        W34=lay34.get_weights()
        W34=W34[0].reshape(g,g)
        tmp=np.matmul(tmp,W34)  # g by g
        
    lay4=model.get_layer('lay4')
    W4=lay4.get_weights()
    W4=W4[0].reshape(g,1)
    W4.shape   
    
    
    
    w=np.multiply(W0,np.matmul(tmp,W4))  # g by 1
    
    Z=np.multiply(z,w)  # g by 2
    ko_stat=np.abs(Z[:,0])-np.abs(Z[:,1])
    #ko_stat=Z[:,0]**2-Z[:,1]**2  # g by 1

    t=knockoff_threshold(ko_stat,fdr=fdr, offset=offset)    
    selected=np.sort(np.where(ko_stat >= t)[0]+1)
    selected = set(selected)
            
            
    output = dict()
    output['selected'] = selected
    output['t'] = t
    output['ko_stat'] = ko_stat    
    output['fdr'] = fdr 
    output['offset'] = offset 
 #%%   
    return(output)