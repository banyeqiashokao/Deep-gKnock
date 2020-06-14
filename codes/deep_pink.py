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
#from keras import optimizers 
from GKlayer import Glayer
from knockoff import knockoff_threshold
from LossHistory import LossHistory 



def deep_pink(y, X , Xk,lasso_init=None,coeff1=0.05,coeff2=0.01,num_epochs=500,batch_size = 50,use_bias=True,fdr=0.2,offset=0,loss='MSE',verbose=False):
    # coeff is the l1 penalty tuning parameter in the dense layer
    #n=X.shape[0]
    p=X.shape[1]
    #num_epochs = 200;
    #batch_size = 500
    #filterNum = 1;
    #bias = True;
    #activation='relu';
    #iterNum = 10;
    
    
    newX=np.concatenate((X,Xk),axis=1) # n by 2p
    
    group=np.arange(1,p+1) # p
    #group_stru=np.array(pd.get_dummies(group))

    group2=np.tile(group,2)  #  2p 
    group_stru2=np.array(pd.get_dummies(group2))
    group_stru2.shape  # 2p by g
    def my_init(shape,dtype):
        return initializers.get(glorot_normal(seed=1))(shape)  
#%%    
    model = Sequential()
    model.add(Glayer(group_stru=group_stru2,input_shape=(2*p,),use_bias=True,name='lay1',kernel_regularizer=regularizers.l1(coeff1),kernel_initializer=Constant(0.1)))  # (2+1)*p
    model.add(Glayer(group_stru=np.eye(p),use_bias=False,name='lay2',kernel_initializer=my_init))  # p
    model.add(Dense(units=p, activation='relu',use_bias=use_bias,kernel_regularizer=regularizers.l1(coeff2),name='lay3',kernel_initializer=my_init))    
    model.add(Dense(units=p, activation='relu',use_bias=use_bias,kernel_regularizer=regularizers.l1(coeff2),name='lay4',kernel_initializer=my_init))    
    if loss=='logit': 
        model.add(Dense(units=1,name='lay5',activation='sigmoid',use_bias=False,kernel_initializer=my_init))    
    else:
        model.add(Dense(units=1,name='lay5',use_bias=False,kernel_initializer=my_init))     
    model.summary()
    history = LossHistory() 
    
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
        tmp2.append(np.zeros((p,)))
        model.layers[0].set_weights(tmp2)
        
    model.fit(newX, y, epochs=num_epochs, batch_size=batch_size,callbacks=[history],verbose=verbose)
    loss=history.losses[-1]
    print('Deep_Pink_loss:'+str(loss)+'\n')
#%%       
    lay1=model.get_layer('lay1')
    tmp=lay1.get_weights()[0]
    z=tmp.reshape(p,2,order='F')  #p by 2


    lay2=model.get_layer('lay2')
    W0=lay2.get_weights()[0]
    W0=W0.reshape(p,1)
    #W0.shape
  
    lay3=model.get_layer('lay3')
    W1=lay3.get_weights()
    W1=W1[0].reshape(p,p)
    #W1.shape


    lay4=model.get_layer('lay4')
    W2=lay4.get_weights()
    W2=W2[0].reshape(p,p)
    #W2.shape

    lay5=model.get_layer('lay5')
    W3=lay5.get_weights()
    W3=W3[0].reshape(p,1)
    #W3.shape
    tmp=np.matmul(W1,W2)  # p by p
    w=np.multiply(W0,np.matmul(tmp,W3))  # g by 1

    Z=np.multiply(z,w)  # p by 2
    ko_stat=np.abs(Z[:,0])-np.abs(Z[:,1])
    #ko_stat=Z[:,0]**2-Z[:,1]**2  # g by 1


    t=knockoff_threshold(ko_stat,fdr=fdr, offset=offset)
    selected = set(np.where(ko_stat >= t)[0]+1)
    selected = np.array(sorted(selected),dtype=int)
    
    
    output = dict()
    output['selected'] = selected
    output['t'] = t
    output['ko_stat'] = ko_stat    
    output['fdr'] = fdr 
    output['offset'] = offset 
 #%%   
    return(output)