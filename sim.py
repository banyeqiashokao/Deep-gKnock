#%%
import os,sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
wd = ''   # change to the path of your folder
os.chdir(wd)
sys.path.append(wd)
sys.path.append(wd+'/codes') # add codes to seach path

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from numpy.random import seed
from tensorflow import set_random_seed
from deep_pink import deep_pink
from deep_gKnock import deep_gKnock
import pyreadr
#%% read data
data = pyreadr.read_r(wd+'/sim-data.Rdata')
p=int(data['p']['p'][0])
n=int(data['n']['n'][0])
m=int(data['m']['m'][0])
g=int(data['g']['g'][0])
X= np.array(data["X"]).reshape(n,p,order='F')
Xk= np.array(data["Xk"]).reshape(n,p,order='F')
newX=np.concatenate((X,Xk),axis=1) # n by 2p
y= np.array(data["y"])
group=np.array(data["group_struc"]).reshape((p,))
group 

group_stru=np.array(pd.get_dummies(group))
group_size=np.sum(group_stru,0).reshape(g)
group_size
group2=np.tile(group,2)  #  2p 
group_stru2=np.array(pd.get_dummies(group2))
group_stru2.shape  # 2p by g
#%%
seed(1)
set_random_seed(1)
PINK=deep_pink(y, X , Xk,num_epochs=500,batch_size=200,verbose=True,coeff1=0.05,coeff2=0.01,fdr=0.2)
ko_stat=PINK['ko_stat']
S_PINK=PINK['selected']
S_PINK
#%%
seed(1)
set_random_seed(1)
gknock=deep_gKnock(y, X, Xk, group,num_epochs=500,batch_size=200,verbose=True,coeff1=0.1,coeff2=0.1,fdr=0.2)
S_gknock=gknock['selected']
S_gknock
