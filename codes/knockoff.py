#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 08:58:32 2019

@author: guangyu
"""

import numpy as np
import pandas as pd

def knockoff_threshold(W, fdr=0.10, offset=1):
    if offset !=1 and offset !=0:
        raise ValueError('Input offset must be either 0 or 1')
    W=np.array(W)
    ts=np.sort(np.insert(abs(W),0,0))
    ratio=pd.Series(ts).apply(lambda t: (offset + sum(W <= -t)) / max(1, sum(W >= t)) ) 
    ratio=np.array(ratio)
    ok=np.where(ratio <= fdr)[0]
    if len(ok) > 0:
        out=ts[ok[0]]
    else:
        out=float("inf")  
    return out
    
