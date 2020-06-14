#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:38:56 2019

@author: guangyu
"""
import numpy as np
#from scipy.stats import nbinom
#from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
numpy2ri.activate()

def cv_coeffs_glmnet(y,X,Xk,nlambda=500):
    p=X.shape[1]
    newX=np.concatenate((X,Xk),axis=1) # n by 2p
    rstring="""
    cv_coeffs_glmnet <- function(X, y, nlambda=500, intercept=T, parallel=T, generate_lambda=T,...) {
      # Standardize variables
      X = scale(X)
  
      n = nrow(X); p = ncol(X)
  
      if (!methods::hasArg(family) ) family = "gaussian"
      else family = list(...)$family
  
      if (generate_lambda) {
        # Unless a lambda sequence is provided by the user, generate it
        print('generate lambda')
        lambda_max = max(abs(t(X) %*% y)) / n
        lambda_min = lambda_max / 2e3
        k = (0:(nlambda-1)) / nlambda
        lambda = lambda_max * (lambda_min/lambda_max)^k
      }
      else {
        lambda = NULL
      }
  
      cv.glmnet.fit <- glmnet::cv.glmnet(X, y, lambda=lambda, intercept=intercept,
                                         standardize=F,standardize.response=F, parallel=parallel, ...)
  
      coef(cv.glmnet.fit, s = "lambda.min")[2:(p+1)]
    }
    """
    cv_coeffs_glmnet=robjects.r(rstring)
    result=np.array(cv_coeffs_glmnet(newX,y,nlambda=nlambda))
    return result
    
