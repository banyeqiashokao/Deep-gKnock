# glasso_max_lambda  
# the maximum values of the  regularization parameter \eqn{\lambda} at which the jth variable 
# enter the model
library(tidyverse)
library(grpreg)
gknockoff.filter <- function(X, Xk, y, group_stru,
                             fdr=0.10,
                             offset=0,nlambda=500,max_lambda=TRUE,
                             family='gaussian',loss='ls',
                             pkg='grpreg',standardize=TRUE){
  # Validate input types.
  if (is.data.frame(X)) {
    X.names = names(X)
    X = as.matrix(X, rownames.force = F)  
  } else if (is.matrix(X)) {
    X.names = colnames(X)
  } else {
    stop('Input X must be a numeric matrix or data frame')
  }
  if (!is.numeric(X)) stop('Input X must be a numeric matrix or data frame')
  
  if (!is.factor(y) && !is.numeric(y)) {
    stop('Input y must be either of numeric or factor type')
  }
  if( is.numeric(y) ) y = as.vector(y)
  
  if(offset!=1 && offset!=0) {
    stop('Input offset must be either 0 or 1')
  }
  # Validate input dimensions
  n = nrow(X); p = ncol(X);
  m = length(unique(group_stru)) # number of group
  #group_stru2=c(group_stru,group_stru+m)
  
  # Compute statistics
  if(max_lambda){
    W = gglasso_max_lambda(X,Xk,y,group=group_stru,nlambda=nlambda,family=family,pkg=pkg,loss=loss,standardize=standardize)
  }
  else{
    W = stat.grpreg_coef(X,Xk,y,group=factor(group_stru),nlambda=nlambda,standardize=standardize)
  }
  #orig = 1:m
  #W = Z[orig] - Z[orig+m]
  
  # Run the knockoff filter
  t = knockoff.threshold(W, fdr=fdr, offset=offset)
  selected = sort(which(W >= t))
  
  #if (!is.null(X.names))
  #  names(selected) = X.names[selected]
  
  # Package up the results.
  structure(list(call = match.call(),
                 #X = X,
                 #Xk = Xk,
                 #y = y,
                 group_stru=group_stru,
                 statistic = W,
                 threshold = t,
                 selected = selected),
            class = 'gknockoff.result')
}



stat.grpreg_coef <- function(X,Xk,y,group, nlambda=100, standardize=T, ...) {
  if (!requireNamespace('grpreg', quietly=T))
    stop('grpreg is not installed', call.=F)
  if (!requireNamespace('gglasso', quietly=T))
    stop('gglasso is not installed', call.=F)
  newX=cbind(X,Xk)
  # Standardize the variables
  if( standardize ){
    newX = scale(newX)
  }
  
  n = nrow(X); p = ncol(X)
  if (!methods::hasArg(family) ) family = "gaussian"
  else family = list(...)$family
  group=factor(group)
  group_label=levels(group)  
  m = length(group_label) 
  levs=c(levels(group),paste0('k_',levels(group)))
  group2=factor(c(group,paste0('k_',group)),levels=levs)  # length = 2*p
  #represent_indeices=match(group_label,group)  # the indices of represent variable of each group
  
  # Unless a lambda sequence is provided by the user, generate it
  lambda_max = max(abs(t(newX) %*% y)) / n
  lambda_min = lambda_max / 2e3
  k = (0:(nlambda-1)) / nlambda
  lambda = lambda_max * (lambda_min/lambda_max)^k
  
  
  #fit <- gglasso::gglasso(X, y, group=group,lambda=lambda, intercept=TRUE)
  cv <- grpreg::cv.grpreg(newX,y, group=group2,lambda=lambda, intercept=TRUE)
  fit <- grpreg::grpreg(newX,y, group=group2,lambda=cv$lambda.min, intercept=TRUE)
  coeff1 = abs(fit$beta[-1])  # Z is group_lasso fitted coeffcient  2*p
  dat=data.frame(coeff1,group2=group2)
  tmp=aggregate( coeff1 ~ group2, dat, mean)
  Z=tmp$coeff1
  names(Z)=tmp$group2
  
  orig=1:m
  W = abs(Z[orig]) - abs(Z[orig+m])
  W
}


gglasso_max_lambda<-function(X,Xk,y,group, nlambda=100, standardize=T, loss='ls',pkg='grpreg',...) {
  if (!requireNamespace('gglasso', quietly=T))
    stop('gglasso is not installed', call.=F)
  newX=cbind(X,Xk)
  # Standardize the variables
  if( standardize ){
    print('standardize')
    newX = scale(newX)
  }
  
  n = nrow(X); p = ncol(X)
  if (!methods::hasArg(family) ) family = "gaussian"
  else family = list(...)$family
  #group=factor(group)
  #group_label=levels(group)  
  #m = length(group_label) 
  #group_label2=c(levels(group),paste0('k_',levels(group)))
  #group2=factor(c(group,paste0('k_',group)),levels=group_label2) 
  
  group_label=unique(group)  
  m = length(group_label) 
  group2=  c(group,group+m)
  group_label2=unique(group2)
  represent_indeices=match(group_label2,group2)  # 2*m the indices of represent variable of each group 
  
  # Unless a lambda sequence is provided by the user, generate it
  lambda_max = max(abs(t(newX) %*% y)) / n
  lambda_min = lambda_max / 2e3
  k = (0:(nlambda-1)) / nlambda
  lambda = lambda_max * (lambda_min/lambda_max)^k
  
  
  if(pkg=='gglasso'){
    print(loss)
    fit <- gglasso::gglasso(newX, y, group=group2,lambda=lambda, intercept=TRUE,loss=loss)
    Betas=fit$beta
  }
  else{
    print(family)
    fit <- grpreg::grpreg(newX,y, group=group2,lambda=lambda, intercept=TRUE,family=family)
    Betas=fit$beta[-1,]
  }
  first_nonzero <- function(x) match(TRUE, abs(x) > 0) # NA if all(x==0)
  # length is 2*p, indices of lambda that the variable is firstly selected
  indices <- apply(Betas, 1, first_nonzero)
  table(group2)
  names(indices) <- NULL
  #indices[group2==4]
  gindices = indices[represent_indeices]  # length is the number of groups
  Z=ifelse(is.na(gindices), 0, fit$lambda[gindices] * n)  # 2*m
  orig=1:m
  W = abs(Z[orig]) - abs(Z[orig+m])
  W
}


knockoff.threshold <- function(W, fdr=0.10, offset=0) {
  if(offset!=1 && offset!=0) {
    stop('Input offset must be either 0 or 1')
  }
  ts = sort(c(0, abs(W)))
  ratio = sapply(ts, function(t)
    (offset + sum(W <= -t)) / max(1, sum(W >= t)))
  ok = which(ratio <= fdr)
  ifelse(length(ok) > 0, ts[ok[1]], Inf)
}
