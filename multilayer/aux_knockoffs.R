# AUXILIARY KNOCKOFFS-RELATED FUNCTIONS
library(Matrix)
library(glmnet)
library(gglasso)

# construct equicorrelated fixed-design knockoffs
construct_knockoffs_equi = function(X, groups){
  # problem dimensions
  N = nrow(X)
  n = ncol(X)
  #stopifnot(N >= 2*n) # make sure problem is low-dimensional
  
  # make sure columns of X are normalized to have norm 1
  Sigma = crossprod(X)
  stopifnot(max(abs(diag(Sigma)-1)) <  1e-10)
  
  # equicorrelated construction
  D = matrix(0, n, n)
  S = matrix(0, n, n)
  G = max(groups)
  for(g in 1:G){
    group_idx = which(groups == g)
    group_size = length(group_idx)
    Sigma_grp = Sigma[group_idx, group_idx]
    S[group_idx, group_idx] = Sigma_grp
    eig = eigen(Sigma_grp)
    D[group_idx, group_idx] = eig$vectors%*%diag(1/sqrt(eig$values), 
                                                 group_size, group_size)%*%t(eig$vectors)
  }
  gamma = min(1, 2*min(eigen(D%*%Sigma%*%D, symmetric = TRUE)$values)) - 1e-10
  S = gamma*S
  U_tilde = eigen(crossprod(t(X)))$vectors[,(n+1):(2*n)]
  C = chol(2*S - S%*%solve(Sigma,S))
  X.knockoff = X%*%(diag(n) - solve(Sigma, S)) + U_tilde%*%C
  return(X.knockoff)
}

# get group lasso signed max statistics 
get_LSM_stats = function(X, X.knockoff, y, groups){
  # get problem dimensions
  n = ncol(X)
  G = max(groups)
  
  # make sure that groups are in consecutive order for gglasso
  ord = order(groups)
  X = X[,ord]
  X.knockoff = X.knockoff[,ord]
  groups = groups[ord]
  
  # augment the problem  
  groups_augmented = c(groups, groups + G)
  X_augmented = cbind(X, X.knockoff) 

  # compute lambda_max  
  inner_prods = t(X_augmented)%*%y
  inner_prods_sq = inner_prods*inner_prods
  inner_product_norms = sqrt(sapply(1:(2*G), function(g)(mean(inner_prods_sq[groups_augmented == g]))))
  lambda_max = max(inner_product_norms)/nrow(X)

  # define lambda grid  
  num_lambda = 1000
  lambda_values = seq(lambda_max, 0, length = num_lambda)

  # run group lasso, using glmnet if groups all have size one because it's faster
  if(n > G){
    output = gglasso(X_augmented, y, group = groups_augmented,
                     lambda = lambda_values, intercept=FALSE)
  } else if(n == G){
    output = glmnet(X_augmented, y, lambda = lambda_values, standardize = FALSE, intercept = FALSE)
  }

  # compute first entry times  
  min_idx = sapply(1:(2*n), function(row)(min(c(num_lambda, which(output$beta[row,] != 0)))))
  first_entries = sapply(1:(2*G), function(g)(min(min_idx[groups_augmented == g])))
  max_lambdas = lambda_values[first_entries]
  
  # compute knockoff statistics
  W = sapply(1:G,
             function(g)(max(max_lambdas[g],
                             max_lambdas[g+G])*sign(max_lambdas[g] - max_lambdas[g+G])))
  
  return(W)
}

