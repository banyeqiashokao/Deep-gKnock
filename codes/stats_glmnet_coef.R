stat.glmnet_coef <- function(X, Xk, y, family='gaussian',generate_lambda=TRUE) {
  # Randomly swap columns of X and Xk
  #swap = rbinom(ncol(X),1,0.5)
  #swap.M = matrix(swap,nrow=nrow(X),ncol=length(swap),byrow=TRUE)
  #X.swap  = X * (1-swap.M) + Xk * swap.M
  #Xk.swap = X * swap.M + Xk * (1-swap.M)
  
  parallel=FALSE
  # Compute statistics
  Z = cv_coeffs_glmnet(cbind(X, Xk), y, family=family, parallel=parallel,generate_lambda=generate_lambda)
  p = ncol(X)
  orig = 1:p
  Z=abs(Z)
  
  
  W = pmax(Z[orig], Z[orig + p])
  chi = sign(Z[orig] - Z[orig + p])#*(1-2*swap)
  
  return(list(W=W,chi=chi))
  #W = abs(Z[orig]) - abs(Z[orig+p])
  
}


cv_coeffs_glmnet <- function(X, y, nlambda=500, intercept=T, parallel=T, generate_lambda,...) {
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
  
  print(family)
  cv.glmnet.fit <- glmnet::cv.glmnet(X, y, lambda=lambda, intercept=intercept,
                                     standardize=F,standardize.response=F, parallel=parallel, ...)
  
  coef(cv.glmnet.fit, s = "lambda.min")[2:(p+1)]
}