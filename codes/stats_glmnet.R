
lasso_max_lambda_lars <- function(X, y, ...) {
  if (!requireNamespace('lars', quietly=T))
    stop('lars is not installed', call.=F)
  
  fit <- lars::lars(X, y, normalize=T, intercept=F, ...)
  lambda <- rep(0, ncol(X))
  for (j in 1:ncol(X)) {
    entry <- fit$entry[j]
    if (entry > 0) lambda[j] <- fit$lambda[entry]
  }
  return(lambda)
}


lasso_max_lambda_glmnet <- function(X, y, nlambda=500, intercept=T, standardize=T, ...) {
  if (!requireNamespace('glmnet', quietly=T))
    stop('glmnet is not installed', call.=F)
  
  # Standardize the variables
  if( standardize ){
    X = scale(X)
  }
    
  n = nrow(X); p = ncol(X)
  if (!methods::hasArg(family) ) family = "gaussian"
  else family = list(...)$family
  
  if (!methods::hasArg(lambda) ) {
    if( identical(family, "gaussian") ) {
      if(!is.numeric(y)) {
        stop('Input y must be numeric.')
      }
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
  }

  fit <- glmnet::glmnet(X, y, lambda=lambda, intercept=intercept, 
                        standardize=F, standardize.response=F, ...)
  
  first_nonzero <- function(x) match(T, abs(x) > 0) # NA if all(x==0)
  indices <- apply(fit$beta, 1, first_nonzero)
  names(indices) <- NULL
  ifelse(is.na(indices), 0, fit$lambda[indices] * n)
}


lasso_max_lambda <- function(X, y, method=c('glmnet','lars'), ...) {
  switch(match.arg(method), 
         glmnet = lasso_max_lambda_glmnet(X,y,...),
         lars = lasso_max_lambda_lars(X,y,...)
         )
}