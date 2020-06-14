create_groupko<-function(X,group_struc,shrink=FALSE){
  
  # Estimate the mean vectorand covariance matrix
  mu = colMeans(X)
  
  # Estimate the covariance matrix
  if(!shrink) {
    Sigma = cov(X)
    # Verify that the covariance matrix is positive-definite
    if(!is_posdef(Sigma)) {
      shrink=TRUE
    }
  }
  if(shrink) {
    if (!requireNamespace('corpcor', quietly=T))
      stop('corpcor is not installed', call.=F)
    Sigma = tryCatch({suppressWarnings(matrix(as.numeric(corpcor::cov.shrink(X,verbose=F)), nrow=ncol(X)))},
                     warning = function(w){}, error = function(e) {
                       stop("SVD failed in the shrinkage estimation of the covariance matrix. Try upgrading R to version >= 3.3.0")
                     }, finally = {})
  }
  
  
  S=create.solve_S(Sigma,group_struc)
  
  SigmaInvS = solve(Sigma,S)
  mu_k = X - sweep(X,2,mu,"-") %*% SigmaInvS
  Sigma_k = 2*S - S %*% SigmaInvS
  X_k = mu_k + matrix(rnorm(ncol(X)*nrow(X)),nrow(X)) %*% chol(Sigma_k)
  
  
}