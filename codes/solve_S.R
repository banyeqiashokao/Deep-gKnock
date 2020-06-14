
create.solve_S <- function(Sigma,group_struc) {
  # Check that covariance matrix is symmetric
  stopifnot(isSymmetric(Sigma))
  p = nrow(Sigma)
  tol = 1e-10
  # Convert the covariance matrix to a correlation matrix
  G = cov2cor(Sigma)
  Gblock=getBlockDiag(G,group_struc)
  Gneghalf=lapply(Gblock,function(M) with(eigen(M), vectors %*% (values^(-0.5) * t(vectors)))
  )
  D=blockMatrixDiagonal(Gneghalf)  # 
  # Check that the input matrix is positive-definite
  M=D%*%G%*%D
  lambda_min<-min_eigen(M)
  lambda_min
  gamma = min(2*lambda_min, 1)
  S=gamma*blockMatrixDiagonal(Gblock)  
  # Compensate for numerical errors (feasibility)
  psd = 0;
  gamma_eps = 1e-8;
  while (psd==0) {
    psd = is_posdef(2*G-(1-gamma_eps)*S)
    if (!psd) {
      gamma_eps = gamma_eps*10
    }
  }
  S = S*(1-gamma_eps)
  # Scale back the results for a covariance matrix
  sd=sqrt(diag(Sigma))  # Standard deviation
  return(sd%diag*% S%*diag% sd)
}


min_eigen<-function(G){
	p=nrow(G)
  if (!is_posdef(G)) {
    stop('The covariance matrix is not positive-definite: cannot solve SDP',immediate.=T)
  }
  
  if (p>2) {
    converged=FALSE
    maxitr=10000
    while (!converged) {
      lambda_min = Re(RSpectra::eigs(G, 1, which="SR", opts=list(retvec = FALSE, maxitr=100000, tol=1e-8))$values)
      if (length(lambda_min)==1) {
        converged = TRUE
      } else {
        if (maxitr>1e8) {
          warning('In creation of equi-correlated knockoffs, while computing the smallest eigenvalue of the 
                  covariance matrix. RSpectra::eigs did not converge. Giving up and computing full SVD with built-in R function.',immediate.=T)
          lambda_min = eigen(G, symmetric=T, only.values = T)$values[p]
          converged=TRUE
        } else {
          warning('In creation of equi-correlated knockoffs, while computing the smallest eigenvalue of the 
                  covariance matrix. RSpectra::eigs did not converge. Trying again with increased number of iterations.',immediate.=T)
          maxitr = maxitr*10
        }
      }
      }
    } else {
      lambda_min = eigen(G, symmetric=T, only.values = T)$values[p]
    }
  
  if (lambda_min<0) {
    stop('In creation of equi-correlated knockoffs, while computing the smallest eigenvalue of the 
         covariance matrix. The covariance matrix is not positive-definite.')
  }
  lambda_min
}


