# Fast versions of diag(d) %*% X and X %*% diag(d).
`%diag*%` <- function(d, X) d * X
`%*diag%` <- function(X, d) t(t(X) * d)

# get the block digonal part of G with bloack code with group
#G=matrix(1:16,nrow=4)
#group=rep(1:2,each=2)
#group2=1:4
#a=getBlockDiag(G,group)
#blockMatrixDiagonal(a)



getBlockDiag=function(G,group){
  Block_list=lapply(unique(group),function(x) {pos=which(group==x);as.matrix(G[pos,pos])})
  #blockMatrixDiagonal(Block_list)
}



# builds a block matrix whose diagonals are the square matrices provided.
# m1=matrix(runif(10*10),nrow=10,ncol=10)
# m2=matrix(runif(5*5),nrow=5,ncol=5)
# blockMatrix<-blockMatrixDiagonal(m1,m2,m2,m1)
# or
# blockMatrix<-blockMatrixDiagonal(list(m1,m2,m2,m1))
# C.Ladroue

blockMatrixDiagonal<-function(...){  
  matrixList<-list(...)
  if(is.list(matrixList[[1]])) matrixList<-matrixList[[1]]
  
  dimensions<-sapply(matrixList,FUN=function(x) dim(x)[1])
  finalDimension<-sum(dimensions)
  finalMatrix<-matrix(0,nrow=finalDimension,ncol=finalDimension)
  index<-1
  for(k in 1:length(dimensions)){
    finalMatrix[index:(index+dimensions[k]-1),index:(index+dimensions[k]-1)]<-matrixList[[k]]
    index<-index+dimensions[k]
  }
  finalMatrix
}



is_posdef = function(A, tol=1e-9) {
  p = nrow(as.matrix(A))
  
  if (p<500) {
    lambda_min = min(eigen(A)$values)
  }
  else {
    oldw <- getOption("warn")
    options(warn = -1)
    lambda_min = Re(RSpectra::eigs(A, 1, which="SM", opts=list(retvec = FALSE, maxitr=100, tol))$values)
    options(warn = oldw)
    if( length(lambda_min)==0 ) {
      # RSpectra::eigs did not converge. Using eigen instead."
      lambda_min = min(eigen(A)$values)
    }
  }
  return (lambda_min>tol*10)
}