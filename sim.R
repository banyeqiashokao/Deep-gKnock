# setup --------
source('sourceDir.R')
sourceDir('codes/')
sourceDir('multilayer/')
library(grpSLOPE)
library(knockoff)

Gpower=matrix(0,1,6)
colnames(Gpower)=c("Deep-gKnock","group-Knockoff ","group-SLOPE","Multilayer","Knockoff","DeepPink" )
Gfdp=Gpower

# parameters--------
n=500
p=500
m=5  # group size
g=p/m  # number of group
k=20
gamma=0.5
rho=0.5 # within grpup
sigma=4
betascale=1.5
#------
Ac=1:(k*m)  # true signal
Ic=(k*m+1):p
Ag=1:k
Ig=(k+1):g
group_struc=rep(1:g,each=m)

SigmaT=matrix(0,p,p)

for(i in 1:p){
  for(j in 1:p){
    if(i==j) SigmaT[i,j]=1
    else if(group_struc[i]==group_struc[j]) SigmaT[i,j]=rho
    else SigmaT[i,j]=gamma*rho
  }
}
set.seed(2019)
beta=c(sample(c(-betascale,betascale),k*m,replace=T),rep(0,p-k*m))

X = matrix(rnorm(n*p),n)%*%chol(SigmaT)
y = (X %*% beta) + sqrt(sigma)*rnorm(n)
Xk<-create_groupko(X,group_struc,shrink=TRUE)
save(file='sim-data.Rdata',beta,y,X,Xk,Z,k,rho,gamma,p,n,g,m,group_struc)

#--------
cat('----Start gknockoff.filter....\n')
gknockoffs=gknockoff.filter(X,Xk,y,group_struc,fdr = 0.2,offset = 1,nlambda=500)
print(gknockoffs$selected)
truepos       <- intersect(gknockoffs$selected, Ag)
n.truepos  <- length(truepos)
n.selected <- length(gknockoffs$selected)
n.falsepos <- n.selected - n.truepos
Gfdp[1,2] <- n.falsepos / max(1, n.selected)
Gpower[1,2] <- n.truepos / length(Ag)
#------
cat('----Start knockoff.filter....\n')
Z = cv_coeffs_glmnet(cbind(X, Xk), y, family='gaussian',generate_lambda = TRUE,nlambda=5)
orig=1:p
W = abs(Z[orig]) - abs(Z[orig+p])
t = knockoff.threshold(W, fdr=0.2, offset=1)
selected = sort(which(W >= t))

gs=unique(group_struc[selected])
truepos <- intersect(gs, Ag)
n.truepos  <- length(truepos)
n.selected <- length(gs)
n.falsepos <- n.selected - n.truepos
Gfdp[1,5] <- n.falsepos / max(1, n.selected)
Gpower[1,5] <- n.truepos / length(Ag)
#------
cat('----Start grpSLOPE....\n')
model <- grpSLOPE(X=X, y=y, group=group_struc, fdr=0.2)
print(model$selected)
truepos       <- intersect(model$selected, Ag)
n.truepos  <- length(truepos)
n.selected <- length(model$selected)
n.falsepos <- n.selected - n.truepos
Gfdp[1,3] <- n.falsepos / max(1, n.selected)
Gpower[1,3] <- n.truepos / length(Ag)

#------
cat('----Start multilayer_knockoff_filter....\n')
M = 1                   # number of layers
groups = matrix(0, p, M)
#groups[,1] = 1:p
groups[,1] = rep(1:g,each=m)
knockoff_type = "fixed_equi"
statistic_type = "group_LSM" 
FDP_hat_type = "kn+"
output = multilayer_knockoff_filter(X, y, groups=groups, q=rep(0.2,M), knockoff_type, statistic_type, FDP_hat_type)
print(multilayer_select<-unique(group_struc[output$S_hat]))
truepos       <- intersect(multilayer_select, Ag)
n.truepos  <- length(truepos)
n.selected <- length(multilayer_select)
n.falsepos <- n.selected - n.truepos
Gfdp[1,4] <- n.falsepos / max(1, n.selected)
Gpower[1,4] <- n.truepos / length(Ag)
#------
cat('Gpower\n')
print(Gpower[1,])
cat('Gfdp\n')
print(Gfdp[1,])
save(file='sim-data.Rdata',beta,y,X,Xk,Z,k,rho,gamma,p,n,g,m,group_struc,Gfdp,Gpower)
