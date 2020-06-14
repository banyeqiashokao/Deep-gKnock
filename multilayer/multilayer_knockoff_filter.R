# MULTILAYER KNOCKOFF FILTER

# (N: number of samples)
# (n: number of variables)
# (M: number of layers)
# X: Nxn design matrix
# y: Nx1 response vector
# groups: nxM matrix, such that groups[j, m] is the group to which variable j belongs at layer m
# q: length-M vector of FDR target levels
# knockoff_type: type of knockoff variables, currently only supports "fixed_equi", for fixed-design equicorrelated statistics
# statistic_type: type of knockoff statistic, currently supports "group_LSM" and "ind_LSM", for group and individual lasso signed max statistics
# FDP_hat_type: "kn" or "kn+" for knockoff or knockoff+ FDP-hat
multilayer_knockoff_filter = function(X, y, groups, q, knockoff_type, statistic_type, FDP_hat_type){
  # problem dimensions
  N = nrow(X) # number of samples
  n = nrow(groups) # number of variables
  M = ncol(groups) # number of layers
  G = apply(groups,2,max) # G[m] = number of groups for at layer m
  
  # check input for correctness
  stopifnot(length(y) == N)
  stopifnot(ncol(X) == n)
  stopifnot(length(q) == M)
  stopifnot(knockoff_type %in% c("fixed_equi"))
  stopifnot(statistic_type %in% c("group_LSM", "ind_LSM"))
  stopifnot(FDP_hat_type %in% c("kn", "kn+"))

  # standardize columns of X
  X = scale(X, center = FALSE)/sqrt(N-1)
    
  # construct knockoffs at each layer
  X.knockoffs = get_knockoff_variables(X, groups, knockoff_type)

  # compute knockoff statistics at each layer
  W = get_knockoff_stats(X, X.knockoffs, y, groups, statistic_type)
  
  # reorder groups based on magnitudes of knockoff statistics
  group_orders = list()               # ordering of groups at each layer
  groups_reordered = matrix(0, n, M)  # reordered group assignments
  for(m in 1:M){
    group_orders[[m]] = order(abs(W[[m]]), decreasing = TRUE)
    groups_reordered[,m] = invPerm(group_orders[[m]])[groups[,m]]
  }
  
  # define one-bit p-values
  P = list()
  allowable_thresh_idx = list()
  for(m in 1:M){
    kn_stats_ordered = W[[m]][group_orders[[m]]]
    signs = sign(kn_stats_ordered)
    allowable_thresh_idx[[m]] = which(abs(kn_stats_ordered[1:(G[m]-1)]) > abs(kn_stats_ordered[2:G[m]]))
    allowable_thresh_idx[[m]] = c(allowable_thresh_idx[[m]], G[m])
    P[[m]] = rep(1, G[m])
    P[[m]][signs == 1] = 0
  }
  
  # run filter to get threshold indices at each layer
  thresh_idx = get_thresholds(P, allowable_thresh_idx, groups_reordered, q, FDP_hat_type)

  # extract knockoff statistic thresholds from threshold indices
  thresh = numeric(M)
  for(m in 1:M){
    if(thresh_idx[m] == 0){
      thresh[m] = Inf
    }
    else{
      thresh[m] = abs(W[[m]][group_orders[[m]][thresh_idx[m]]])
    }
  }

  # get selection set
  S_hat = get_S_hat(P, groups_reordered, thresh_idx)
  
  # return output: selection set, threshold indices, and thresholds
  output=list()
  output$S_hat = S_hat
  output$thresh_idx = thresh_idx
  output$thresh = thresh
  return(output)
}