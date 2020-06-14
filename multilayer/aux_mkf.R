# AUXILIARY FUNCTIONS FOR MULTILAYER KNOCKOFF FILTER

# compute knockoff variables at each layer
get_knockoff_variables = function(X, groups, knockoff_type){
  switch(knockoff_type,
     fixed_equi = {
       X.knockoffs = list()
       for(m in 1:M){
         cat(sprintf("Constructing knockoff variables for layer %d...\n", m))
		 # The next line is the original one by the author, but it doe's not applicable to high dimensional case so I replace it with my own
         #X.knockoffs[[m]] = construct_knockoffs_equi(X, groups[,m]) 
		 X.knockoffs[[m]] = create_groupko(X, groups[,m])
       }
     }       
  )
  return(X.knockoffs)  
}

# compute knockoff statistics at each layer
get_knockoff_stats = function(X, X.knockoffs, y, groups, statistic_type){
  #source("aux_knockoffs.R")
  
  # problem dimensions
  n = nrow(groups)
  M = ncol(groups)
  G = apply(groups,2,max) 
  
  # initialize
  W = list()
  
  # compute W layer by layer
  for(m in 1:M){
    cat(sprintf("Computing knockoff statistics at layer %d...\n", m))
    # get knockoffs at this layer
    X.knockoff = X.knockoffs[[m]]
    # switch based on statistic type
    switch(statistic_type,
           # group lasso signed max (Fisher-like)
           group_LSM = {
             reg_groups = groups[,m]
             W[[m]] = get_LSM_stats(X, X.knockoff, y, reg_groups)
           },
           
           # individual lasso signed max (Simes-like)
           ind_LSM = {
             reg_groups = 1:n
             W_ind = get_LSM_stats(X, X.knockoff, y, reg_groups)
             FUN = function(g){
               if(max(W_ind[groups[,m] == g]) + min(W_ind[groups[,m] == g]) == 0){
                 return(0)
               } else{
                 return(W_ind[groups[,m] == g][which.max(abs(W_ind[groups[,m] == g]))])
               }
             }
             W[[m]] = sapply(1:G[m], FUN)
           }
    )
  }
  return(W)
}

# find multilayer knockoff filter thresholds at each layer
get_thresholds = function(P, allowable_thresh_idx, groups, q, FDP_hat_type){
  cat(sprintf("Searching for thresholds...\n"))
  n = nrow(groups)
  M = ncol(groups)
  
  G = apply(groups,2,max) # G[m] = # groups, for grouping m
  
  # initialize thresholds
  thresh_idx = G
  
  # find thresholds
  done = FALSE
  while(!done){
    thresh_idx_old = thresh_idx
    for(m in 1:M){
      if(thresh_idx[m] >= 1){
        thresh_idx_m = 0
        for(thresh_idx_m_tmp in thresh_idx[m]:1){
          if(thresh_idx_m_tmp %in% allowable_thresh_idx[[m]]){
            thresh_idx_tmp = thresh_idx
            thresh_idx_tmp[m] = thresh_idx_m_tmp
            FDP_hat = get_FDP_hat(P, groups, thresh_idx_tmp, FDP_hat_type)
            if(FDP_hat[m] <= q[m]){
              thresh_idx_m = thresh_idx_m_tmp
              break
            }
          }
        }
        thresh_idx[m] = thresh_idx_m
        if(thresh_idx[m] == 0){
          thresh_idx = rep(0, M)
          done = TRUE
          break
        }
      }
    }
    if(all(thresh_idx_old==thresh_idx)){done = TRUE}
  }
  
  return(thresh_idx)  
}

# get estimated number of false discoveries at each layer for given thresholds
get_V_hats = function(P, thresh_idx, FDP_hat_type){
  M = length(thresh_idx)
  
  V_hats = numeric(M)
  switch(FDP_hat_type,
     "kn" = {
       V_hats[thresh_idx == 0] = 0
       for(m in which(thresh_idx > 0)){
         V_hats[m] = sum(P[[m]][1:thresh_idx[m]] == 1)
       }    
     },
     
     "kn+" = {
       V_hats[thresh_idx == 0] = 1
       for(m in which(thresh_idx > 0)){
         V_hats[m] = 1 + sum(P[[m]][1:thresh_idx[m]] == 1)
       }    
     }
  )
  return(V_hats)
}

# get selection set for given thresholds
get_S_hat = function(P, groups, thresh_idx){
  n = nrow(groups)
  M = ncol(groups)
  if(any(thresh_idx == 0)){
    S_hat = numeric(0)    
  }
  else{
    S_hat = 1:n # current selection set
    for(m in 1:M){
      S_tilde_m = which(is.element(groups[,m],intersect(which(P[[m]] == 0), 1:thresh_idx[m])))
      S_hat = intersect(S_hat, S_tilde_m)
    }
  }
  return(S_hat)
}

# get FDP-hat for given thresholds 
get_FDP_hat = function(P, groups, thresh_idx, FDP_hat_type){
  n = nrow(groups)
  M = ncol(groups)
  
  S_hat = get_S_hat(P, groups, thresh_idx)
  
  S_hats_m = sapply(1:M, function(m)(length(unique(groups[S_hat,m]))))    
  V_hats_m = get_V_hats(P, thresh_idx, FDP_hat_type)
  
  FDP_hat = V_hats_m/(pmax(1, S_hats_m))
  
  return(FDP_hat)
}