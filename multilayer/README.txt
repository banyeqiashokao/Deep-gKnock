Multilayer Knockoff Filter Software (beta)

Author: Eugene Katsevich (email: ekatsevi@stanford.edu)
Date: 2017
Reference: Katsevich and Sabatti (2017), https://arxiv.org/abs/1706.09375
Dependencies: Matrix, glmnet, gglasso

Comments: This code currently supports only the functionalities described in Katsevich and Sabatti (2017). It will soon be expanded into an R package that includes more options for knockoff and statistic construction, including model-X knockoffs. 

Getting started: See mkf_example.R for a simple demonstration of how to use this code.

Descriptions of functions:

- multilayer_knockoff_filter.R: Main function for carrying out the multilayer knockoff filter. It takes as arguments the design matrix, response vector, grouping information for variables, target FDR levels, and options for knockoff construction, knockoff statistic type, and FDP-hat type. See comment at the top of this function for input format instructions. 

- aux_mkf.R: Auxiliary functions for multilayer knockoff filter, such as computing FDP-hat or finding the thresholds.

- aux_knockoffs.R: Basic knockoffs-related functions, like constructing knockoffs or computing group lasso signed-max statistics.

- mkf_example.R: Simple demo of multilayer knockoff filter.
