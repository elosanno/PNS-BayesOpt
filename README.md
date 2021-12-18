# PNS-BayesOpt

Gaussian Process based Bayesian Optimization (GP-BO) of peripheral nerve stimulation (PNS). 

This repository contains the code used in the paper: https://doi.org/10.1088/1741-2552/ac3f6c. PNS protocols are optimized through GP-BO to evoke target movements. 

Requires the GPML toolbox version 1.3 by Carl Edward Rasmussen and Hannes Nickisch (http://gaussianprocess.org/gpml/code/matlab/release/gpml-matlab-v1.3-2006-09-08.zip). 

Three multi-output GP models are implemented: 
- IGP: multiple independent GPs with shared hyperparameters. 
- MTGP: multi-task GP model (code modified from the implementation by Edwin V. Bonilla available at https://github.com/ebonilla/mtgp). 
- SLFM: semiparametric latent factor model. 

Upper Confidence Bound (UCB) like acquisition functions are implemented.
