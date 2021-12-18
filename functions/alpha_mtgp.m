function [alpha, Kf, L, Kxstar, Kss] = alpha_mtgp(x_observed,y_observed,N_out,...
    idx_out,idx_in,nx,x_star,gp_par)

% Predictions in multi-output MTGP model
% Modified from https://github.com/ebonilla/mtgp (Copyright (c) 2009, Edwin
% V. Bonilla)
%
% INPUT:
% - x_observed : queried inputs 
% - y_observed : observed outputs 
% - N_out : number of outputs              
% - idx_in : Vector containing the indexes of the data-points in x to
%           which each observation y corresponds
% - idx_out : Vector containing the indexes of the output to which
%            each observation y corresponds
% - nx : number of times each element of y has been observed 
%       usually nx(i)=1 unless the corresponding y is an average
% - x_star : sample points at which predict the outputs 
% - gp_par : GP model infos and hyperparameters
%
% OUTPUT:
% - alpha : The solution to the ( (Kf x Kx) + (Sigma x I) )^{-1} y
% - Kf : The Covariance matrix for the outputs
% - L : The cholesky factorization of  C = (Kf x Kx) + Sigma
% - Kxstar : sample-observed points input covariances
% - Kss : sample-sample points variances


% *** General settings here ****
MIN_NOISE = 0;
% ******************************

covfunc_x = gp_par.covfunc_x;
logtheta_all = gp_par.logtheta_all;
irank = gp_par.irank;

if ischar(covfunc_x), covfunc_x = cellstr(covfunc_x); end % convert to cell if needed

D = size(x_observed,2);     % Dimensionality used when covfunc_x is called
n = length(y_observed);     % Total number of output observations

% Output covariance Kf
nlf = irank*(2*N_out - irank +1)/2;     % number of parameters for Lf
theta_lf = logtheta_all(1:nlf);     % parameters for Lf
Lf = vec2lowtri_inchol(theta_lf,N_out,irank);
Kf = Lf*Lf';

% Input covariance Kx for observed data 
n_theta_x = eval(feval(covfunc_x{:}));    % number of parameters of Kx
theta_x = logtheta_all(nlf+1:nlf+n_theta_x);    % parameters of Kx
Kx = feval(covfunc_x{:}, theta_x, x_observed); 

% Noise matrix
sigma2n = exp(2*logtheta_all(nlf+n_theta_x+1:end-1));  % Noise parameters
Sigma2n = diag(sigma2n);    % Noise Matrix
Var_nx = diag(1./nx);

% Multi-output covariance for observed data
K = Kf(idx_out,idx_out).*Kx(idx_in,idx_in);                       
K = K + (Sigma2n(idx_out,idx_out).*Var_nx);
Sigma_noise = MIN_NOISE*eye(n);
K = K + Sigma_noise;

% Alpha 
mu = logtheta_all(end);     % constant mean function

L = chol(K)';   % cholesky factorization of the covariance
alpha = solve_chol(L',y_observed-mu);

% Sample-Observed points input covariances
[Kss, Kxstar] = feval(covfunc_x{:}, theta_x, x_observed, x_star);          
