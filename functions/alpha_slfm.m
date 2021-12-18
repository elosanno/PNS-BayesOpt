function [alpha, K, L, Kstar, Kss] = alpha_slfm(x_observed,y_observed,N_out,...
    idx_out,idx_in,nx,x_star,gp_par)

% Predictions in multi-output SLFM model
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
% - Kxstar : sample-observed data input covariances
% - Kss : sample-sample data variances


% *** General settings here ****
MIN_NOISE = 0;
% ******************************

covfunc_x = gp_par.covfunc_x;
logtheta_all = gp_par.logtheta_all;
Q = gp_par.Q;

if ischar(covfunc_x), covfunc_x = cellstr(covfunc_x); end % convert to cell if needed

D = size(x_observed,2);     % Dimensionality used when covfunc_x is called
n = length(y_observed);     % Total number of output observations

% Initializations 
K = zeros(n,n);

N_star = size(x_star,1);

Kstar = {};
Kss = {};

for it = 1:N_star
    Kss{it} = zeros(N_out,N_out);
    Kstar{it} = zeros(n,N_out);
end

% Compute multi-output covariance for observed data & Sample-Observed points input covariances
cc_x = 0;
cc_lf = 0;

for iq = 1:Q    % for each latent function 
    
    theta_lf = logtheta_all(cc_lf+1:cc_lf+N_out);
    cc_lf = cc_lf + N_out;
    Kf = theta_lf*theta_lf';    % Covariance matrix for the outputs 
    
    ltheta_x = eval(feval(covfunc_x{iq}{:}));
    theta_x = logtheta_all(N_out*Q+cc_x+1:N_out*Q+cc_x+ltheta_x);
    cc_x = cc_x + ltheta_x;
    Kx_q = feval(covfunc_x{iq}{:}, theta_x, x_observed);    % Input covariance for observed data 
    
    K = K + Kf(idx_out,idx_out).*Kx_q(idx_in,idx_in);   % Multi-output covariance for observed data 
    
    [Kss_q, Kxstar_q] = feval(covfunc_x{iq}{:}, theta_x, x_observed, x_star);   % sample-sample and sample-observed input covariances
    
    for it = 1:N_star % sample-sample and sample-observed multi-output covariances
        
        Kstar{it} = Kstar{it} + Kf(idx_out,:).*Kxstar_q(idx_in,it);
        
        Kss{it} = Kss{it} + Kf.*Kss_q(it);
    end
end

% Noise matrix 
sigma2n = exp(2*logtheta_all(cc_lf+cc_x+1:end-1));  % Noise parameters 
Sigma2n = diag(sigma2n);    % Noise Matrix
Var_nx = diag(1./nx);

K = K + (Sigma2n(idx_out,idx_out) .* Var_nx);
Sigma_noise = MIN_NOISE*eye(n);
K = K + Sigma_noise;

% Alpha 
mu = logtheta_all(end);     % constant mean function

L = chol(K)';	% cholesky factorization of the covariance
alpha = solve_chol(L',y_observed-mu);
