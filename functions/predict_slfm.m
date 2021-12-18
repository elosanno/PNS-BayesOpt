function [Ypred, Vpred, cov_pred] = predict_slfm(x_observed,y_observed,N_out,...
    idx_out,idx_in,nx,x_star,gp_par)

% Makes predictions at sample points x_star for all outputs with SLFM model
% 
% INPUT:
% - x_observed : queried inputs 
% - y_observed : observed outputs 
% - N_out : number of outputs 
% - idx_out : Vector containing the indexes of the output to which
%            each observation y corresponds
% - idx_in : Vector containing the indexes of the x data-points to
%           which each observation y corresponds
% - nx : number of times each element of y has been observed 
%       usually nx(i)=1 unless the corresponding y is an average
% - logtheta_all : all hyperparameters
% - x_star : sample points at which predict the outputs 
% - gp_par : GP model infos and hyperparameters 
%
% OUTPUT
% - Ypred : (N_star x N_out) Matrix of MTGP Predicted Means
% - Vpred : (N_star x N_out) Matrix of MTGP Predicted Variances
%           Where M is the number of tasks and N_star is number of sample points  
% - cov_pred : N_star cells of (N_out x N_out) predicted covariance matrices 


% Compute alpha
[alpha, Kf, L, Kxstar, Kss] = alpha_slfm(x_observed,y_observed,N_out,idx_out,...
    idx_in,nx,x_star,gp_par);

% Predicted mean and var for all outputs at x_star, and predicted covariance at x_star
mu = gp_par.logtheta_all(end);

N_star = size(x_star,1);

Ypred = zeros(N_star,N_out);
Vpred = zeros(N_star,N_out);
cov_pred = {};

for it = 1:N_star
    
    Ypred(it,:) = mu + (Kxstar{it}'*alpha)';
    cov_pred{it} = Kss{it} - Kxstar{it}'*solve_chol(L',Kxstar{it});
    Vpred(it,:) = diag(cov_pred{it});
end

