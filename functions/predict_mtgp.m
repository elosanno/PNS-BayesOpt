function [Ypred, Vpred, cov_pred] = predict_mtgp(x_observed,y_observed,N_out,...
    idx_out,idx_in,nx,x_star,gp_par)

% Makes predictions at sample points x_star for all outputs with MTGP model
% Modified from https://github.com/ebonilla/mtgp (Copyright (c) 2009, Edwin
% V. Bonilla)
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
% - x_star : sample points at which predict the outputs 
% - gp_par : GP model infos and hyperparameters 
%
% OUTPUT
% - Ypred : (N_star x N_out) Matrix of MTGP Predicted Means
% - Vpred : (N_star x N_out) Matrix of MTGP Predicted Variances
%           Where M is the number of tasks and N_star is number of sample points  
% - cov_pred : N_star cells of (N_out x N_out) predicted covariance matrices   


% Compute alpha 
[alpha, Kf, L, Kxstar, Kss] = alpha_mtgp(x_observed,y_observed,N_out,idx_out,...
    idx_in,nx,x_star,gp_par);
                
% Predicted mean and var for all outputs at x_star
mu = gp_par.logtheta_all(end);
all_Kxstar = Kxstar(idx_in,:);

N_samples = size(x_star,1);

Ypred = zeros(N_samples,N_out);
Vpred = zeros(N_samples,N_out);

for task = 1 : N_out
    
  Kf_task = Kf(idx_out,task);
  Kstar = repmat(Kf_task,1,N_samples).*all_Kxstar; 
  Ypred(:,task) = mu + Kstar'*alpha;
  v = L\Kstar;
  Vpred(:,task) = Kf(task,task)*Kss - sum(v.*v)';
  
end

% Predicted covariance matrix at x_star
cov_pred = {};

for it = 1:N_samples
    
    Kstar = Kf(idx_out,:).*all_Kxstar(:,it);
    cov_pred{it} = Kf.*Kss(it) - Kstar'*solve_chol(L',Kstar);
end


