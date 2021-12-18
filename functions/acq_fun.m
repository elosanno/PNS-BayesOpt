function scores = acq_fun(x_observed,y_observed,x_samples,gp_par,Target,k_ucb)

% Implement UCB-like acquisition functions suited to different multi-output
% GP models
% 
% INPUT:
% - x_observed : queried inputs 
% - y_observed : observed outputs 
% - x_samples : sample points at which compute the scores  
% - gp_par : GP model infos and hyperparameters
% - Target : muscles to maximize or minimize and relative weights 
% - k_ucb : parameter to balance exploration and exploitation
%
% OUTPUT:
% - scores : scores (mean_obj + k_ucb * std_obj) corresponding to each of 
% the x_samples


%% Predict mean and var of the output of X_samples from the GP
% Y_mu = predicted means [n_samples x n_outputs]
% Y_s2 = predicted variances [n_samples x n_outputs]
% cov_pred = predicted covariances [n_samples cells of n_outputs x n_outputs]

if strcmp(gp_par.model,'igp')
    
    % Perform prediction
    [Y_mu, Y_s2] = igp(gp_par.logtheta_all, gp_par.covfunc_x, x_observed, y_observed, x_samples);
    
elseif strcmp(gp_par.model,'mtgp')
    
    % Reshape the observations y in a single column + take the corresponding
    % indexes of inputs (idx_in) and outputs (idx_out)
    y_observed_res = reshape(y_observed,[],1);
    idx_out = [];
    idx_in = [];
    for iO = 1:size(y_observed,2)
        idx_out((iO-1)*size(y_observed,1)+1:iO*size(y_observed,1),1) = iO;
        idx_in((iO-1)*size(y_observed,1)+1:iO*size(y_observed,1),1) = 1:size(y_observed,1);
    end
    
    nx = ones(length(y_observed_res),1); % number of times each element of y has been observed (usually nx(i)=1)
    
    % Perform prediction
    [Y_mu, Y_s2, cov_pred] = predict_mtgp(x_observed,y_observed_res,size(y_observed,2),idx_out,idx_in,nx,x_samples,gp_par);
    
elseif strcmp(gp_par.model,'slfm')
    
    % Reshape the observations y in a single column + take the corresponding
    % indexes of inputs (idx_in) and outputs (idx_out)
    y_observed_res = reshape(y_observed,[],1);
    idx_out = [];
    idx_in = [];
    for iO = 1:size(y_observed,2)
        idx_out((iO-1)*size(y_observed,1)+1:iO*size(y_observed,1),1) = iO;
        idx_in((iO-1)*size(y_observed,1)+1:iO*size(y_observed,1),1) = 1:size(y_observed,1);
    end
    
    nx = ones(length(y_observed_res),1); % number of times each element of y has been observed (usually nx(i)=1)
    
    % Perform prediction
    [Y_mu, Y_s2, cov_pred] = predict_slfm(x_observed,y_observed_res,size(y_observed,2),idx_out,idx_in,nx,x_samples,gp_par);

end


%% Compute the scores
% Exploitation term
mean_obj = Y_mu*Target.Weights';

% Exploration term
if strcmp(gp_par.model,'igp')
    
    std_obj = sqrt(Y_s2*(Target.Weights.^2)');
    
elseif strcmp(gp_par.model,'mtgp') || strcmp(gp_par.model,'slfm')
    
    std_obj = [];
    
    for ix = 1:length(cov_pred)
        
        cov_pred_weight = cov_pred{ix} .* (Target.Weights'*Target.Weights);
        std_obj(ix,1) = sqrt(sum(sum(cov_pred_weight)));
    end
    
end

% Scores
scores = mean_obj + k_ucb * std_obj;