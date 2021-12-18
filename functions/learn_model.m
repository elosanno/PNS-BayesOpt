function gp_par = learn_model(gp_par,x_train,y_train)

% Optimize GP model hyperparameters using minimize.m of GPML toolbox 
% 
% INPUT
% - gp_par : GP model infos and initial hyperparameter values
% - x_train : training input data 
% - y_train : training output data
%
% OUTPUT
% - gp_par : GP model infos and optimized hyperparameter values


% Initial value of hyperparameters to optimize
logtheta0 = gp_par.logtheta_all;

% Number of iterations for minimize function of GPML toolbox
n_iter = 100;

% Apply minimize function on the chosen model to optimize hyperparameters
if strcmp(gp_par.model,'igp')
    
    % Optimize
    logtheta_opt = minimize(logtheta0, 'igp', n_iter, gp_par.covfunc_x, x_train, y_train);
    
elseif strcmp(gp_par.model,'mtgp')
    
    N_out = size(y_train,2);
    
    % Reshape the observations y in a single column + take the corresponding
    % indexes of inputs (idx_in) and outputs (idx_out)
    y_train_res = reshape(y_train,[],1);
    idx_out = [];
    idx_in = [];
    for io = 1:size(y_train,2)
        idx_out((io-1)*size(y_train,1)+1:io*size(y_train,1),1) = io;
        idx_in((io-1)*size(y_train,1)+1:io*size(y_train,1),1) = 1:size(y_train,1);
    end
    
    nx = ones(length(y_train_res),1); % number of times each element of y has been observed (usually nx(i)=1)
    
    % Optimize
    logtheta_opt = minimize(logtheta0, 'nmargl_mtgp', n_iter, gp_par.logtheta_all, ...
        gp_par.covfunc_x, x_train, y_train_res, N_out, gp_par.irank, nx, ...
        idx_out, idx_in, gp_par.deriv_range);
    
elseif strcmp(gp_par.model,'slfm')
    
    N_out = size(y_train,2);
    
    % Reshape the observations y in a single column + take the corresponding
    % indexes of inputs (idx_in) and outputs (idx_out)
    y_train_res = reshape(y_train,[],1);
    idx_out = [];
    idx_in = [];
    for io = 1:size(y_train,2)
        idx_out((io-1)*size(y_train,1)+1:io*size(y_train,1),1) = io;
        idx_in((io-1)*size(y_train,1)+1:io*size(y_train,1),1) = 1:size(y_train,1);
    end
    
    nx = ones(length(y_train_res),1); % number of times each element of y has been observed (usually nx(i)=1)
    
    % Optimize
    logtheta_opt = minimize(logtheta0, 'nmargl_slfm', n_iter, gp_par.logtheta_all, ...
        gp_par.covfunc_x, x_train, y_train_res, N_out, gp_par.Q, nx, ...
        idx_out, idx_in, gp_par.deriv_range);
end

% Save the optimized hyperparameters
gp_par.logtheta_all = logtheta_opt;



