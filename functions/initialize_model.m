function gp_par = initialize_model(model,n_out)

% Inizialize GP model hyperparameters
% 
% INPUT
% - model type : 'igp'/'mtgp'/'slfm'
% - n_out : number of outputs of the GP model
%
% OUTPUT:
% - gp_par : GP model infos and initial hyperparameter values 


%% Save model type
gp_par.model = model;


%% Initialize the mean function
% We use a constant mean function, initialized at 0
mu0 = 0;


%% Initialize input and output covariance functions + noise variances
if strcmp(model,'igp')
    
    gp_par.covfunc_x = {'covProd',{'covMatern5ard','covMatern5ard'}}; % input covariance type 
    logtheta_kx0 = [0 ; inf ; 0 ; inf ; 0 ; 0]; % input covariance hyperparameters   
    logtheta_lf0 = [];  % no output covariance 
    logtheta_sigma0 = -1; % noise variance

elseif strcmp(model,'mtgp')
    
    gp_par.covfunc_x = {'covProd',{'covMatern5ard','covMatern5ard'}}; % input covariance type
    logtheta_kx0 = [0 ; inf ; 0 ; inf ; 0 ; 0]; % input covariance hyperparameters  
    gp_par.irank = n_out; % full-rank output covariance 
    Kf0 = eye(n_out); % output covariance initialized to a diagonal matrix (no task correlations)
    Lf0 = chol(Kf0)';
    logtheta_lf0 = lowtri2vec_inchol(Lf0,n_out,gp_par.irank); % output covariance hyperparameters
    logtheta_sigma0 = -1 * ones(n_out,1); % noise variances 
    
elseif strcmp(model,'slfm')
    
    gp_par.Q = n_out; % number of latent functions
    for iq = 1:gp_par.Q
        gp_par.covfunc_x{iq,1} = {'covProd',{'covMatern5ard','covMatern5ard'}}; % input covariance type 
    end
    logtheta_kx0 = repmat([0 ; inf ; 0 ; inf ; 0 ; 0],gp_par.Q,1); % input covariance hyperparameters
    Kf0 = eye(n_out,gp_par.Q); % Initialize the output covariance to a diagonal matrix (no task correlations)
    logtheta_lf0 = reshape(Kf0,n_out*gp_par.Q,1); % output covariance hyperparameters
    logtheta_sigma0 = -1 * ones(n_out,1); % noise variances
    
end


%% Save the initial hyperparameters
gp_par.logtheta_all = [logtheta_lf0; logtheta_kx0; logtheta_sigma0; mu0]; % initial values 
gp_par.deriv_range = 1:length(gp_par.logtheta_all); % which hyperparameters we want to optimize 



