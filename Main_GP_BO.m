clear all
close all
clc

currentFolder = pwd;
addpath(genpath(currentFolder));


%% Load the dataset
% The dataset must be composed of the following variables 
% - Input = [n_data x n_inputs] matrix = matrix with the values of the tested
%           input features (e.g. Input = [1 0; 1 0; 1 0; 1 0.1; 1 0.1; ...], 
%           where I column is the stim channel ID, II column is the normalized stim amplitude value)          
% - Output = [n_data x n_outputs] matrix = matrix with the values of the observed
%           output features, i.e., preprocessed muscle responses, corresponding to those inputs
%           (e.g., Output = [0.03 0.003 0.004 0.007; 0.04 0.003 0 0.006; ...])
% - muscles = [n_outputs] cell = cell with the muscles names in the order 
%              they appear in Output (e.g., muscles = {'TA','SOL','GM','PL'})


%% Separate the dataset into training and test sets
% The training set is used to optimize the GP model hyperparameters
% The test set is used to perform BO
[tr_idx,test_idx] = find_tr_test_idx(Input);

Input_tr = Input(tr_idx,:);
Output_tr = Output(tr_idx,:);

Input_test = Input(test_idx,:);
Output_test = Output(test_idx,:);


%% TARGET preparation
Target = target_preparation('sciatic',muscles);


%% Compute max objective (ground truth)
for iT = 1:length(Target)
    
    target_name = Target(iT).Name;
    
    for it = 1:size(Output_test,1)
        
        Objective.(target_name)(it).Ch = Input_test(it,1);
        Objective.(target_name)(it).Amp = Input_test(it,2);
        Objective.(target_name)(it).MusclesActivity = Output_test(it,:);
        
        % Compute objective function
        y = Output_test(it,:);
        Objective.(target_name)(it).Obj_value = y*Target(iT).Weights';
        
    end
    
    % Find the max of the objective (BEST POINT)
    [max_obj,max_obj_idx] = max([Objective.(target_name).Obj_value]);
    Objective.(target_name)(1).BestParam = [Objective.(target_name)(max_obj_idx).Ch Objective.(target_name)(max_obj_idx).Amp];
    Objective.(target_name)(1).MaxObj = max_obj;
    
end


%% Choose the multi-output GP model
% IGP = multiple independent GPs with shared hyperparameters ()
% MTGP = multi-task gaussian process model (Bonilla et al. 2007)
% SLFM = semiparametric latent factor model (Teh et al. 2005)
models = {'igp','mtgp','slfm'};
mdl_idx = input(sprintf('Which model you want to use [1/2/3]: \n 1. IGP \n 2. MTGP \n 3. SLFM \n'));
model = models{mdl_idx};


%% Initialize the model and optimize the hyperparameters
% Initialize GP model
gp_par = initialize_model(model,size(Output,2));

% Optimize hyperparameters by maximizing the marginal likelihood of training set
gp_par = learn_model(gp_par,Input_tr,Output_tr);


%% Apply GP-BO
% Set parameters
N_runs = 50; % number of runs
N_rand = 1; % how many random points to query at the beginning of each run
N_iter = 100; % after how many iterations the run is stopped
k_ucb = 1; % parameter of UCB acquisition function to balance exploration and exploitation

% GO
All_results = struct([]);

for iT = 1:length(Target)
    
    target_name = Target(iT).Name;
    All_results(1).(target_name) = struct([]);
    
    for ir = 1:N_runs
        
        disp(['Target: ' target_name ' - Run: ' num2str(ir)]);
        
        % Initializations
        x_observed = [];
        y_observed = [];
        obj_observed = [];
        Inst_regret = [];
        Avrg_regret = [];
        Min_regret = [];
        
        for iRand = 1:N_rand
            
            % Select the random point
            idx_rand = randperm(size(Input_test,1),1);
            
            x_observed(iRand,:) = Input_test(idx_rand,:);
            
            % Query the random point (observe the responses and compute objective function)
            y_observed(iRand,:) = Output_test(idx_rand,:);
            obj_observed(iRand,1) = y_observed(iRand,:)*Target(iT).Weights';
            
            % Compute regret
            Inst_regret(iRand) = Objective.(target_name)(1).MaxObj-obj_observed(iRand,1);
            Avrg_regret(iRand) = sum(Inst_regret)/iRand;
            Min_regret(iRand) = min(Inst_regret);
            
        end
        
        % Run GP-BO
        for ii = N_rand+1:N_iter
            
            % Select the next point to query
            next_point_idx = opt_acq_fun(x_observed,y_observed,gp_par,Input_test,Target(iT),k_ucb);
            
            x_observed(ii,:) = Input_test(next_point_idx,:);
            
            % Query the selected point
            y_observed(ii,:) = Output_test(next_point_idx,:);
            obj_observed(ii,1) = Output_test(next_point_idx,:)*Target(iT).Weights';
            
            % Compute regret
            Inst_regret(ii,1) = Objective.(target_name)(1).MaxObj-obj_observed(ii);
            Avrg_regret(ii,1) = sum(Inst_regret)/ii;
            Min_regret(ii,1) = min(Inst_regret);
            
        end
        
        % Save the results
        All_results.(target_name)(ir).X_observed = x_observed;
        All_results.(target_name)(ir).Y_observed = y_observed;
        All_results.(target_name)(ir).Obj_observed = obj_observed;
        
        [~,best_point_idx] = max(obj_observed);
        All_results.(target_name)(ir).BestPoint = x_observed(best_point_idx,:);
        
        All_results.(target_name)(ir).Avrg_regret = Avrg_regret;
        All_results.(target_name)(ir).Min_regret = Min_regret;
        
        % Compute iterations for convergence
        converg = strfind(sum(diff(All_results.(target_name)(ir).X_observed),2)',zeros(1,5));
        if isempty(converg)
            All_results.(target_name)(ir).Actions_to_converge = NaN;
        else
            All_results.(target_name)(ir).Actions_to_converge = converg(1);
        end
        
    end
end


%% Plot results
plot_results(Target,All_results,N_runs,N_iter);
