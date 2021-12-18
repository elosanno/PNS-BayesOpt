function next_point_idx = opt_acq_fun(x_observed,y_observed,gp_par,Input_test,Target,k_ucb)

% Choose next point to query based on UCB-like acquisition functions 
% 
% INPUT:
% - x_observed : queried inputs 
% - y_observed : observed outputs 
% - gp_par : GP model infos and hyperparameters
% - Input_test : available points that can be queried   
% - Target : muscles to maximize or minimize and relative weights 
% - k_ucb : parameter to balance exploration and exploitation
%
% OUTPUT:
% - next_point_idx : index of next point in Input_test to query 


% X that can be sampled: unique ch-amp combinations 
x_samples = unique(Input_test,'rows');

% Compute acquisition function on the x_samples
scores = acq_fun(x_observed,y_observed,x_samples,gp_par,Target,k_ucb);

% Randomly select an element from the set with maximal scores
all_next_candidates = find(scores == max(scores));
next_query_idx = all_next_candidates(randperm(length(all_next_candidates),1));
next_query = x_samples(next_query_idx,:);

% Randomly select one among the available points with same x 
available_points = find(Input_test(:,1)==next_query(1) & Input_test(:,2)==next_query(2));

next_point_idx = available_points(randperm(length(available_points),1));