function Target = target_preparation(nerve,muscles)

% Prepare the targets depending on stimulation type and desired movement
%
% INPUT
% - nerve : nerve to which stimulation is applied (sciatic/median/radial)
% - muscles : names of recorded muscles 
%
% OUTPUT
% - Target : muscles to maximize or minimize and relative weights,
%           depending on the target movement 

if strcmp(nerve,'sciatic')
    
    % Ankle flexion
    Target(1).Name = 'flexion';
    Target(1).Muscles =  {'TA','SOL','GM','PL'};
    Target(1).Weights = [1 -1 -1 -1];
    
    % Ankle extension
    Target(2).Name = 'extension';
    Target(2).Muscles =  {'TA','SOL','GM','PL'};
    Target(2).Weights = [-1 1 1 1];
    
    
elseif strcmp(nerve,'median')
    
    % Thumb
    Target(1).Name = 'thumb';
    Target(1).Muscles = {'FCR','PL','FDP','FDS','1DI','THE'};
    Target(1).Weights = [-1 -1 -1 -1 -1 1];
    
    % Wrist
    Target(2).Name = 'wrist';
    Target(2).Muscles = {'FCR','PL','FDP','FDS','1DI','THE'};
    Target(2).Weights = [1 1 -1 -1 -1 -1];
    
    % Cylinder
    Target(3).Name = 'cylinder';
    Target(3).Muscles = {'FCR','PL','FDP','FDS','1DI','THE'};
    Target(3).Weights = [-1 -1 1 1 -1 -1];
    
    % Pinch
    Target(4).Name = 'pinch';
    Target(4).Muscles = {'FCR','PL','FDP','FDS','1DI','THE'};
    Target(4).Weights = [-1 -1 -1 -1 1 1];
    
    % Sphere
    Target(5).Name = 'sphere';
    Target(5).Muscles = {'FCR','PL','FDP','FDS','1DI','THE'};
    Target(5).Weights = [-1 -1 1 1 -1 1];
    
    
elseif strcmp(nerve,'radial')
    
    % Ulnar wrist deviation
    Target(1).Name = 'ulnar';
    Target(1).Muscles = {'ECR','EDC','ECU','APL'};
    Target(1).Weights = [-1 -1 1 -1];
    
    % Radial wrist extension
    Target(2).Name = 'radial';
    Target(2).Muscles = {'ECR','EDC','ECU','APL'};
    Target(2).Weights = [1 -1 -1 -1];
    
    % Fingers Extension
    Target(3).Name = 'fingers';
    Target(3).Muscles = {'ECR','EDC','ECU','APL'};
    Target(3).Weights = [-1 1 -1 1];
    
end

% Sort the muscles indexes depending on the dataset
for iT = 1:length(Target)
    
    weights_sort = [];
    
    for iM = 1:length(muscles)
        
        idx_m = find(strcmp(Target(iT).Muscles,muscles{iM}));

        weights_sort(iM) = Target(iT).Weights(idx_m);
    end
    
    Target(iT).Muscles = muscles;
    Target(iT).Weights = weights_sort;
end