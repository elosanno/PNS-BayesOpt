function [tr_idx,test_idx] = find_tr_test_idx(Input)

% Determine test and Training indexes
ch_used = unique(Input(:,1));

tr_idx = [];
test_idx = [];

for iC = 1:length(ch_used)
    
    idx = find(Input(:,1)==ch_used(iC));
    
    amp_per_ch = unique(Input(idx,2));
    
    for iA = 1:length(amp_per_ch)
        
        idx_amp = find(Input(:,1)==ch_used(iC) & Input(:,2)==amp_per_ch(iA));
        
        if length(idx_amp)==1 % if there is only one repetition it goes in test set
            test_idx(end+1) = idx_amp;
        elseif length(idx_amp)==2 % 2 repetitions go both in test set
            test_idx(end+1:end+2) = idx_amp(1:2);
        elseif length(idx_amp)>2 % if they are more, 2 repetitions (chosen randomly) go in test set, all the others in training set
            idx_rand = randperm(length(idx_amp));
            test_idx(end+1:end+2) = idx_amp(idx_rand(1:2));
            tr_idx(end+1:end+length(idx_amp)-2) = idx_amp(idx_rand(3:end));
        end
        
    end
end

