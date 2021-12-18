function plot_results(Target,All_results,N_runs,N_iter)

% Plot the results of N_runs of Bayesian Optimization (regret and actions for covergence)

%% Plot regret
for iT = 1:length(Target)
    
    target_name = Target(iT).Name;
    
    min_regret_all = [All_results.(target_name).Min_regret]';
    avrg_regret_all = [All_results.(target_name).Avrg_regret]';
    
    figure;
    
    % Minimum regret
    subplot(1,2,1);
    hold on;
    
    plot(1:N_iter,mean(min_regret_all),'k','linewidth',1.5);
    shadedErrorBar(1:N_iter,mean(min_regret_all),std(min_regret_all),'lineProps',{'color','k'});
    
    xlim([1 N_iter]);
    xlabel('Time (actions)');
    ylabel('Minimum regret');
    set(gca,'fontsize',15);
    
    
    % Average time regret
    subplot(1,2,2);
    hold on;
    
    plot(1:N_iter,mean(avrg_regret_all),'k','linewidth',1.5);
    shadedErrorBar(1:N_iter,mean(avrg_regret_all),std(avrg_regret_all),'lineProps',{'color','k'});
    
    xlim([1 N_iter]);
    xlabel('Time (actions)');
    ylabel('Average time regret');
    set(gca,'fontsize',15);

    mtit(target_name,'Interpreter','none','fontsize',15,'xoff',-.4,'yoff',+.05);    
end


%% Plot nbr of actions for convergence
figure;
hold on;

x_ticks = {};

for iT = 1:length(Target)
    
    target_name = Target(iT).Name;
    x_ticks{iT} = target_name;
    
    actions_all = [All_results.(target_name).Actions_to_converge];
    
    % Plot mean and std
    errorbar(iT,nanmean(actions_all),nanstd(actions_all),'k');
    h = bar(iT,nanmean(actions_all));
    
    % Plot the scattered data
    x_pos = -h.BarWidth/2 + h.BarWidth*(rand(1,N_runs)) + iT;
    plot(x_pos,actions_all,'o','MarkerFacecolor','w','MarkerEdgecolor','k');   
end

xlim([0 length(Target)+1]);
ylim([0 N_iter]);
xlabel('Target');
ylabel('Actions to converge');
set(gca,'fontsize',15,'Xtick',1:length(Target),'Xticklabel',x_ticks);