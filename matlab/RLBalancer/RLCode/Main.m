clear
clc
close all
%%
% Q-learning AP RL model for standing balance.
% Writer: Amin Nasr - SMPLab - amin.nasr@ubc.ca

% This code uses the Q-LEarning as the reinforcement learning schema
% To obtain the policy of controling the standing balance of an AP 
% inverted pendulum model. QAgent and Environment Classes accompanies 
% this code. They should be in the same folder as this code. 

mainAddress = pwd;

% % For Windows
mainAddress = horzcat(mainAddress,'\');
% For Linux
% mainAddress = horzcat(mainAddress,'/');

% The scope that the avarage of q-changes is obtained
window = 1000;
% Options for Epochs if it's needed
Epochs_options = [10000];


% The slider idea: The base for delay and noise condition 
% base_setting = [52 0 0 0 0];
% base_setting = [52 0.02 0.003 0.001 0.01 0.0001];
base_setting = [52 0.02 0.003 0.001 0 0];

 
% The slider idea: The conditions for delay and noise
% setting_options = [52 12 32 72 92
%                    0 0.002 0.02 0.2 2
%                    0 0.0003 0.003 0.03 0.3
%                    0 0.0001 0.001 0.01 0.1environment
%                    0 0.001 0.01 0.1 1];

setting_options = [52;
                   0.02
                   0.003
                   0.001 
                   0
                   0];

% The Q-Value stop Criteria
QStopCriteria = 0.000002;
timingStopCriteria = 55;
GainTorque = 1;
% The time step for the dynamic of the system.
dtms = 4; %ms
dt = dtms/1000;

% ThR_options = [-6 0; -5 1;-4 2;-3 3;-2 4; -1 5; 0 6];
ThR_options = [-3 6];
ThDR_options = [-15 15];
thetaNState = 10;
thetaDNState = 16;

alphasDeNum = [75];
expDeNum = [7];
gammaOpt = [0.975];
PunishOpt = [-200 -200];
rewardSides = [1 1];


% The setting for the iterations and different random seed in the model.
startIterations = 1;
endIterations = 1;


flagRun = 1;
for iThDRange = 1:size(ThDR_options,1)
    for iThRange = 1:size(ThR_options,1)
        for iGamma = 1:numel(gammaOpt)
           for iAlpha = 1:numel(alphasDeNum)         
                for iExp = 1:numel(expDeNum)       
                    for iPunishOpt = 1:size(PunishOpt,1)
                        for iRewOpt = 1:size(rewardSides,1)
                        
                            for iItr = startIterations:endIterations
                                rng(iItr);
                                
                                % To prevent having multiple runs for a same configuration
                                counterForTests = zeros(1,size(base_setting,1));
                                
                                % This for loop will iterate among the noise base conditions
                                for iSetting = 1:size(base_setting,1)
                                
                                    % This for loop will iterate among the configurations
                                    for iOption = 1:size(setting_options,1)
                                
                                        % This for loop will iterate among the options for configurations
                                        for iValueOpt = 1:size(setting_options,2)
                                      
                                            % setting variable is the configuration for this run. 
                                            setting = base_setting(iSetting,:);
                                            setting(iOption) = setting_options(iOption, iValueOpt);
                            
                                            % To prevent having a same configuration for multiple runs
                                            if setting(:)' == base_setting(iSetting,:) 
                                                counterForTests(iSetting) = ...
                                                    counterForTests(iSetting) + 1;
                                            end 
                                            if all(counterForTests(iSetting) > 1) && ...
                                                    all(setting == base_setting(iSetting,:))
                                                continue;
                                            end
                                            
                                            for iEpochNum = 1:numel(Epochs_options)
                                                
                                                tic;

                                                close all
                                                % To clear previous run. 
                                                clear env qagent act_prev action_buff q_changes q_history 
                                                clear state_buff t_history r_history state_history powerCST_history                                        
                            
                    %                             if Epochs_options(iEpochNum) == 10000 && ...
                    %                                     setting(1) == 52 && setting(2) == 0.02 && ...
                    %                                     setting(3) == 0.003 && setting(4) == 0.01 ...
                    %                                     && setting(5) == 0.01
                    %         
                    %                                 flagRun = 1;
                    %         
                    %                             end
                            
                                                if flagRun == 1
                                                    
                                                    % Starting the dynamic of the system.
                                                    ThRange = ThR_options(iThRange,:); 
                                                    ThDotRange = ThDR_options(iThDRange,:);
                                                    env = Environment;
                            
                                                    env.setMyConstant(iItr,ThRange,ThDotRange);
                                                    n = env.num_samples;
                                                         
                                                    % Adapting the configuration.
                                                    timing_delay = setting(1); %ms
                                                    env.mn_coeff = setting(2);
                                                    env.thn_coeff = setting(3);
                                                    env.thdotn_coeff = setting(4);  
                                                    
                                                    % Setting up the QL
                                                    cst = setting(5);
                            
                                                    qagent = QAgent;            
                                                    qagent.Q_vals(:,floor(qagent.n_action/2)) = 0; 
                                                    qagent.cst = setting(5);
                                                    qagent.cstc = setting(6);
                                                    qagent.setMyConstant( iItr, ThRange,ThDotRange,thetaNState, thetaDNState, ...
                                                        rewardSides(iRewOpt,:), PunishOpt(iPunishOpt,:), ...
                                                        gammaOpt(iGamma), alphasDeNum(iAlpha), expDeNum(iExp));
                                                    n_epochs = Epochs_options(iEpochNum);
                                                    q_changes = ones(1,n_epochs);
                                                    q_history = ones(1,n_epochs);
                                                    r_history = zeros(1,n_epochs);
                                                    t_history = zeros(1,n_epochs);
                                                    powerCST_history = zeros(1,n_epochs);
                                
                            %                         env.motor_noise(1,1)
                                    
                                                    % The name for figures and folders. 
                                                    Description = horzcat('QL','ITR',num2str(iItr), ...
                                                        'D',num2str(timing_delay), ...
                                                        'MN',num2str(env.mn_coeff), ...
                                                        'TN',num2str(env.thn_coeff), ...
                                                        'TDN',num2str(env.thdotn_coeff),...
                                                        'MC',num2str(qagent.cst), ...
                                                        'MCC',num2str(qagent.cstc), ...
                                                        'RB',num2str(rewardSides(iRewOpt,1)), ...
                                                        'RF',num2str(rewardSides(iRewOpt,2)), ...
                                                        'PB',num2str(PunishOpt(iPunishOpt,1)), ...
                                                        'PF',num2str(PunishOpt(iPunishOpt,2)), ...
                                                        'Gam',num2str(gammaOpt(iGamma)), ...
                                                        'Exp',num2str(expDeNum(iExp)), ...
                                                        'Alp',num2str(alphasDeNum(iAlpha)), ...
                                                        'nEp',num2str(n_epochs), ...
                                                        'rTh',horzcat(num2str(ThRange(1)),'to',num2str(ThRange(2))),...
                                                        'rTHD',horzcat(num2str(ThDotRange(1)),'to',num2str(ThDotRange(2))));
                                                    
                                                    
                                                    mkdir(Description);
                                                    cd(horzcat(mainAddress, Description));
                                                    mkdir('QValueMap');
                                                    cd(mainAddress);
                                                    
                                                    % Setting the timing and delays. 
                                                    decisionLoopTiming = timing_delay; %ms
                                                    decisionLoop = decisionLoopTiming/dtms;
                                                    delayTimeAction = timing_delay;
                                                    delayTimeSensory = timing_delay;
                                                    delaybuffAction = delayTimeAction/dtms;
                                                    delaybuffSensory = delayTimeSensory/dtms;
                                                    
                                                    n_successHist = zeros(1, window);
                                                    % Run the simulation.
                                                    for iEpisode = 1:n_epochs
                                    
                                                        % Just to watch for the speed of simulation.
                                                        if mod(iEpisode,200) == 0
                                                            iEpisode
                                                        end
                                                    
                                                        % Initialize the state
                                                        %  now_state = [qagent.n_thdot(randsample ...
                                                        %      (1:qagent.n_thetadot,1)); ...
                                                        %      qagent.n_th(randsample(1:qagent.n_theta,1))];
                                                        totalDelay = delayTimeSensory + delayTimeAction;
                    %                                     now_state = env.initiate_test(totalDelay)*pi/180;
                                                        % now_state = (randn(2,1) + [0; 3 + ThRange(1)])* pi/180;
                                                        now_state = (randn(2,1) + [0; 0.5])* pi/180;
                                                        true_state = now_state;
                                                        [delayed_disc_th, delayed_disc_thdot] = ...
                                                            qagent.find_state(now_state);
                                                        
                                                        % reset history for the passive stiffness.
                                                        env.reset_pt(true_state);
                                                        
                                                        % Initiating the action and state buffer to implement 
                                                        % the delays
                                                        action_buff = ones(delaybuffAction) * ...
                                                            floor(qagent.n_action/2);
                                                        now_act = qagent.action_zero;
                                                        for iDelay=1:delaybuffSensory
                                                            state_buff(:,iDelay) = now_state;
                                                        end               
                                                        
                                                        % This will store the accumulative reward obtained in
                                                        % the episode
                                                        episode_record = [];
                                                        rewardSum = 0;
                                    
                                                        % Simulate
                                                        for iTime = 1:n
                                                            
                                                            % To have buffer according to the time delay for
                                                            % the actions in the simulation 
                                                            action_buff(delaybuffAction) = now_act;
                                                            delayed_act = action_buff(1);
                                                            action_buff(1) = [];
                                                            
                                                            % To have states according to the time delay for
                                                            % perceiving states in the simulation. 
                                                            state_buff(:,delaybuffSensory) = now_state;
                                                            delayed_state = state_buff(:,1);
                                                            state_buff(:,1) = [];
                                                    
                                                            % Choose the action and save the previous one for 
                                                            % the QL updates.
                                                            if mod(iTime,decisionLoop) == 1
                                                                 
                                                                % Save the discretized state that the current 
                                                                % decision is being made for. This is for QL.
                                                                act_prev = now_act;
                                                                disc_prev = delayed_disc_th;
                                                                disc_prevdot = delayed_disc_thdot;
                                                                q_ind_prev = qagent.qind ...
                                                                    (disc_prevdot, disc_prev);
                                                                
                                                                % Get the current Q index - this will get saved
                                                                [delayed_disc_th, delayed_disc_thdot] = ...
                                                                    qagent.find_state(delayed_state);
                                                                delayed_q_ind = qagent.qind ...
                                                                    (delayed_disc_thdot, delayed_disc_th);
                                                                
                                                                % Choose the action based on the current
                                                                % Q-Value map.
                                                                now_act = qagent.action(delayed_q_ind);
                                                                
                                                                % This will find the raw control torque based 
                                                                % on the torque map or pair to actions.
                                                                c_torque = qagent.n_acts( delayed_act ) * GainTorque;
                                                    
                                                            end
                                                            
                                                            % Checking the torque and its sign for the muscle
                                                            % dynamic function
                                                            if sign(c_torque) == -1
                                                                MVC = 100 * GainTorque;
                                                            else
                                                                MVC = 50 * GainTorque;
                                                            end
                                                            
                                                            % passing the control torque to the muscle dynamic 
                                                            % funtion. 
                                                            true_torque = env.muscle_dyn (c_torque,MVC, 0.004);
                                                    
                                                            % Passive stiffness torque on the ankle.
                                                            ps_torque = env.passive_torque(true_state);
                                                            
                                                            % The next state based on the dynamics and inputs. 
                                                            [now_state, true_state] = env.dynamics ...
                                                                (true_state, true_torque, ...
                                                                ps_torque, iTime, iEpisode, dt);
                    
                                                            episode_record(1:2,iTime) = true_state;
                                                            episode_record(3,iTime) = true_torque;
                    
                                                            
                                                            % Checking the failure condition. 
                                                            failure_flag = env.failure_check(delayed_state);
                                                            
                                                            % Update the q-value according to the failure and 
                                                            % end this episode. 
                                                            if failure_flag == 1  
                                                                instantPunish = qagent.qPunish (now_state);
                                                                rewardSum = rewardSum + ...
                                                                    qagent.qUpdate_failure ...
                                                                    (delayed_q_ind, act_prev, now_act);
                                                                break;
                                                            end
                                                    
                                                            % Q-update for a non-failure time step. 
                                                            if mod(iTime,decisionLoop) == 1
                                                                instantReward = qagent.qReward (now_state);
                                                                [disc_th_next, disc_thdot_next] = ...
                                                                    qagent.find_state(delayed_state);
                                                                if isempty( disc_thdot_next )
                                                                    disc_thdot_next = 20;
                                                                end
                                                                q_ind_next = qagent.qind(disc_thdot_next, ...
                                                                    disc_th_next);
                                                                rewardSum = rewardSum + qagent.qUpdate ...
                                                                    (q_ind_prev, q_ind_next, act_prev, ...
                                                                    now_act);
                                                                
                                                            end          
                                                    
                                                        end
                                                        
                                                        % Storing the time and reward history in each episode.
                                                        t_history(iEpisode) = iTime * dt;
                                                        r_history(iEpisode) = rewardSum;
                                                        powerCST_history(iEpisode) = sum(abs(episode_record(1,:)).*abs(episode_record(3,:)))./ iTime;
                                
                                                        if failure_flag == 1
                                
                                                            n_successHist = horzcat(n_successHist, 0);
                                                        else
                                                            n_successHist = horzcat(n_successHist, 1);
                                
                                                        end
                                                        
                                                        n_successHist(1) = [];
                                                            
                                                        % Recording the q-history and check for the
                                                        % Q-Convergance
                                                        q_history(iEpisode) = mean(mean(qagent.Q_vals));
                                                        if iEpisode>window
                                                            q_changes(iEpisode-window)= (q_history(iEpisode)...
                                                                - q_history(iEpisode-window))/window;
                                                        end
%                                                         if iEpisode > window && ...
%                                                                 abs(q_changes(iEpisode-window)) < QStopCriteria 
%                                                                 abs(q_changes(iEpisode-window)) 
%                                                             break;
%                                                         end 

                                                        if iEpisode > window && ...
                                                                mean(t_history(iEpisode-window:iEpisode)) > timingStopCriteria 
                                                            mean(t_history(iEpisode-window:iEpisode)) 
                                                            break;
                                                        end 
                                                         
                                    
                                                    end
                                                    havingPlot = 0;
                                                    % 10 Tests for simulations.
                                                    for iTest = 1:10
                                    
                                                        % Store the character for the result of the test.
                                                        result_char = 's'; % s: successful - f: failed
                                                        
                                                        % Initialize the state and action
                                                        % now_state = [n_thdot(randsample(1:20,1)); ...
                                                        % n_th(randsample(1:20,1))];
                                                        % now_state = randn(2,1) * pi/180;
                                                        % now_act = floor(env.n_action/2);
                            %                             totalDelay = delayTimeSensory + delayTimeAction;
                            %                             now_state = env.initiate_test(totalDelay)*pi/180;
                                                        now_state = ([0; 0])* pi/180;
                                    
                                                        now_act = qagent.action_zero;
                                                        [delayed_disc_th, delayed_disc_thdot] = ...
                                                            qagent.find_state(now_state);                        
                                                        true_state = now_state;
                                                        
                                                        % Building the buffer for the simulations. 
                                                        action_buff = ones(delaybuffAction) * ...
                                                            floor(qagent.n_action/2);
                                                        for iDelay=1:delaybuffSensory
                                                                state_buff(:,iDelay) = now_state;
                                                        end
                                                            
                                                        % State history for this run
                                                        state_history = zeros( 6, n );
                                                        
                                                        % The reset for the passive stiffness. 
                                                        env.reset_pt(true_state)
                                                        
                                                        for iTime=1:n
                                                            
                                                            % Working with the buffer and doing the delay
                                                            action_buff(delaybuffAction) = now_act;
                                                            delayed_act = action_buff(1);
                                                            action_buff(1) = [];
                                                            state_buff(:,delaybuffSensory) = now_state;
                                                            delayed_state = state_buff(:,1);
                                                            state_buff(:,1) = [];
                                                                                    
                                                            % action period
                                                            if mod(iTime,decisionLoop) == 1
                                                        
                                                                % Save the discretized state that the current
                                                                % decision is being made for
                                                                act_prev = now_act;
                                                                disc_prev = delayed_disc_th;
                                                                disc_prevdot = delayed_disc_thdot;
                                                                q_ind_prev = qagent.qind(disc_prevdot, ...
                                                                    disc_prev);
                                                        
                                                                % Get the current Q index - this will get saved
                                                                [delayed_disc_th, delayed_disc_thdot] = ...
                                                                    qagent.find_state(delayed_state);
                                                                if isempty( delayed_disc_thdot )
                                                                    delayed_disc_thdot = 20;
                                                                end
                                                                delayed_q_ind = qagent.qind ...
                                                                    (delayed_disc_thdot, delayed_disc_th);
                                                                
                                                                % Choose the action based on the policy map.
                                                                now_act = qagent.policy(delayed_q_ind);  
                                                        
                                                                % Implementing the map from action to torque. 
                                                                c_torque = qagent.n_acts( delayed_act ) * GainTorque;
                                                        
                                                            end
                                                            
                                                            % Checking the torque and its sign for the muscle
                                                            % dynamic function
                                                            if sign(true_torque) == -1
                                                                MVC = 100 * GainTorque;
                                                            else
                                                                MVC = 50 * GainTorque;
                                                            end
                                                        
                                                            % Implementing the muscle dynamics. 
                                                            true_torque = env.muscle_dyn (c_torque,MVC, 0.004);
                                                            
                                                            % Implementing the nonlinear passive stiffness. 
                                                            ps_torque = env.passive_torque(true_state);
                                                            
                                                            % The dynamic for the tests. the difference is that
                                                            % for the noises they are predefined and stored in
                                                            % the root folder
                                                            [now_state, true_state, exerted_torque] =  ...
                                                                env.dynamics_test (true_state, true_torque, ...
                                                                ps_torque, iTime, iTest, dt );
                                                        
                                                            % Checking the fai
                                                            failure_flag = env.failure_check(delayed_state);
                                                        
                                                            % Storage for the test period
                                                            state_history(1:2, iTime) = true_state;
                                                            state_history(3:4, iTime) = now_state;
                                                            state_history(5, iTime) = c_torque;
                                                            state_history(6, iTime) = exerted_torque;
                                                        
                                                            % Stoping the simulation in case of failure.
                                                            if failure_flag == 1
                                                                disp( "Failed" );
                                                                result_char = 'f';
                                                                break;
                                                            end
                                                        end
                    
                                                        if failure_flag == 0
                                                            history2plot = state_history;
                                                            havingPlot = 1;
                                                        end
                                                                            
                                                        % Graphing the pendulum behaviour
                                                        figure('visible','off');
                                                        x0=300;
                                                        y0=300;
                                                        width=800;
                                                        height=350;  
                                                        set(gcf,'position',[x0,y0,width,height])   
                                                        t = 0:dt:(n-1)*dt;
                                                        plot(t, state_history(2,:)*180/pi )
                                                        nameFigure = horzcat('PendulumBehavior', ...
                                                            num2str(iTest), result_char);
                                                        title(nameFigure,'fontsize',15)
                                                        xlabel('Time(s)','fontsize',12)
                                                        ylabel('Pendulum Angle (deg)','fontsize',12) 
                                                        ylim(env.th_range)
                            %                             formatFile = horzcat(mainAddress, Description, '\', ...
                            %                                 nameFigure);
                                                        formatFile = nameFigure;
                                                        cd(horzcat(mainAddress, Description));
                                                        saveas(gcf,horzcat(formatFile,'.png'))
                                                        save(horzcat(formatFile, '.mat'),'state_history')
                                                        cd(mainAddress);
                                    
                                                    end
                                                    if havingPlot == 0
                                                        history2plot = state_history;
                                                    end
                                                        
                                                                  
                                                    % Q-Values Map.
                                                    for iAction = 1:qagent.n_action
                                    
                                                        for ith = 1:qagent.n_theta
                                                            for ithdot = 1:qagent.n_thetadot
                                                        
                                                                Qind = qagent.qind(ithdot, ith);                        
                                                                qValMap(ith,ithdot) = ...
                                                                    (qagent.Q_vals(Qind,iAction));
                                                        
                                                            end
                                                        end
                                
                                                        q_map(iAction,:,:) = qValMap;
                                    

                                
                                                        nameFigure = 'QValMap';
                            %                             formatFile = horzcat(mainAddress, Description, ...
                            %                               '\', nameFigure);
                                                        formatFile = nameFigure;
                                                        cd(horzcat(mainAddress, Description));
                                                        save(horzcat(formatFile,'.mat'),'q_map')
                                                        cd(mainAddress);
                                    
                                                    end
                                    
                                                    % Policy Map   
                                                    for ith = 1:qagent.n_theta
                                                        for ithdot = 1:qagent.n_thetadot
                                                    
                                                            delayed_q_ind = qagent.qind(ithdot, ith);                        
                                                            actMap(ith,ithdot) = qagent.n_acts ...
                                                                (qagent.policy(delayed_q_ind));
                                                            
                                                            num_visits_report(ith, ithdot) = ...
                                                                qagent.num_visits(delayed_q_ind);
                                                                
                                                    
                                                        end
                                                    end
                                    
                            
                                                    figure('visible','off');
                                                    x0=100;
                                                    y0=100;
                                                    width=1000;
                                                    height=500;        
                                                    set(gcf,'position',[x0,y0,width,height])
                                                    subplot(3,5,[1 2 3 6 7 8 11 12 13])
                                                    cDataa = actMap;
                                                    x = 2:thetaNState;
                                                    x = x - 1 + min(ThRange);
                                                    y = 2:thetaDNState;
                                                    y = (y - 1)*2;
                                                    y = y + min(ThDotRange);
                                                    imagesc(x,y,cDataa(2:end,2:end)'); axis xy;
                                                    xShift = (ThRange(2)-ThRange(1))/(2*(thetaNState-1));
                                                    yShift = (ThDotRange(2)-ThDotRange(1))/(2*(thetaDNState-1));
                                                    xticks(x+xShift);
                                                    yticks(y+yShift);                      
                                                    % Set the x and y tick labels to display the actual X and Y values
                                                    xticklabels(arrayfun(@num2str, x, 'UniformOutput', false));
                                                    yticklabels(arrayfun(@num2str, y, 'UniformOutput', false));
                                                    set(gca, 'TickDir', 'out');
                                                    % Get the size of the matrix
                                                    [matrixRows, matrixCols] = size(cDataa(2:end,2:end)');
                                                    
                                                    % Hold the current plot
                                                    hold on;
                                                    
                                                    % Draw vertical lines
                                                    for col = x(1)+(x(2)-x(1))/2:x(2)-x(1):(x(end)+(x(2)-x(1))/2)
                                                        xline(col, 'Color', [0.8, 0.8, 0.8], 'LineWidth', 0.7);
                    %                                         plot([col, col], [x(1)+0.5, x(end)+0.5], 'Color', [0.8, 0.8, 0.8], 'LineWidth', 0.5);
                                                    end
                                                    
                                                    % Draw horizontal lines
                                                    for row = y(1)+(y(2)-y(1))/2:y(2)-y(1):(y(end)+(y(2)-y(1))/2)
                                                        yline(row, 'Color', [0.8, 0.8, 0.8], 'LineWidth', 0.7);
                    %                                         plot([y(1)+0.5, y(end)+0.5], [row, row], 'Color', [0.8, 0.8, 0.8], 'LineWidth', 0.5);
                                                    end
                                                    
                                                    % Release the plot
                                                    hold off;
                                                    hcb = colorbar;
                                                    % Data range and proportions for the colormap
                                                    dataMin = -100;
                                                    dataMax = 50;
                                                    zeroPoint = (0 - dataMin) / (dataMax - dataMin);  % Proportion of zero in the data range
                                                    % Number of points in the colormap
                                                    colormapSize = 256;
                                                    midPoint = round(colormapSize * zeroPoint);         
                                                    % Create the colormap
                                                    blueToWhite = [linspace(0, 1, midPoint)', linspace(0, 1, midPoint)', ones(midPoint, 1)];
                                                    whiteToRed = [ones(colormapSize - midPoint, 1), linspace(1, 0, colormapSize - midPoint)', linspace(1, 0, colormapSize - midPoint)'];
                                                    customColormap = [blueToWhite; whiteToRed];                            
                                                    % Apply the colormap
                                                    colormap(customColormap);                            
                                                    % Optionally update colorbar limits
                                                    caxis([dataMin dataMax]);
                                                    title(hcb,'Torque (N.m)','fontsize',7)
                    %                                     nameFigure = horzcat('Trace on policy map',fileName2(end-5));
                                                    nameFigure = 'Policy map';
                                                    title(nameFigure,'fontsize',10)
                                                    xlabel('Angle (deg)','fontsize',7)
                                                    ylabel('Velocity (deg/s)','fontsize',7)
                    %                                     colormap(gray)
                                                    hold on;
                    %                                     plot (history(2,:)*180/pi+0.5,history(1,:)*180/pi+1, LineWidth = 0.75, Color=[0 0.5 0.2 0.4])
                    
                                                    subplot(3,5,[4 5 9 10])
                                                    cDataa = actMap;
                                                    x = 2+1:thetaNState-2;
                                                    x = x - 1 + min(ThRange);
                                                    y = 2+4:thetaDNState-4;
                                                    y = (y - 1)*2;
                                                    y = y + min(ThDotRange);
                                                    imagesc(x,y,cDataa(3:8,6:12)'); axis xy;
                                                    xticks(x+xShift);
                                                    yticks(y+yShift);                           
                                                    % Set the x and y tick labels to display the actual X and Y values
                                                    xticklabels(arrayfun(@num2str, x, 'UniformOutput', false));
                                                    yticklabels(arrayfun(@num2str, y, 'UniformOutput', false));
                                                    set(gca, 'TickDir', 'out');

                   
                                                    nameFigure = 'Trace on policy map';
                                                    title(nameFigure,'fontsize',9)
                    %                                     xlabel('Angle (deg)','fontsize',7)
                                                    ylabel('Velocity (deg/s)','fontsize',7)
                    %                                     colormap(gray)
                                                    hold on;
                                                    plot (history2plot(2,:)*180/pi+0.5,history2plot(1,:)*180/pi+1, LineWidth = 0.75, Color=[0 0.5 0.2 0.4])
                    %                                     h1 = plot(history(2,1)*180/pi+0.5,history(1,1)*180/pi+1,  'o', 'MarkerSize', 10, 'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'yellow'); % Initial position of the dot
                    
                                                    subplot(3,5,[14 15])
                                                    box off
                                                    T = 60;
                                                        plot (dt:dt:T,history2plot(2,:)*180/pi, LineWidth = 0.75, Color=[0 0.5 0.2 1])
                    %                                     box off
                    %                                     h2 = plot(dt:dt:T,history(2,1)*180/pi+1, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'yellow'); % Initial position of the dot
                    
                                                    title('Pendulum angle','fontsize',9)
                                                    xlabel('time (s)','fontsize',7)
                                                    ylabel('Angle (deg)','fontsize',7)
                                                    ylim ([-2 4])
                                                    set(gca, 'TickDir', 'out');
                                                    box off
                    
                    
                                                    formatFile = 'PolicyMap';
                                                    cd(horzcat(mainAddress, Description));
                                                    saveas(gcf,horzcat(formatFile,'.png'))
                                                    save(horzcat(formatFile,'.mat'),'actMap')
                                                    formatFile = 'ExpResults';
                    
                                                    figure('visible','off');
                                                    x0=100;
                                                    y0=100;
                                                    width=1000;
                                                    height=500;        
                                                    set(gcf,'position',[x0,y0,width,height])
                                                    plot(1:n_epochs,t_history,'Color', [0, 0, 1, 0.3])
                                                    hold on
                                                    plot(1:n_epochs,movmean(t_history,100),'Color', [0, 0, 1, 1])
                                                    title('Episode Length History','fontsize',9)
                                                    xlabel('Episode','fontsize',7)
                                                    ylabel('Time (s)','fontsize',7)
                                                    formatFile = 'EpisodeLength';
                                                    saveas(gcf,horzcat(formatFile,'.png'))
                    
                    
                                                    figure('visible','off');
                                                    x0=100;
                                                    y0=100;
                                                    width=1000;
                                                    height=500;        
                                                    set(gcf,'position',[x0,y0,width,height])
                                                    plot(1:n_epochs, powerCST_history,'Color', [0, 0, 1, 0.3])
                                                    hold on
                                                    plot(1:n_epochs,movmean(powerCST_history,100),'Color', [0, 0, 1, 1])
                                                    title('Cost of Each Episode','fontsize',9)
                                                    xlabel('Episode','fontsize',7)
                                                    ylabel('Power Cost (w)','fontsize',7)
                                                    formatFile = 'PowerCost';
                                                    saveas(gcf,horzcat(formatFile,'.png'))
                                                    
                                                    figure('visible','off');
                                                    x0=100;
                                                    y0=100;
                                                    width=1000;
                                                    height=500;        
                                                    set(gcf,'position',[x0,y0,width,height])
                                                    plot(1:n_epochs, r_history,'Color', [0, 0, 1, 0.3])
                                                    hold on
                                                    plot(1:n_epochs,movmean(r_history,100),'Color', [0, 0, 1, 1])
                                                    title('Return history','fontsize',9)
                                                    xlabel('Episode','fontsize',7)
                                                    ylabel('Return','fontsize',7)
                                                    formatFile = 'ReturnHistory';
                                                    saveas(gcf,horzcat(formatFile,'.png'))
                    
                                                    figure('visible','off');
                                                    x0=100;
                                                    y0=100;
                                                    width=1000;
                                                    height=500;        
                                                    set(gcf,'position',[x0,y0,width,height])
                                                    plot(1:n_epochs, q_history,'Color', [0, 0, 1, 1])                                
                                                    title('Sum of Q value history','fontsize',9)
                                                    xlabel('Episode','fontsize',7)
                                                    ylabel('Q value','fontsize',7)
                                                    formatFile = 'QvalueHistory';
                                                    saveas(gcf,horzcat(formatFile,'.png'))
                                                    elapsedTime = toc;

                                                    % Saving the outputs.
                                                    formatFile = 'LearningExp';
                                                    writematrix(elapsedTime, horzcat(formatFile, 'ET.txt'))
                                                    writematrix(q_changes(iEpisode-window), horzcat(formatFile, 'QC.txt'))
                                                    writematrix(q_history, horzcat(formatFile, 'QH.txt'))
                                                    writematrix(iEpisode, horzcat(formatFile, 'NE.txt'))
                                                    writematrix(r_history, horzcat(formatFile, 'RH.txt'))
                                                    writematrix(t_history, horzcat(formatFile, 'TH.txt'))
                                                    writematrix(n_successHist, horzcat(formatFile, 'LS.txt'))
                                                    writematrix(powerCST_history, horzcat(formatFile, 'PC.txt'))
                                                    save(horzcat(formatFile,'NV.mat'),'num_visits_report')
                            %                         writematrix(num_visits_report, horzcat(formatFile, 'NV.mat'))
                                                    cd(mainAddress);
                            
                                                end                                                                                          
                                            end
                                        end
                                    end
                                end
                            end                   
                        end
                    end
                end
            end
        end
    end
end

