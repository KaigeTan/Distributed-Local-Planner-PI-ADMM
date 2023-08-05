clear;
% define trajectory
param.dt = 0.1; % 4 seconds, 5m/s
param.Nt = 5;
param.L = 1;
param.num_ho = 5;
param.num_veh = 2;
param.dis_thres = 1.5;
param.spd = [4; 8];
param.beta = 1000;
param.Pnorm = 5;
param.Pcost = 1;
param.iter_num = 100;
param.rho = 1;
param.eps_pri = 1;   % assign fixed threshold for now
param.eps_dual = 1;

Nt = param.Nt;
dt = param.dt;

% reference trajectory of vehicles, each size: 2 * N_step
ref_traj_A = [linspace(-10, 10, Nt/dt); zeros(1, Nt/dt)];

% traj_B1 = [2*ones(1, 1/dt); linspace(0, 1, 1/dt)]';
% traj_B2 = circular_traj(0, pi/2, [0, 1], 2, 3/dt);
% traj_B3 = [linspace(0, -1, 1/dt); 3*ones(1, 1/dt)]';
% ref_traj_B = [traj_B1; traj_B2(2: end-1, :); traj_B3]';
ref_traj_B = [zeros(1, Nt/dt); linspace(20, -20, Nt/dt)];

%% parameter selection

x_vec = [];
theta_vec = [];
u_vec = [];
% xt - initial vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
% xt = [-0.5, 1.5, 0; 2, 0, pi/2];
xt = [-10, 0, 0; 0, 20, -pi/2];
% reference trajectory of all vehicles, size: (2*N_vehicle) * N_step
ref_traj = [ref_traj_A; ref_traj_B];
windup_sat = 50;
%% iteration to get distributed ADMM
% k_i = 1;
% k_p = 1;

trad = 0;
sum_iter_num = 0;
iter_his = zeros(1, Nt/dt - param.num_ho);

tic
for num_step = 1: Nt/dt - param.num_ho
    K_I_coeff = 3;
    % initialize the seed trajectory, size: N_vehicle * N_horizon
    % suppose x keep the same speed and y with no steering
    x_seed_traj = xt(:, 1) + param.dt*param.spd.*cos(xt(:, 3));
    y_seed_traj = xt(:, 2) + param.dt*param.spd.*sin(xt(:, 3));
    
    % initialize the local variables, edge variables and dual variables
    pos_old = zeros(2*param.num_veh, param.num_ho+1);
    hat_pos_old = 1e-4*ones(2*param.num_veh, param.num_ho+1);
    dual_var_old = 1e-4*ones(2*param.num_veh, param.num_ho+1);
    last_iter_hat_pos = hat_pos_old;
    
    % record the vehicle control input
    primal_u = zeros(param.num_veh, param.num_ho);
    sum_err = 0;
    diff_val = 0;
    err_rk_vec = [];
    for i_iter = 1: param.iter_num
        %%  perform decentralized optimization on each vehicle
        for i_veh = 1: param.num_veh
            cvx_begin quiet
            variable veh_u(1, param.num_ho);
            expression theta_pred(1, param.num_ho+1);
            expression x_pred(1, param.num_ho+1);
            expression y_pred(1, param.num_ho+1);
            % Cost function to be minimized
            minimize(cost_function_primal(num_step, veh_u, xt(i_veh, :), ref_traj, param, x_pred, y_pred, theta_pred, hat_pos_old(2*i_veh-1: 2*i_veh, :), dual_var_old(2*i_veh-1: 2*i_veh, :), i_veh))
            % local constraints
            % control input limits, -30 ~ 30 degrees
            veh_u <= pi/6*ones(size(veh_u));
            %veh_u <= pi/6*ones(size(veh_u));
            % control input limits, -30 ~ 30 degrees
            veh_u(2: end) - veh_u(1: end-1) <= pi/9;
            veh_u(1: end-1) - veh_u(2: end) <= pi/9;
            cvx_end
            % store the current iteration of pos
            [pos_old(2*i_veh-1, :), pos_old(2*i_veh, :), ~] = dynamic_update_local(xt(i_veh, :), veh_u, param, i_veh);
            primal_u(i_veh, :) = veh_u;
        end
        
        %% perform optimization for the edge side
        prev_pred_pos = [x_seed_traj, y_seed_traj];     % size: N_vehicle * (2*N_horizon)
        cvx_begin quiet
        variable hat_u(param.num_veh, param.num_ho);
        expression theta_pred(param.num_veh, param.num_ho+1);
        expression x_pred(param.num_veh, param.num_ho+1);
        expression y_pred(param.num_veh, param.num_ho+1);
        expression edge_pos(2*param.num_veh, param.num_ho+1);
        % Cost function to be minimized
        minimize(cost_function_edge(hat_u, xt, param, x_pred, y_pred, theta_pred, edge_pos, pos_old, dual_var_old, prev_pred_pos))
        % local constraints
        % control input limits, -30 ~ 30 degrees
        hat_u <= pi/6*ones(size(hat_u));
        %hat_u <= pi/6*ones(size(hat_u));
        % control input limits, -30 ~ 30 degrees
        hat_u(:, 2:end) - hat_u(:, 1:end-1) <= pi/9;
        hat_u(:, 1:end-1) - hat_u(:, 2:end) <= pi/9;
        cvx_end
        % store the current iteration of pos
        [hat_pos_old_x, hat_pos_old_y, ~] = dynamic_update_edge(xt, hat_u, param);
        for i_veh = 1: param.num_veh
            hat_pos_old(2*i_veh-1, :) = hat_pos_old_x(i_veh, :);
            hat_pos_old(2*i_veh, :) = hat_pos_old_y(i_veh, :);
        end
        
        %% dual variable update
        % K_I = 1, K_P = 0, the initial parameter setting
        
        % estiamte the distance, only for 2 vehicles!!!
        
        [x_curr_pred, y_curr_pred, theta_curr_pred] = dynamic_update_edge(xt, primal_u, param);
        pos_veh1 = [x_curr_pred(1, :); y_curr_pred(1, :)];
        pos_veh2 = [x_curr_pred(2, :); y_curr_pred(2, :)];
        dis_vec = sqrt(diag((pos_veh1 - pos_veh2)'*(pos_veh1 - pos_veh2)));
        dis_min = min(dis_vec);

        K_I = K_I_coeff/dis_min;
        K_P = min(5/dis_min, 3);
        param.rho = max(1, min(5, 4/dis_min));

        if trad == 1
            dual_var_old = dual_var_old + param.rho*(pos_old - hat_pos_old) + diff_val;
        else
            dual_var_old = sum_err + K_P*(pos_old - hat_pos_old);
            sum_err = sum_err + K_I*(pos_old - hat_pos_old) + diff_val; % for integration part
        end

        % add saturation
        dual_var_old_ori = dual_var_old;
        dual_var_old = min(windup_sat, max(dual_var_old, -windup_sat));
        % add anti-wind-up part
        if sum(dual_var_old_ori ~= dual_var_old, 'all') > 0
            diff_val = (dual_var_old - dual_var_old_ori);
%             windup_sat = windup_sat + 10;
%             K_I_coeff = K_I_coeff/2;
        else
            diff_val = 0;
        end
        
        %% summary
        error_sk = sqrt(sum((param.rho*(last_iter_hat_pos - hat_pos_old)).^2, 'all')); % dual residual
        error_rk = sqrt(sum((pos_old - hat_pos_old).^2, 'all')); % primal residual
        err_rk_vec = [err_rk_vec, error_rk];
        if error_rk <= param.eps_pri && error_sk <= param.eps_dual %&& dis_vec(2) > param.dis_thres
            curr_iter = i_iter;
            if num_step == 20
                disp(''); % KI3.5, KP10/
            end
            break;
        end
        % store the hat_pos from the last iteration, size: (2*N_vehicle) * (N_horizon+1)
        last_iter_hat_pos = hat_pos_old;
        sum_iter_num = sum_iter_num + 1;
    end
    fprintf('iteration number: %d, max dual: %d, min dual: %d, rho: %d\n', curr_iter, max(dual_var_old(:)), min(dual_var_old(:)), param.rho);
    iter_his(num_step) = curr_iter;
    ut = primal_u(:, 1);   % only take one action step as the actual input, size: N_vehicle * 1
    
    % update dynamic, xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
    for i_veh = 1: param.num_veh
        xt(i_veh, 1) = x_curr_pred(i_veh, 2);
        xt(i_veh, 2) = y_curr_pred(i_veh, 2);
        xt(i_veh, 3) = theta_curr_pred(i_veh, 2);
    end
    % recording the seed trajectory predicted by the current step control,
    % used for linearization in collision avoidance constraint in the next
    % step.
    x_seed_traj = [x_curr_pred(:, 3: end), x_curr_pred(:, end)];     % size: N_vehicle * time_horizon (only consider predicted pos), dupulicate the last time horizon term
    y_seed_traj = [y_curr_pred(:, 3: end), y_curr_pred(:, end)];
    
    x_vec = [x_vec; xt(:, 1:2)'];   % xt(:, 1:2)' size: 2 * N_timestep
    theta_vec = [theta_vec; xt(:, 3)'];   % track theta change
    u_vec = [u_vec; ut'];

    traj_A_x = x_vec(1:2:end, 1)';
    traj_A_y = x_vec(2:2:end, 1)';
    plot(traj_A_x, traj_A_y, 'marker', 'o', 'color', [0.00,0.45,0.74]);
    hold on
    traj_B_x = x_vec(1:2:end, 2)';
    traj_B_y = x_vec(2:2:end, 2)';
    plot(traj_B_x, traj_B_y,  'marker', 'd', 'color', [0.85,0.33,0.10]);
    refreshdata
    drawnow

end
t_cost = toc;
%% visualize
traj_A_x = x_vec(1:2:end, 1)';
traj_A_y = x_vec(2:2:end, 1)';
plot(traj_A_x, traj_A_y,  '*');
hold on
traj_B_x = x_vec(1:2:end, 2)';
traj_B_y = x_vec(2:2:end, 2)';
plot(traj_B_x, traj_B_y,  'o');

%% function

function cost = cost_function_primal(num_step, u, xt, ref_traj, param, x_pred, y_pred, theta_pred, hat_pos, dual_var, veh_index)
    % estimate the cost for the control input u - this is for the single vehicle
    % num_step - current time step number
    % u - control input, steering angle, size: 1 * time_horizon
    % prev_pred_pos - 2 * N_horizon
    % hat_pos - edge variables, size: 2 * time_horizon+1
    % dual_var - dual variables, size: 2 * time_horizon+1
    
    % ref_val_x: x reference trajectory of the vehicle-i in the future time horizon, size: 1 * time_horizon+1
    ref_val_x = ref_traj(2*veh_index-1, num_step: num_step+param.num_ho);
    ref_val_y = ref_traj(2*veh_index, num_step: num_step+param.num_ho);
    
    % x_pred - prediction x trajectory of all vehicles in finite time horizons, size: 1 * time_horizon+1 (consider current pos)
    [x_pred, y_pred, ~] = dynamic_update_local(xt, u, param, veh_index, x_pred, y_pred, theta_pred);
    % limit the lane change of each vehicle
    cost_norm = param.Pnorm*sum(sum((ref_val_x - x_pred).^2 + (ref_val_y - y_pred).^2));
    % a smooth steering input requirement
    cost_smooth = 0;
    for i = 3: length(u)
        cost_smooth = cost_smooth + sum((u(:, i) - 2*u(:, i-1) + u(:, i-2)).^2);
    end
    % augmented Lagrangian term
    cost_AL = param.rho/2*sum(sum(([x_pred; y_pred] - hat_pos + dual_var).^2));
    % add additional term to minimize u term for fast convergence
%     cost_u = 0;
    cost_u = param.Pcost*sum(sum(u.^2));
    cost = cost_norm + cost_smooth + cost_AL + cost_u;
end

function cost = cost_function_edge(hat_u, xt, param, x_pred, y_pred, theta_pred, edge_pos, pos_old, dual_var_old, prev_pred_pos)
    % estimate the cost for the control input u - this is for the edge side, only the collision avoidance is considered
    % hat_u - control input, steering angle, size: N_vehicle * time_horizon
    % prev_pred_pos - 2 * N_horizon
    % pos_old - local variables, size: 2*N_vehicle * time_horizon+1
    % dual_var_old - dual variables, size: 2*N_vehicle * time_horizon+1
    % prev_pred_pos - predicted trajectory from the last time step iteration, size: N_vehicle * (2*time_horizon)
    % edge_pos - used for AL calculation, size: 2*N_vehicle * time_horizon+1
    
    % apply the dynamic constraint
    [x_pred, y_pred, ~] = dynamic_update_edge(xt, hat_u, param, x_pred, y_pred, theta_pred);
    
    num_veh = param.num_veh;
    cost_punish = 0;
    for i = 1: num_veh-1
        last_pos_i = repmat(prev_pred_pos(i, :), num_veh-1, 1);        % (N_vehicle-1) * (2*N_horizon), real value
        last_pos = prev_pred_pos;
        last_pos(i, :) = [];        % (N_vehicle-1) * 2, real value
        last_dis = last_pos - last_pos_i; % estimated distance from the last time step, (N_vehicle-1) * 2, real value
        last_dis = [ones(1, param.num_ho)*last_dis(1), ones(1, param.num_ho)*last_dis(2)];
        curr_pos_i = repmat([x_pred(i, 2:end), y_pred(i, 2:end)], num_veh-1, 1);        % (N_vehicle-1) * (2*N_horizon), CVX variable
        % expression curr_pos(2*num_veh, param.num_ho);
        curr_pos = [x_pred(:, 2:end), y_pred(:, 2:end)];    % N_vehicle * (2*N_horizon), CVX variable
        curr_pos(i, :) = []; % (N_vehicle-1) * (2*N_horizon), CVX variable
        curr_dis = curr_pos - curr_pos_i;   % (N_vehicle-1) * (2*N_horizon), CVX variable
        dis_temp = 2.*last_dis.*curr_dis - last_dis.^2;        % (N_vehicle-1) * (2*N_horizon), CVX variable
        
        % punishment for collision avoidance failure
        cost_punish = cost_punish + param.beta*sum(sum(max(0, (param.dis_thres^2 - (dis_temp(:, 1: end/2) + dis_temp(:, end/2+1: end))))));
        
        % update the edge_pos
        edge_pos(2*i-1, :) = x_pred(i);
        edge_pos(2*i, :) = y_pred(i);
    end
    % add additional term to minimize u term for fast convergence
    cost_u = param.Pcost*sum(sum(hat_u.^2));
%     cost_u = 0;
    cost_AL = param.rho/2*sum(sum((pos_old - edge_pos + dual_var_old).^2));
    cost = cost_punish + cost_u + cost_AL;

end

function [x_pred, y_pred, theta_pred] = dynamic_update_local(xt, u, param, veh_index, x_pred, y_pred, theta_pred)
    % xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
    % u - control input, steering angle,    size: N_vehicle * time_horizon
    % x_pred - prediction x trajectory of all vehicles in finite time
    % horizons,     size: N_vehicle * time_horizon+1 (consider current pos)
    
    if nargin  == 4       % only the case when the real number u is given
        x_pred = zeros(1, param.num_ho+1);
        y_pred = zeros(1, param.num_ho+1);
        theta_pred = zeros(1, param.num_ho+1);
        x_pred(:, 1) = xt(:, 1);
        y_pred(:, 1) = xt(:, 2);
        theta_pred(:, 1) = xt(:, 3);
        for k = 1: param.num_ho
            const_spd = param.spd(veh_index);   % velocity of vehicle
            % calculate linearized x_dot, y_dot and theta_dot, estimate the trajectory
            x_dot = -const_spd*sin(theta_pred(1, k))*theta_pred(1, k) + ...
                (const_spd*cos(theta_pred(1, k)) + const_spd*theta_pred(1, k)*sin(theta_pred(1, k)));
            x_pred(1, k+1) = x_pred(1, k) + x_dot*param.dt;
            y_dot = const_spd*cos(theta_pred(1, k))*theta_pred(1, k) + ...
                (const_spd*sin(theta_pred(1, k)) - const_spd*theta_pred(1, k)*cos(theta_pred(1, k)));
            y_pred(1, k+1) = y_pred(1, k) + y_dot*param.dt;
            theta_dot = const_spd/param.L*u(1, k);
            theta_pred(1, k+1) = theta_pred(1, k) + theta_dot*param.dt;
        end
    else
        x_pred(:, 1) = xt(:, 1);
        y_pred(:, 1) = xt(:, 2);
        theta_pred(:, 1) = xt(:, 3);

        for k = 1: param.num_ho
            const_spd = param.spd(veh_index);   % velocity of vehicle
            % calculate linearized x_dot, y_dot and theta_dot, estimate the trajectory
            x_dot = -const_spd*sin(xt(1, 3))*theta_pred(1, k) + ...
                (const_spd*cos(xt(1, 3)) + const_spd*xt(1, 3)*sin(xt(1, 3)));
            x_pred(1, k+1) = x_pred(1, k) + x_dot*param.dt;
            y_dot = const_spd*cos(xt(1, 3))*theta_pred(1, k) + ...
                (const_spd*sin(xt(1, 3)) - const_spd*xt(1, 3)*cos(xt(1, 3)));
            y_pred(1, k+1) = y_pred(1, k) + y_dot*param.dt;
            theta_dot = const_spd/param.L*u(1, k);
            theta_pred(1, k+1) = theta_pred(1, k) + theta_dot*param.dt;
        end
    end
end

function [x_pred, y_pred, theta_pred] = dynamic_update_edge(xt, u, param, x_pred, y_pred, theta_pred)
    % xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
    % u - control input, steering angle,    size: N_vehicle * time_horizon
    % x_pred - prediction x trajectory of all vehicles in finite time
    % horizons,     size: N_vehicle * time_horizon+1 (consider current pos)
    
    if nargin  == 3       % only the case when the real number u is given
        x_pred = zeros(param.num_veh, param.num_ho+1);
        y_pred = zeros(param.num_veh, param.num_ho+1);
        theta_pred = zeros(param.num_veh, param.num_ho+1);
        x_pred(:, 1) = xt(:, 1);
        y_pred(:, 1) = xt(:, 2);
        theta_pred(:, 1) = xt(:, 3);
        for k = 1: param.num_ho
            for i_veh = 1: param.num_veh
                const_spd = param.spd(i_veh);   % velocity of vehicle
                % calculate linearized x_dot, y_dot and theta_dot, estimate the trajectory
                x_dot = -const_spd*sin(theta_pred(i_veh, k))*theta_pred(i_veh, k) + ...
                    (const_spd*cos(theta_pred(i_veh, k)) + const_spd*theta_pred(i_veh, k)*sin(theta_pred(i_veh, k)));        % sin(theta_pred(i_veh, 1)) or sin(theta_pred(i_veh, k))?
                x_pred(i_veh, k+1) = x_pred(i_veh, k) + x_dot*param.dt;
                y_dot = const_spd*cos(theta_pred(i_veh, k))*theta_pred(i_veh, k) + ...
                    (const_spd*sin(theta_pred(i_veh, k)) - const_spd*theta_pred(i_veh, k)*cos(theta_pred(i_veh, k)));
                y_pred(i_veh, k+1) = y_pred(i_veh, k) + y_dot*param.dt;
                theta_dot = const_spd/param.L*u(i_veh, k);
                theta_pred(i_veh, k+1) = theta_pred(i_veh, k) + theta_dot*param.dt;
            end
        end
    else
        x_pred(:, 1) = xt(:, 1);
        y_pred(:, 1) = xt(:, 2);
        theta_pred(:, 1) = xt(:, 3);

        for k = 1: param.num_ho
            for i_veh = 1: param.num_veh
                const_spd = param.spd(i_veh);   % velocity of vehicle
                % calculate linearized x_dot, y_dot and theta_dot, estimate the trajectory
                x_dot = -const_spd*sin(xt(i_veh, 3))*theta_pred(i_veh, k) + ...
                    (const_spd*cos(xt(i_veh, 3)) + const_spd*xt(i_veh, 3)*sin(xt(i_veh, 3)));        % sin(theta_pred(i_veh, 1)) or sin(theta_pred(i_veh, k))?
                x_pred(i_veh, k+1) = x_pred(i_veh, k) + x_dot*param.dt;
                y_dot = const_spd*cos(xt(i_veh, 3))*theta_pred(i_veh, k) + ...
                    (const_spd*sin(xt(i_veh, 3)) - const_spd*xt(i_veh, 3)*cos(xt(i_veh, 3)));
                y_pred(i_veh, k+1) = y_pred(i_veh, k) + y_dot*param.dt;
                theta_dot = const_spd/param.L*u(i_veh, k);
                theta_pred(i_veh, k+1) = theta_pred(i_veh, k) + theta_dot*param.dt;
            end
        end
    end
end
