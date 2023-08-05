clear
rng(1)
%cvx_solver Gurobi %sedumi, Gurobi, SDPT3, Mosek
% cvx_save_prefs

param.dt = 0.1; % time step
param.Nt = 5;
param.L = 1;
param.num_ho = 8;
param.num_veh = 2;
param.dis_thres = 2;
param.spd = [4; 8];
param.beta = 1000;
param.Pnorm = 5;
param.Pcost = 1;
param.iter_num = 100;
param.rho = 3.5;
param.eps_pri = 0.1;   % assign fixed threshold for now
param.eps_dual = 0.1;
param.kP = 0;
param.kI = param.rho;
param.theta1 = 5;
param.theta2 = 3;
param.PI = 1; % PI -- 1: PI-ADMM, 0: static penalty factor
param.windup = 1; % windup -- 1: with backcalculation 0: without backcalculation

Nt = param.Nt;
dt = param.dt;

% reference trajectory of vehicles, each size: 2 * N_step
ref_traj_A = [linspace(-10, 10, Nt/dt); zeros(1, Nt/dt)];
ref_traj_B = [zeros(1, Nt/dt); linspace(20, -20, Nt/dt)];

%% parameter selection

x_vec = [];
theta_vec = [];
u_vec = [];
% xt - initial vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
xt = [-10, 0, 0; 0, 20, -pi/2];
% reference trajectory of all vehicles, size: (2*N_vehicle) * N_step
ref_traj = [ref_traj_A; ref_traj_B];
windup_sat = 30;

%% iteration to get distributed ADMM
trad = 0;
sum_iter_num = 0;
iter_his = zeros(1, Nt/dt - param.num_ho);
t_cost_his = zeros(1, Nt/dt - param.num_ho);
primal_u = zeros(param.num_veh, param.num_ho);

for num_step = 1: Nt/dt - param.num_ho
    tic
    % initialize the seed trajectory, size: N_vehicle * N_horizon
    % suppose x keep the same speed and y with no steering
    x_seed_traj = xt(:, 1) + param.dt*param.spd.*cos(xt(:, 3));
    y_seed_traj = xt(:, 2) + param.dt*param.spd.*sin(xt(:, 3));
    
    % initialize the local variables, edge variables and dual variables
    pos_old = zeros(2*param.num_veh, param.num_ho+1);
    hat_pos_old{param.num_veh, param.num_veh} = [];
    [hat_pos_old{:}] = deal(zeros(2, param.num_ho+1));
    dual_var_old{param.num_veh, param.num_veh} = [];
    [dual_var_old{:}] = deal(zeros(2, param.num_ho+1));
    last_iter_hat_pos = hat_pos_old;
    % the matrix to store the back calculation results
    diff_val_veh{param.num_veh, param.num_veh} = [];
    [diff_val_veh{:}] = deal(zeros(2, param.num_ho+1));

    % edge_mat to record the potential collision pairs
    edge_mat = zeros(param.num_veh, param.num_veh);
    kP_mat = param.kP*ones(param.num_veh, param.num_veh);

    % record the vehicle control input
    sum_err_veh1 = 0;
    sum_err_veh2 = 0;

    err_rk_vec = [];
    flag = 0;
    curr_iter = 1;
    primal_u = zeros(param.num_veh, param.num_ho);

    for i_iter = 1: param.iter_num
        %%  perform decentralized optimization on each vehicle
        for i_veh = 1: param.num_veh
            cvx_begin quiet
            variable veh_u(1, param.num_ho);
            expression theta_pred(1, param.num_ho+1);
            expression x_pred(1, param.num_ho+1);
            expression y_pred(1, param.num_ho+1);
            % Cost function to be minimized
            minimize(cost_function_primal(num_step, veh_u, xt(i_veh, :), ref_traj, param, x_pred, y_pred, theta_pred, hat_pos_old, dual_var_old, i_veh))
            % local constraints
            % control input limits, -30 ~ 30 degrees
            veh_u <= pi/6*ones(size(veh_u));
            % control input limits, -20 ~ 20 degrees
            veh_u(2: end) - veh_u(1: end-1) <= pi/9;
            veh_u(1: end-1) - veh_u(2: end) <= pi/9;
            cvx_end
            % store the current iteration of pos
            [pos_old(2*i_veh-1, :), pos_old(2*i_veh, :), ~] = dynamic_update_local(xt(i_veh, :), veh_u, param, i_veh);
            primal_u(i_veh, :) = veh_u;
        end
        
        %% judge if collision
        % edge_mat: binary matrix, 1 if link collision, 0 if none collision
        for i = 1: param.num_veh
            for j = i+1: param.num_veh
                edge_mat(i, j) = max(((pos_old(2*i-1, :) - pos_old(2*j-1, :)).^2 + (pos_old(2*i, :) - pos_old(2*j, :)).^2) < param.dis_thres^2);
            end
        end
        % no collision if no edge links
        if sum(edge_mat, 'all') == 0 && flag == 0
            break;
        else
        % o.w. update dual/consensus variables of each link separately
            flag = 1;
        end

        %% perform optimization for the edge side
        [edge_row, edge_col] = find(edge_mat == 1);
        for i_edge = 1: length(edge_col)
            veh1 = edge_row(i_edge);
            veh2 = edge_col(i_edge);
            prev_pred_pos = [x_seed_traj(veh1), y_seed_traj(veh1); 
                             x_seed_traj(veh2), y_seed_traj(veh2)];     % size: 2 * 2
            xt_edge = [xt(veh1, :); xt(veh2, :)];     % size: 2 * 3
            pos_old_edge = [pos_old(2*veh1-1: 2*veh1, :); pos_old(2*veh2-1: 2*veh2, :)];   % size: 2 * param.num_ho+1
            dual_var_old_edge = [dual_var_old{veh1, veh2}; dual_var_old{veh2, veh1}];

            cvx_begin quiet
            variable hat_u(2, param.num_ho);
            expression x_pred(2, param.num_ho+1);
            expression y_pred(2, param.num_ho+1);
            expression theta_pred(2, param.num_ho+1);
            expression edge_pos(2, param.num_ho+1);
            % Cost function to be minimized
            minimize(cost_function_edge(hat_u, xt_edge, param, x_pred, y_pred, theta_pred, edge_pos, pos_old_edge, dual_var_old_edge, prev_pred_pos))  % only two vehicles here!!! a link pair is considered
            % local constraints
            % control input limits, -30 ~ 30 degrees
            hat_u <= pi/6*ones(size(hat_u));
            % control input limits, -20 ~ 20 degrees
            hat_u(:, 2:end) - hat_u(:, 1:end-1) <= pi/9;
            hat_u(:, 1:end-1) - hat_u(:, 2:end) <= pi/9;
            cvx_end

            % store the current iteration of pos
            [hat_pos_old_x, hat_pos_old_y, ~] = dynamic_update_edge(xt_edge, hat_u, param);
            hat_pos_old{veh1, veh2} = [hat_pos_old_x(1, :); hat_pos_old_y(1, :)];
            hat_pos_old{veh2, veh1} = [hat_pos_old_x(2, :); hat_pos_old_y(2, :)];

            %% dual variable update
            pos_veh1 = pos_old(1:2, :); % [x_curr_pred(1, :); y_curr_pred(1, :)];
            pos_veh2 = pos_old(3:4, :); % [x_curr_pred(2, :); y_curr_pred(2, :)];
            dis_vec = sqrt(diag((pos_veh1 - pos_veh2)'*(pos_veh1 - pos_veh2)));
            if param.PI == 0
                dual_var_old{veh1, veh2} = dual_var_old{veh1, veh2} + param.rho*(pos_old(2*veh1-1: 2*veh1, :) - hat_pos_old{veh1, veh2});
                dual_var_old{veh2, veh1} = dual_var_old{veh2, veh1} + param.rho*(pos_old(2*veh2-1: 2*veh2, :) - hat_pos_old{veh2, veh1});
            else
                kP_mat(veh1, veh2) = param.theta1 - param.theta2/(1 + exp(-min(dis_vec)));
                curr_err_veh1 = pos_old(2*veh1-1: 2*veh1, :) - hat_pos_old{veh1, veh2};  % current error for PI controller
                curr_err_veh2 = pos_old(2*veh2-1: 2*veh2, :) - hat_pos_old{veh2, veh1};  % current error for PI controller
                sum_err_veh1 = sum_err_veh1 + param.kI*curr_err_veh1 + diff_val_veh{veh1, veh2}; % for integration part
                sum_err_veh2 = sum_err_veh2 + param.kI*curr_err_veh2 + diff_val_veh{veh2, veh1}; % for integration part
                dual_var_old{veh1, veh2} = sum_err_veh1 + kP_mat(veh1, veh2)*curr_err_veh1;
                dual_var_old{veh2, veh1} = sum_err_veh2 + kP_mat(veh1, veh2)*curr_err_veh2;
            end

            % add saturation
            if param.windup == 0
                continue
            end
            for i_veh = 1: param.num_veh
                for j_veh = 1: param.num_veh
                    if i_veh == j_veh
                        continue
                    else
                        dual_var_old_ori = dual_var_old{i_veh, j_veh};
                        dual_var_old{i_veh, j_veh} = min(windup_sat, max(dual_var_old{i_veh, j_veh}, -windup_sat));
                        % add anti-wind-up part
                        if (sum(dual_var_old_ori ~= dual_var_old{i_veh, j_veh}, 'all') > 0) && param.windup == 1
                            diff_val_veh{i_veh, j_veh} = (dual_var_old{i_veh, j_veh} - dual_var_old_ori);
                        else
                            diff_val_veh{i_veh, j_veh} = zeros(2, param.num_ho+1);
                        end
                    end
                end
            end
        end

        %% summary
        error_sk = 0;
        error_rk = 0;
        for i_edge = 1: length(edge_col)
            veh1 = edge_row(i_edge);
            veh2 = edge_col(i_edge);
            error_sk = error_sk + 2*sqrt(sum((param.rho*(last_iter_hat_pos{veh1, veh2} - hat_pos_old{veh1, veh2})).^2, 'all')); % dual residual
            veh_pos_old = [pos_old(2*veh1-1, :); pos_old(2*veh1, :)];
            error_rk = error_rk + 2*sqrt(sum((veh_pos_old - hat_pos_old{veh1, veh2}).^2, 'all')); % primal residual
        end
        err_rk_vec = [err_rk_vec, error_rk];
        if error_rk <= param.eps_pri && error_sk <= param.eps_dual && dis_vec(2) > param.dis_thres
            curr_iter = i_iter;
            break;
        end
        % store the hat_pos from the last iteration, size: (2*N_vehicle) * (N_horizon+1)
        last_iter_hat_pos = hat_pos_old;
        sum_iter_num = sum_iter_num + 1;
        curr_iter = i_iter;
    end
    t_cost = toc;
    [x_curr_pred, y_curr_pred, theta_curr_pred] = dynamic_update_edge(xt, primal_u, param);
    fprintf('step: %d, iteration number: %d, rho: %d\n', num_step, curr_iter, kP_mat(1, 2)*param.PI + param.rho*(1 - param.PI));
    iter_his(num_step) = curr_iter;
    t_cost_his(num_step) = t_cost;
    ut = primal_u(:, 1);   % only take one action step as the actual input, size: N_vehicle * 1
    
    % update dynamic, xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
    for i_veh = 1: param.num_veh
        xt(i_veh, 1) = x_curr_pred(i_veh, 2);
        xt(i_veh, 2) = y_curr_pred(i_veh, 2);
        xt(i_veh, 3) = theta_curr_pred(i_veh, 2);
    end

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

%% summary
fprintf('iterations: %d, avg iter time: %d, max iter time: %d\n', sum(iter_his), sum(t_cost_his)/(Nt/dt - param.num_ho), max(t_cost_his));

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
    cost_smooth = sum((u(:, 3:end) - 2*u(:, 2:end-1) + u(:, 1:end-2)).^2);
    % augmented Lagrangian term
    cost_AL = 0;
    for i = 1: param.num_veh
        if i == veh_index
            continue
        end
        cost_AL = cost_AL + param.rho/2*sum(sum(([x_pred; y_pred] - hat_pos{veh_index, i} + dual_var{veh_index, i}).^2));
    end
    % add additional term to minimize u term for fast convergence
    cost_u = param.Pcost*sum(sum(u.^2));
    cost = cost_norm + cost_smooth + cost_AL + cost_u;
end

function cost = cost_function_edge(hat_u, xt, param, x_pred, y_pred, theta_pred, edge_pos, pos_old, dual_var_old, prev_pred_pos)
    % estimate the cost for the control input u - this is for the edge side, only the collision avoidance is considered
    % hat_u - control input, steering angle, size: 2 * time_horizon
    % prev_pred_pos - 2 * N_horizon
    % pos_old - local variables, size: 2 * time_horizon+1
    % dual_var_old - dual variables, size: 2 * time_horizon+1
    % prev_pred_pos - predicted trajectory from the last time step iteration, size: 2 * 2
    % edge_pos - used for AL calculation, size:  * time_horizon+1
    
    % apply the dynamic constraint
    [x_pred, y_pred, ~] = dynamic_update_edge(xt, hat_u, param, x_pred, y_pred, theta_pred);
    last_dis = prev_pred_pos(2, :) - prev_pred_pos(1, :);
    curr_dis = [(x_pred(2, 2:end) - x_pred(1, 2:end)); (y_pred(2, 2:end) - y_pred(1, 2:end))];
    dis_temp = 2*last_dis*curr_dis - last_dis*last_dis';  % size: 1 * time_horizon

    % punishment for collision avoidance failure
    cost_punish = param.beta*sum(sum(max(0, (param.dis_thres^2 - dis_temp))));

    % update the edge_pos (4 X T+1)
    edge_pos(1, :) = x_pred(1, :);
    edge_pos(3, :) = x_pred(2, :);
    edge_pos(2, :) = y_pred(1, :);
    edge_pos(4, :) = y_pred(2, :);
    % add additional term to minimize u term for fast convergence
    cost_u = param.Pcost*sum(sum(hat_u.^2));
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
