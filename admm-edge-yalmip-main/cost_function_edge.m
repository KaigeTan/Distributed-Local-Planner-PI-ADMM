function cost = cost_function_edge(hat_u, xt, param, pos_old, dual_var_old, prev_pred_pos, x_pred, y_pred, theta_pred, edge_pos)
% estimate the cost for the control input u - this is for the edge side, only the collision avoidance is considered
% hat_u - control input, steering angle, size: N_vehicle * time_horizon
% prev_pred_pos - 2 * N_horizon
% pos_old - local variables, size: 2*N_vehicle * time_horizon+1
% dual_var_old - dual variables, size: 2*N_vehicle * time_horizon+1
% prev_pred_pos - predicted trajectory from the last time step iteration, size: N_vehicle * (2*time_horizon)
% edge_pos - used for AL calculation, size: 2*N_vehicle * time_horizon+1

% apply the dynamic constraint
[x_pred, y_pred, ~] = dynamic_update_edge(xt, hat_u, param, 1:param.num_veh, 1:param.num_ho, x_pred, y_pred, theta_pred);

num_veh = param.num_veh;
cost_punish = 0;
for i = 1: num_veh
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
    %cost_punish = cost_punish + param.beta*sum(sum((param.dis_thres^2 - (dis_temp(:, 1: end/2) + dis_temp(:, end/2+1: end)))));
    
    % update the edge_pos
    edge_pos(2*i-1, :) = x_pred(i, :);
    edge_pos(2*i, :) = y_pred(i, :);
end
% add additional term to minimize u term for fast convergence
cost_u = param.Pcost*sum(sum(hat_u.^2));
cost_AL = param.rho/2*sum(sum((pos_old - edge_pos + dual_var_old).^2));
cost = cost_punish + cost_u + cost_AL;
end