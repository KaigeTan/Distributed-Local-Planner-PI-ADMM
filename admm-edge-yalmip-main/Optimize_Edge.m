switch lower(modelsys)
    case {'fmincon','matlab','native'}
        x_pred = zeros(param.num_veh, param.num_ho+1);
        y_pred = zeros(param.num_veh, param.num_ho+1);
        theta_pred = zeros(param.num_veh, param.num_ho+1);
        edge_pos = zeros(param.num_veh, param.num_ho+1);
        cost_edge = @(u) cost_function_edge(u, xt_edge, param, pos_old_edge, dual_var_old_edge, prev_pred_pos, x_pred, y_pred, theta_pred, edge_pos);
        nonlcon = @(u) nonlcon_function(u);
        A=[];B=[];Aeq=[];beq=[];Lb=[];Ub=[];
        initial_u_edge = zeros(2, param.num_ho);
        [hat_u, cost] = fmincon(cost_edge, initial_u_edge, A, B, Aeq, beq, Lb, Ub, nonlcon, options);
        
    case 'cvx'
        cvx_begin quiet
        variable hat_u(param.num_veh, param.num_ho);
        expression x_pred(param.num_veh, param.num_ho+1);
        expression y_pred(param.num_veh, param.num_ho+1);
        expression theta_pred(param.num_veh, param.num_ho+1);
        expression edge_pos(param.num_veh, param.num_ho+1);
        % Cost function to be minimized
        minimize(cost_function_edge(hat_u, xt_edge, param, pos_old_edge, dual_var_old_edge, prev_pred_pos, x_pred, y_pred, theta_pred, edge_pos))  % only two vehicles here!!! a link pair is considered
        % local constraints
        % control input limits, -30 ~ 30 degrees
        hat_u <= pi/6*ones(size(hat_u));
        % control input limits, -20 ~ 20 degrees
        hat_u(:, 2:end) - hat_u(:, 1:end-1) <= pi/9;
        hat_u(:, 1:end-1) - hat_u(:, 2:end) <= pi/9;
        cvx_end
        
    case 'yalmip'
        yalmip clear;
        hat_u = sdpvar(param.num_veh, param.num_ho);
        x_pred = sdpvar(param.num_veh, param.num_ho+1);
        y_pred = sdpvar(param.num_veh, param.num_ho+1);
        theta_pred = sdpvar(param.num_veh, param.num_ho+1);
        edge_pos = sdpvar(param.num_veh, param.num_ho+1);
        % Cost function to be minimized
        objective = cost_function_edge(hat_u, xt_edge, param, pos_old_edge, dual_var_old_edge, prev_pred_pos, x_pred, y_pred, theta_pred, edge_pos);
        % local constraints
%         constraints = [...
%             % control input limits, -30 ~ 30 degrees
%             hat_u <= pi/6*ones(size(hat_u));
%             % control input limits, -20 ~ 20 degrees
%             hat_u(:, 2:end) - hat_u(:, 1:end-1) <= pi/9;
%             hat_u(:, 1:end-1) - hat_u(:, 2:end) <= pi/9;
%             ];
        constraints = nonlcon_function(hat_u) <= 0;
        optimize(constraints, objective, options);
        
    otherwise
        error('Unknown optimization framework ''%s''', modelsys);
end