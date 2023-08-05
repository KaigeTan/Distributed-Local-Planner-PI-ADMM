switch lower(modelsys)
    case {'fmincon','matlab','native'}
        % Pre-allocate "expressions" for the duration of the prediction horizon
        x_pred = zeros(param.num_veh, param.num_ho+1);
        y_pred = zeros(param.num_veh, param.num_ho+1);
        theta_pred = zeros(param.num_veh, param.num_ho+1);
        for i_veh = 1: param.num_veh
            cost_veh = @(u) cost_function_primal(num_step, u, xt, ref_traj, param, x_pred, y_pred, theta_pred, hat_pos_old, dual_var_old, i_veh);
            nonlcon = @(u) nonlcon_function(u);
            A=[];B=[];Aeq=[];beq=[];Lb=[];Ub=[];
            initial_u_veh = [primal_u(:, 2: end), primal_u(:, end)];
            [optimal_u_veh, ~] = fmincon(cost_veh, initial_u_veh, A, B, Aeq, beq, Lb, Ub, nonlcon, options);
            veh_u = optimal_u_veh;
            % store the current iteration of pos
            [x_old_temp, y_old_temp, ~] = dynamic_update_edge(xt, veh_u, param, i_veh, 1:param.num_ho, x_pred, y_pred, theta_pred);
            pos_old([2*i_veh-1,2*i_veh], :) = [x_old_temp(i_veh, :); y_old_temp(i_veh, :)]; 
            primal_u(i_veh, :) = veh_u(i_veh, :);
        end
        
    case 'cvx'
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
            [pos_old(2*i_veh-1, :), pos_old(2*i_veh, :), ~] = dynamic_update_local(xt(i_veh, :), veh_u, param, i_veh, 1:param.num_ho);
            primal_u(i_veh, :) = veh_u;
        end
        
    case 'yalmip'
        for i_veh = 1: param.num_veh
            yalmip clear;
            veh_u = sdpvar(1, param.num_ho);
            %         assign(veh_u, zeros(size(veh_u)));
            x_pred = sdpvar(1, param.num_ho+1);
            y_pred = sdpvar(1, param.num_ho+1);
            theta_pred = sdpvar(1, param.num_ho+1);
            % Cost function to be minimized
            objective = cost_function_primal(num_step, veh_u, xt(i_veh, :), ref_traj, param, x_pred, y_pred, theta_pred, hat_pos_old, dual_var_old, i_veh);
            % local constraints
            constraints = [...
                % control input limits, -30 ~ 30 degrees
                veh_u <= pi/6*ones(size(veh_u));
                % control input limits, -20 ~ 20 degrees
                veh_u(2: end) - veh_u(1: end-1) <= pi/9;
                veh_u(1: end-1) - veh_u(2: end) <= pi/9;
                ];
            optimize(constraints, objective, options);
            % store the current iteration of pos
            [pos_old(2*i_veh-1, :), pos_old(2*i_veh, :), ~] = dynamic_update_local(xt(i_veh, :), value(veh_u), param, i_veh, 1:param.num_ho);
            primal_u(i_veh, :) = value(veh_u);
        end
    otherwise
        error('Unknown optimization framework ''%s''', modelsys);
end