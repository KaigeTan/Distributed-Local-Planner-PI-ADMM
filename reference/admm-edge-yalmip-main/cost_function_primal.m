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
[x_pred, y_pred, ~] = dynamic_update_edge(xt, u, param, veh_index, 1:param.num_ho, x_pred, y_pred, theta_pred);
% limit the lane change of each vehicle
cost_norm = param.Pnorm*sum(sum((ref_val_x - x_pred(veh_index, :)).^2 + (ref_val_y - y_pred(veh_index, :)).^2));
% a smooth steering input requirement
cost_smooth = sum((u(veh_index, 3:end) - 2*u(veh_index, 2:end-1) + u(veh_index, 1:end-2)).^2);
% augmented Lagrangian term
cost_AL = 0;
for i = 1: param.num_veh
    if i == veh_index
        continue
    end
    cost_AL = cost_AL + param.rho/2*sum(sum(([x_pred(veh_index, :); y_pred(veh_index, :)] - hat_pos{veh_index, i} + dual_var{veh_index, i}).^2));
end
% add additional term to minimize u term for fast convergence
cost_u = param.Pcost*sum(sum(u.^2));
cost = cost_norm + cost_smooth + cost_AL + cost_u;
end