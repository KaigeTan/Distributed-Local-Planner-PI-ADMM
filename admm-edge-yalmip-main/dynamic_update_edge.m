function [x_pred, y_pred, theta_pred] = dynamic_update_edge(xt, u, param, veh, horizon, x_pred, y_pred, theta_pred)
% xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
% u - control input, steering angle,    size: N_vehicle * time_horizon
% x_pred - prediction x trajectory of all vehicles in finite time
% horizons,     size: N_vehicle * time_horizon+1 (consider current pos)

x_pred(veh, 1) = xt(veh, 1);
y_pred(veh, 1) = xt(veh, 2);
theta_pred(veh, 1) = xt(veh, 3);

for i_veh = veh
    const_spd = param.spd(i_veh);   % velocity of vehicle
    for k = horizon
        % calculate linearized x_dot, y_dot and theta_dot, estimate the trajectory
        x_dot = const_spd * -sin(xt(i_veh, 3)) * (theta_pred(i_veh, k) - xt(i_veh, 3)) + ... % = df(x₀)/dx × (x - x₀)
            const_spd * cos(xt(i_veh, 3)); % f(x₀)
        x_pred(i_veh, k+1) = x_pred(i_veh, k) + x_dot*param.dt;
        y_dot = const_spd * cos(xt(i_veh, 3)) * (theta_pred(i_veh, k) - xt(i_veh, 3)) + ...
            const_spd * sin(xt(i_veh, 3)); 
        y_pred(i_veh, k+1) = y_pred(i_veh, k) + y_dot*param.dt;
        theta_dot = const_spd/param.L*u(i_veh, k);
        theta_pred(i_veh, k+1) = theta_pred(i_veh, k) + theta_dot*param.dt;
    end
end
