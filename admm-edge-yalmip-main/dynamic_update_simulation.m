function [x_pred, y_pred, theta_pred] = dynamic_update_simulation(xt, u, param, vehicles, horizon)
% xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
% u - control input, steering angle,    size: N_vehicle * time_horizon
% x_pred - prediction x trajectory of all vehicles in finite time
% horizons,     size: N_vehicle * time_horizon+1 (consider current pos)

x_pred(:, 1) = xt(:, 1);
y_pred(:, 1) = xt(:, 2);
theta_pred(:, 1) = xt(:, 3);

for i_veh = vehicles
    const_spd = param.spd(i_veh);   % velocity of vehicle
    for k = horizon
        % use non-linear kinematic model to simulate the motion
        x_dot = const_spd * cos(theta_pred(i_veh, k));
        x_pred(i_veh, k+1) = x_pred(i_veh, k) + x_dot*param.dt;
        y_dot = const_spd * sin(theta_pred(i_veh, k));
        y_pred(i_veh, k+1) = y_pred(i_veh, k) + y_dot*param.dt;
        theta_dot = const_spd * 1/param.L * tan(u(i_veh, k));
        theta_pred(i_veh, k+1) = theta_pred(i_veh, k) + theta_dot*param.dt;
    end
end
