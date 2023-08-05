function [c, ceq] = nonlcon_function(u)
    ineq1 = u - pi/6*ones(size(u));
    ineq2 = -u - pi/6*ones(size(u));
    % control input limits, -30 ~ 30 degrees
    ineq3 = [u(:, 2: end) - u(:, 1: end-1) - pi/9, zeros(size(u, 1), 1)];
    ineq4 = [u(:, 1: end-1) - u(:, 2: end) - pi/9, zeros(size(u, 1), 1)];
    ceq = [];
    c = [ineq1; ineq2; ineq3; ineq4];
end
