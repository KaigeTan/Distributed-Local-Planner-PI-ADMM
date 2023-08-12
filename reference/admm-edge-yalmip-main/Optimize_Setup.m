switch lower(modelsys)
    case {'fmincon','matlab','native'}
        options = optimoptions('fmincon', ...
            'Algorithm', 'interior-point',... %'sqp'
            'Display', 'none', ...
            'MaxIterations', 100);
        
    case 'cvx'
        cvx_solver Gurobi %sedumi, Gurobi, SDPT3, Mosek
        % cvx_save_prefs
        
    case 'yalmip'
        options = sdpsettings('verbose', false, 'solver', 'ipopt', 'usex0', false);
        
    otherwise
        error('Unknown optimization framework ''%s''', modelsys);
end