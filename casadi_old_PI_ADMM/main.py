# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:03 2023

@author: ladmin
"""

from PI_ADMM_class import PI_ADMM_CASADI
import numpy as np
# from casadi import *
import casadi as ca
import matplotlib.pyplot as plt
import time

# parameter setting
trad = 0
sum_iter_num = 0

# PI_ADMM structure
PI_ADMM = PI_ADMM_CASADI()

# initialize the parameter
xt = np.array([[-10, 0, 0], [0, 20, -np.pi/2]])
windup_sat = 20
K_I_coeff = 3


# initialize the local variables, edge variables and dual variables
iter_his = np.zeros(int(PI_ADMM.param.Nt/PI_ADMM.param.dt - PI_ADMM.param.num_ho))
primal_u = np.zeros([int(PI_ADMM.param.num_veh), int(PI_ADMM.param.num_ho)])
x_vec = []
theta_vec = []
u_vec = []

# opts = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-5}
opts = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-5}
# opts = {'qpsol': 'qpoases'}

t = time.time()
total_iter = 0
for num_step in range(int(PI_ADMM.param.Nt/PI_ADMM.param.dt - PI_ADMM.param.num_ho)):
    # initialize the seed trajectory, size: N_vehicle * N_horizon
    # suppose x keep the same speed and y with no steering
    x_seed_traj = np.around(xt[:, 0] + PI_ADMM.param.dt*PI_ADMM.param.spd*np.cos(xt[:, 2]), 4)
    y_seed_traj = np.around(xt[:, 1] + PI_ADMM.param.dt*PI_ADMM.param.spd*np.sin(xt[:, 2]), 4)
    
    # initialize the local variables, edge variables and dual variables
    pos_old = np.zeros([2*PI_ADMM.param.num_veh, PI_ADMM.param.num_ho+1])
    hat_pos_old = np.zeros([2*PI_ADMM.param.num_veh, PI_ADMM.param.num_ho+1])
    dual_var_old = np.zeros([2*PI_ADMM.param.num_veh, PI_ADMM.param.num_ho+1])
    last_iter_hat_pos = hat_pos_old.copy()
    
    # record the vehicle control input
    sum_err = 0
    diff_val = 0
    err_rk_vec = []
    # receive signal, include:
        # hat_pos_old_veh, 2 X PI_ADMM.param.num_ho+1
        # dual_var_old_veh, 2 X PI_ADMM.param.num_ho+1
    for i_iter in range(PI_ADMM.param.iter_num):
        total_iter += 1
        # %% #######  perform decentralized optimization on each vehicle  #######
        for i_veh in range(PI_ADMM.param.num_veh):
            # construct the opt problem with casadi
            ui_t = ca.SX.sym('ui_t', 1, PI_ADMM.param.num_ho) # optimization variables
            cost_veh = PI_ADMM.cost_function_primal(
                num_step, ui_t, xt, hat_pos_old[2*i_veh: 2*(i_veh+1), :], 
                dual_var_old[2*i_veh: 2*(i_veh+1), :], i_veh) # objective function
            ceq_veh, cieq_veh = PI_ADMM.nonlcon_function(ui_t)  # optimization constraints
            cvx_prob_veh = {}                 # cvx_prob_veh declaration
            cvx_prob_veh['x'] = ui_t # decision vars
            cvx_prob_veh['f'] = cost_veh # objective
            cvx_prob_veh['g'] = ca.reshape(cieq_veh, -1, 1) # casadi constraints not accept matrix, convert to vector forms
            
            # # Create solver instance
            # F_prob_veh = ca.qpsol("solver", "qpoases", cvx_prob_veh) #, opts)
            # # Solve the problem using a guess
            # init_u = np.append(np.copy(primal_u[i_veh][1:]), np.copy(primal_u[i_veh][-1]))
            # sol = F_prob_veh(x0=init_u)  # initial guess
            # veh_u = np.around(np.array(sol['x']), 4)
            
            # Create solver instance
            F_prob_veh = ca.nlpsol('solver', 'ipopt', cvx_prob_veh, opts) # sqpmethod ipopt qpsol
            # Solve the problem using a guess
            init_u = np.append(np.copy(primal_u[i_veh][1:]), np.copy(primal_u[i_veh][-1]))
            results = F_prob_veh(x0 = init_u)
            veh_u = np.around(np.array(results['x']), 4)
            
            pos_old[2*i_veh], pos_old[2*i_veh+1], _ = PI_ADMM.dynamic_update_local(xt[i_veh], veh_u, i_veh, if_SX=0)
            primal_u[i_veh] = veh_u
        
        # %% #######  perform optimization for the edge side  #######
        if num_step == 17: # for debug
            print()
        prev_pred_pos = np.hstack((x_seed_traj.reshape(-1, 1), y_seed_traj.reshape(-1, 1)))    # size: N_vehicle * (2*N_horizon)
        # construct the opt problem with casadi
        hat_u = ca.SX.sym('hat_u', PI_ADMM.param.num_veh, PI_ADMM.param.num_ho) # optimization variables
        cost_edge = PI_ADMM.cost_function_edge(hat_u, xt, pos_old, dual_var_old, prev_pred_pos)
        ceq_edge, cieq_edge = PI_ADMM.nonlcon_function(hat_u)  # optimization constraints
        cvx_prob_edge = {}                   # cvx_prob_veh declaration
        cvx_prob_edge['x'] = ca.reshape(hat_u, -1, 1)           # decision vars
        cvx_prob_edge['f'] = cost_edge       # objective
        cvx_prob_edge['g'] = ca.reshape(cieq_edge, -1, 1)       # constraints
        
        # # Create solver instance
        # F_prob_edge = ca.qpsol("solver", "qpoases", cvx_prob_edge)
        # # Solve the problem using a guess
        # init_hat_u = np.zeros([PI_ADMM.param.num_veh, PI_ADMM.param.num_ho])
        # init_hat_u = ca.reshape(init_hat_u, -1, 1)
        # results = F_prob_edge(x0 = init_hat_u)  # initial guess
        # optimal_u_edge = np.around(np.array(results['x']).reshape(PI_ADMM.param.num_veh, PI_ADMM.param.num_ho), 4)
        
        # Create solver instance
        F_prob_edge = ca.nlpsol('solver','ipopt',cvx_prob_edge, opts) # ipopt sqpmethod qpsol
        # Solve the problem using a guess
        init_hat_u = np.zeros([PI_ADMM.param.num_veh, PI_ADMM.param.num_ho])
        init_hat_u = ca.reshape(init_hat_u, -1, 1)
        results = F_prob_edge(x0 = init_hat_u) # TODO
        optimal_u_edge = np.around(np.array(results['x']).reshape(PI_ADMM.param.num_veh, PI_ADMM.param.num_ho))
        
        # store the current iteration of positions
        hat_pos_old_x, hat_pos_old_y, _ = PI_ADMM.dynamic_update_edge(xt, optimal_u_edge, if_SX=0)
        for i_veh in range(PI_ADMM.param.num_veh):
            hat_pos_old[2*i_veh] = hat_pos_old_x[i_veh]
            hat_pos_old[2*i_veh + 1] = hat_pos_old_y[i_veh]
        
        # %% #######  dual variable update  #######
        x_curr_pred, y_curr_pred, theta_curr_pred = PI_ADMM.dynamic_update_edge(xt, primal_u, if_SX=0)
        pos_veh1 = np.vstack((x_curr_pred[0], y_curr_pred[0]))
        pos_veh2 = np.vstack((x_curr_pred[1], y_curr_pred[1]))
        delta_pos = pos_veh1 - pos_veh2
        dis_vec = np.sqrt(np.diag(np.transpose(delta_pos) @ delta_pos))
        dis_min = min(dis_vec)

        K_I = 3
        K_P = min(5/dis_min, 2.5)
        PI_ADMM.param.rho = max(1, min(5, 4/dis_min))
        if trad == 1:
            dual_var_old += PI_ADMM.param.rho*(pos_old - hat_pos_old) + diff_val
        else:
            dual_var_old = sum_err + K_P*(pos_old - hat_pos_old)
            sum_err = sum_err + K_I*(pos_old - hat_pos_old) + 2*diff_val # for integration part
        
        # add saturation
        dual_var_old_ori = dual_var_old
        dual_var_old = np.minimum(windup_sat, np.maximum(dual_var_old, -windup_sat))
        # add anti-wind-up part
        if np.sum(dual_var_old_ori !=dual_var_old) > 0:
            diff_val = dual_var_old - dual_var_old_ori
        else:
            diff_val = 0
            
        # %%     #######  summary  #######
        error_sk = np.sqrt(np.sum((PI_ADMM.param.rho*(last_iter_hat_pos - hat_pos_old))**2)) # dual residual
        error_rk = np.sqrt(np.sum((pos_old - hat_pos_old)**2)) # primal residual
        err_rk_vec.append(error_rk)
        if error_rk <= PI_ADMM.param.eps_pri and error_sk <= PI_ADMM.param.eps_dual and dis_vec[1] > PI_ADMM.param.dis_thres:
            curr_iter = i_iter + 1
            if num_step == 15:
                print() # KI3.5, KP10/
            break
        # store the hat_pos from the last iteration, size: (2*N_vehicle) * (N_horizon+1)
        last_iter_hat_pos = hat_pos_old
        sum_iter_num += 1
        # print('goes here!')
        
    # %% finish the iterations of optimization in one time step
    iter_his[num_step] = curr_iter
    ut = primal_u[:, 0]   # only take one action step as the actual input, size: N_vehicle * 1
    # update dynamic, xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
    for i_veh in range(PI_ADMM.param.num_veh):
        xt[i_veh, 0] = x_curr_pred[i_veh, 1]
        xt[i_veh, 1] = y_curr_pred[i_veh, 1]
        xt[i_veh, 2] = theta_curr_pred[i_veh, 1]
    print("t_step: {}, iter: {}, max dual: {}, min dual: {}, rho: {}, veh_x: {}, veh_y: {}"\
          .format(num_step+1, curr_iter, np.max(dual_var_old), np.min(dual_var_old), 
                  PI_ADMM.param.rho, xt[:, 0], xt[:, 1]))
    
    ### recording the seed trajectory predicted by the current step control,
    ### used for linearization in collision avoidance constraint in the next step.
    # size: N_vehicle * time_horizon (only consider predicted pos), 
    # dupulicate the last time horizon term
    x_seed_traj = np.append(x_curr_pred[:, 2:], x_curr_pred[:, -1]) 
    y_seed_traj = np.append(y_curr_pred[:, 2:], y_curr_pred[:, -1])
    
    # xt(:, 1:2)' size: 2 * N_timestep
    x_vec.append(np.copy(np.transpose(xt[:, :2]))) # xt(:, 1:2)' size: 2 * N_timestep
    theta_vec.append(np.copy(np.transpose(xt[:, -1]))) # track theta change
    u_vec.append(np.copy(ut))
    
elapsed_time = time.time() - t

# %% draw the plots of the trajectory
veh1_traj = np.stack([arr[0] for arr in x_vec])
veh2_traj = np.stack([arr[1] for arr in x_vec])

# Create a new figure and axis
fig, ax = plt.subplots()
# Add the scatter plot of the first line
ax.scatter(veh1_traj[:, 0], veh1_traj[:, 1], color='blue')
# Add the scatter plot of the second line
ax.scatter(veh2_traj[:, 0], veh2_traj[:, 1],color='red')
# Set the title and axis labels
ax.set_title('2D Scatter of Two Lines')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
# Show the plot
plt.show()


