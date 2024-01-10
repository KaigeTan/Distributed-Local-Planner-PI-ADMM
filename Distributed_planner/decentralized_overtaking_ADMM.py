# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:39:13 2024

@author: kaiget
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
from pypoman import plot_polygon
import sys
sys.path.append('../')
import decentralized
import copy


state_record = []
iter_num_record = []
lambda_record = []
# %% obca optimization
if_comm_delay = 0
min_dis = 1
optimizer = decentralized.OBCAOptimizer(min_dis=min_dis, prob = if_comm_delay)
init_state = [arr[0, :] for arr in optimizer.ref_traj]
init_state = np.vstack(init_state)
state_record = [init_state]
init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1

bar_state_prev = optimizer.create_bar_state()
for t_step in range(int(optimizer.T/optimizer.dt - optimizer.N_horz)): # TODO: check if -1
    print('************************************************')
    if t_step == 10:
        print('goes here!')
    # initialize at each time step, bar_state: lambda_bar, b_bar, s_bar; bar_ctrl: a_opt, steerate_opt
    
    optimizer.bar_state = copy.deepcopy(bar_state_prev)
    bar_ctrl_prev = np.array([[0]*(optimizer.N_horz-1)*optimizer.n_controls]*optimizer.num_veh)
    i_iter = 0
    while True:
        i_iter += 1
        rho = 1
       
        # optimize from vehicle side
        bar_x = []
        bar_ctrl = []
        bar_lambda_loc = []
        bar_fullx = []
        for i_veh in range(optimizer.num_veh):
            optimizer.local_initialize(t_step, init_state[i_veh*optimizer.n_states: 
                                                          (i_veh+1)*optimizer.n_states], 
                                       i_veh, max_x=150, max_y=20)
            optimizer.local_build_model()
            optimizer.local_generate_constrain()
            optimizer.local_generate_variable()
            r = 0.1*np.eye(optimizer.n_controls)
            q = 1*np.eye(optimizer.n_states)
            optimizer.local_generate_object(r, q, rho)
            optimizer.local_solve()
            
            bar_x += [np.array(optimizer.bar_x)] # TODO: put it in bar_state
            bar_ctrl += [np.append(np.array(optimizer.a_opt).T, np.array(optimizer.steerate_opt).T)]
            bar_lambda_loc += [np.array(optimizer.bar_lambda_loc)]
            bar_fullx += [np.array(optimizer.bar_fullx)]
        
        # information exchange, bar_A & bar_b update
        optimizer.bar_state_update(bar_fullx, bar_lambda_loc)
        
        # optimize from RSU side
        optimizer.edge_initialize(max_x=150, max_y=20)
        optimizer.edge_build_model()
        optimizer.edge_generate_constrain()
        optimizer.edge_generate_variable()
        optimizer.edge_generate_object(rho)
        optimizer.edge_solve()
        
        # update lambda
        optimizer.lambda_update(1)
        
        # check residuals
        primal_res = np.sum(np.sqrt((bar_ctrl[0]-bar_ctrl_prev[0])*(bar_ctrl[0]-bar_ctrl_prev[0])) + \
                            np.sqrt((bar_ctrl[1]-bar_ctrl_prev[1])*(bar_ctrl[1]-bar_ctrl_prev[1])))
        dual_res = np.sum(np.sqrt((optimizer.bar_state.lamb_bar-bar_state_prev.lamb_bar)*
                                  (optimizer.bar_state.lamb_bar-bar_state_prev.lamb_bar)))
        if (primal_res <= optimizer.primal_thres and dual_res <= optimizer.dual_thres) or i_iter > 50: # TODO: check terminate condition
            bar_state_prev = optimizer.iterate_next_state(bar_state_prev)
            if i_iter > 50:
                print('goes here')
            break # primal and dual residual within threshold, converge
        else:
            print('iter: %d, primal res: %.3f, dual res: %.3f' %(i_iter, primal_res, dual_res))
            # update bar_state in the current iteration
            bar_state_prev = copy.deepcopy(optimizer.bar_state)
            # update previous iteration result
            bar_ctrl_prev = copy.deepcopy(bar_ctrl)
    
    # iterate to the next time step
    init_state = np.vstack((bar_x[0][1], bar_x[1][1]))
    state_record += [init_state]
    iter_num_record += [i_iter]
    init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1
    lambda_record += [optimizer.bar_state.lamb_bar]
# visualization
fig, ax = plt.subplots()

# plot the vehicle center trajectories
v1x = [vec[0, 0] for vec in state_record]
v1y = [vec[0, 1] for vec in state_record]
v2x = [vec[1, 0] for vec in state_record]
v2y = [vec[1, 1] for vec in state_record]
ax.plot(v1x, v1y, 'go', ms=3, label='vehicle1 path')
ax.plot(v2x, v2y, 'ro', ms=3, label='vehicle2 path')
ax.set_xlim(0,90)

# plot the vehicle polygons
for i in range(len(state_record)):
    v1_verts = decentralized.generate_vehicle_vertices(state_record[i][0, :])
    plot_polygon(v1_verts, fill=False, color='b')
    v2_verts = decentralized.generate_vehicle_vertices(state_record[i][1, :])
    plot_polygon(v2_verts, fill=False, color='r')

plt.show()


