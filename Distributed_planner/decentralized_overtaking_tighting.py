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
import decentralized_tighting
import copy

state_record = []
iter_num_record = []
lambda_record = []
# %% obca optimization
if_comm_delay = 0
min_dis = 1
optimizer = decentralized_tighting.OBCAOptimizer(min_dis=min_dis, prob = if_comm_delay)
init_state = [arr[0, :] for arr in optimizer.ref_traj]
init_state = np.vstack(init_state)
state_record = [init_state]
init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1

optimizer.bar_state = optimizer.create_bar_state()
for t_step in range(int(optimizer.T/optimizer.dt - optimizer.N_horz)): # TODO: check if -1
    # print('************************************************')
    if t_step == 9:
        print('goes here!')
    
    # %% main part for the distributed optimization, iterate between parallel vehicles and RSU
    # optimize from vehicle side
    bar_x = []
    for i_veh in range(optimizer.num_veh):
        optimizer.local_initialize(t_step, init_state[i_veh*optimizer.n_states: 
                                                      (i_veh+1)*optimizer.n_states], 
                                   i_veh, max_x=150, max_y=50)
        optimizer.local_build_model()
        optimizer.local_generate_constrain(t_step)
        optimizer.local_generate_variable()
        r = np.array([[1000, 0], [0, 1]]) # 1000*np.eye(optimizer.n_controls)
        q = 1000*np.eye(optimizer.n_states)
        optimizer.local_generate_object(r, q)
        optimizer.local_solve()
        bar_x += [np.array(optimizer.bar_x)] # TODO: put it in bar_state
    
    # information exchange, bar_A & bar_b update
    optimizer.bar_state_update(bar_x)
    
    # optimize from RSU side
    optimizer.edge_initialize(max_x=150, max_y=20)
    optimizer.edge_build_model()
    optimizer.edge_generate_constrain()
    optimizer.edge_generate_variable()
    optimizer.edge_generate_object()
    optimizer.edge_solve()
    
    # iterate to the next time step
    init_state = np.vstack((bar_x[0][1], bar_x[1][1]))
    state_record += [init_state]
    init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1
    lambda_record += [optimizer.bar_state.lamb_ij]
# visualization
fig, ax = plt.subplots()

# plot the vehicle center trajectories
v1x = [vec[0, 0] for vec in state_record]
v1y = [vec[0, 1] for vec in state_record]
v2x = [vec[1, 0] for vec in state_record]
v2y = [vec[1, 1] for vec in state_record]
ax.plot(v1x, v1y, 'go', ms=3, label='vehicle1 path')
ax.plot(v2x, v2y, 'ro', ms=3, label='vehicle2 path')
#ax.set_xlim(0,40)

# plot the vehicle polygons
for i in range(len(state_record)):
    v1_verts = decentralized_tighting.generate_vehicle_vertices(state_record[i][0, :])
    plot_polygon(v1_verts, fill=False, color='b')
    v2_verts = decentralized_tighting.generate_vehicle_vertices(state_record[i][1, :])
    plot_polygon(v2_verts, fill=False, color='r')

plt.show()


