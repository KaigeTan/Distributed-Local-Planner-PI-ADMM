import numpy as np
import math as m
import matplotlib.pyplot as plt
from pypoman import plot_polygon
import sys
sys.path.append('../')
import decentralized


state_record = []
# %% obca optimization
optimizer = decentralized.OBCAOptimizer()
init_state = [arr[0, :] for arr in optimizer.ref_traj]
init_state = np.vstack(init_state)
state_record = [init_state]
init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1

for t_step in range(int(optimizer.T/optimizer.dt - optimizer.N_horz)): # TODO: check if -1
    bar_state = optimizer.create_bar_state()
    # optimize from vehicle side
    bar_z = []
    for i_veh in range(optimizer.num_veh):
        optimizer.local_initialize(t_step, init_state[i_veh*optimizer.n_states: (i_veh+1)*optimizer.n_states], 
                                   bar_state, i_veh, max_x=150, max_y=20)
        optimizer.local_build_model()
        optimizer.local_generate_constrain()
        optimizer.local_generate_variable()
        r = 0.1*np.eye(optimizer.n_controls)
        q = np.eye(optimizer.n_states)
        optimizer.local_generate_object(r, q)
        optimizer.local_solve()
        bar_z += [np.array(optimizer.bar_z)]
    # optimize from RSU side
    optimizer.edge_initialize(bar_z)
    optimizer.edge_build_model()
    optimizer.edge_generate_constrain()
    optimizer.edge_generate_variable()
        
    # check the optimization results
    x_opt = np.array(np.round(optimizer.x_opt, decimals=3))
    y_opt = np.array(np.round(optimizer.y_opt, decimals=3))
    v_opt = np.array(np.round(optimizer.v_opt, decimals=3))
    heading_opt = np.array(np.round(optimizer.theta_opt, decimals=3))
    steer_opt = np.array(np.round(optimizer.steer_opt, decimals=3))
    a_opt = np.array(np.round(optimizer.a_opt, decimals=3))
    steerate_opt = np.array(np.round(optimizer.steerate_opt, decimals=3))
        
    # iterate to the next time step
    init_state = np.concatenate((x_opt[:, 1].reshape((-1, 1)), 
                                 y_opt[:, 1].reshape((-1, 1)), 
                                 v_opt[:, 1].reshape((-1, 1)),
                                 heading_opt[:, 1].reshape((-1, 1)),
                                 steer_opt[:, 1].reshape((-1, 1))), axis=1)
    state_record += [init_state]
    init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1
    
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
    v1_verts = centralized.generate_vehicle_vertices(state_record[i][0, [0, 1, 3]])
    plot_polygon(v1_verts, fill=False, color='b')
    v2_verts = centralized.generate_vehicle_vertices(state_record[i][1, [0, 1, 3]])
    plot_polygon(v2_verts, fill=False, color='r')

plt.show()

# plot_polygon(parking_obs1)
# plot_polygon(parking_obs2, color='r')
# ax.legend()
# result_state = np.array([x_opt, y_opt, heading_opt]).T
# for state in result_state:
#     verts = pyobca.generate_vehicle_vertices(
#         pyobca.SE2State(state[0], state[1], state[2]), base_link=True)
#     plot_polygon(verts, fill=False, color='b')
# v_opt = optimizer.v_opt.elements()
# a_opt = optimizer.a_opt.elements()
# t = [optimizer.T*k for k in range(len(v_opt))]
# t_a = [optimizer.T*k for k in range(len(a_opt))]


# fig2, ax2 = plt.subplots(3)
# ax2[0].plot(t, v_opt, label='v-t')
# ax2[1].plot(t_a, a_opt, label='a-t')
# ax2[2].plot(t, steer_opt, label='steering-t')
# ax2[0].legend()
# ax2[1].legend()
# ax2[2].legend()
# plt.show()

