import numpy as np
import math as m
import matplotlib.pyplot as plt
from pypoman import plot_polygon
from cProfile import label
import sys
sys.path.append('../')
import centralized


state_record = []
lambda_record = []
# %% obca optimization
if_comm_delay = 0
min_dis = 1
optimizer = centralized.OBCAOptimizer()
init_state = [arr[0, :] for arr in optimizer.ref_traj]
init_state = np.vstack(init_state)
state_record = [init_state]
init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1
for t_step in range(int(optimizer.T/optimizer.dt - optimizer.N_horz)): # TODO: check if -1
    if t_step == 9:
        print('')
    optimizer.initialize(t_step, init_state, max_x=150, max_y=20, 
                         prob=if_comm_delay, min_dis=min_dis)
    optimizer.build_model() # TODO: check if this step needs to be in the loop
    optimizer.generate_constrain()
    optimizer.generate_variable()
        
    r = 0.1*np.eye(optimizer.n_controls*optimizer.num_veh)
    q = np.eye(optimizer.n_states*optimizer.num_veh)
    optimizer.generate_object(r, q)
    optimizer.solve()
        
    # check the optimization results
    x_opt = np.array(np.round(optimizer.x_opt, decimals=3))
    y_opt = np.array(np.round(optimizer.y_opt, decimals=3))
    v_opt = np.array(np.round(optimizer.v_opt, decimals=3))
    heading_opt = np.array(np.round(optimizer.theta_opt, decimals=3))
    steer_opt = np.array(np.round(optimizer.steer_opt, decimals=3))
    a_opt = np.array(np.round(optimizer.a_opt, decimals=3))
    steerate_opt = np.array(np.round(optimizer.steerate_opt, decimals=3))
    lambda_opt = np.array(np.round(optimizer.lambda_result, decimals=3))
    lambda_opt = np.reshape(lambda_opt, (optimizer.N_horz-1, optimizer.num_veh*optimizer.n_dual_variable)) # time-step wise
    lambda_opt = np.reshape(lambda_opt, (optimizer.N_horz-1, optimizer.num_veh, optimizer.n_dual_variable), 'F') # N_horz X num_veh X n_dual_variable
    
        
    # iterate to the next time step
    init_state = np.concatenate((x_opt[:, 1].reshape((-1, 1)), 
                                 y_opt[:, 1].reshape((-1, 1)), 
                                 v_opt[:, 1].reshape((-1, 1)),
                                 heading_opt[:, 1].reshape((-1, 1)),
                                 steer_opt[:, 1].reshape((-1, 1))), axis=1)
    state_record += [init_state]
    lambda_record += [lambda_opt]
    init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1
    
# visualization
fig, ax = plt.subplots(figsize=(20, 4))

# plot the vehicle center trajectories
v1x = [vec[0, 0] for vec in state_record]
v1y = [vec[0, 1] for vec in state_record]
v2x = [vec[1, 0] for vec in state_record]
v2y = [vec[1, 1] for vec in state_record]
ax.plot(v1x, v1y, 'go', ms=10, label='vehicle1 path')
ax.plot(v2x, v2y, 'ro', ms=10, label='vehicle2 path')
ax.set_xlim(0,100)

# plot the vehicle polygons
for i in range(len(state_record)):
    v1_verts = centralized.generate_vehicle_vertices(state_record[i][0, :])
    plot_polygon(v1_verts, fill=False, linewidth=5, color='b')
    v2_verts = centralized.generate_vehicle_vertices(state_record[i][1, :])
    plot_polygon(v2_verts, fill=False, linewidth=5, color='r')

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

