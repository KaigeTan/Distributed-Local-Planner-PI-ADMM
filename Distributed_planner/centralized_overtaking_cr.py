import numpy as np
import math as m
import matplotlib.pyplot as plt
from pypoman import plot_polygon
from cProfile import label
import sys
sys.path.append('../')
import centralized

import os
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
import numpy as np
import vehiclemodels


# %% commonroad scenario loader
# load the CommonRoad scenario that has been created in the CommonRoad tutorial
file_path = os.path.join(os.getcwd(), 'modified_scenario.xml')
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
# plot the scenario for each time step
# for i in range(0, 40):
#     plt.figure(figsize=(25, 10))
#     rnd = MPRenderer()
#     rnd.draw_params.time_begin = i
#     scenario.draw(rnd)
#     planning_problem_set.draw(rnd)
#     rnd.render()
    
dT = scenario.dt # time step
# get the initial state of the ego vehicle from the planning problem set
planning_problem_v1 = planning_problem_set.find_planning_problem_by_id(100)
planning_problem_v2 = planning_problem_set.find_planning_problem_by_id(101)
initial_state_v1 = planning_problem_v1.initial_state
initial_state_v2 = planning_problem_v2.initial_state


state_record = []
# %% obca optimization
if_comm_delay = 0
min_dis = 1
optimizer = centralized.OBCAOptimizer()
optimizer.get_static_obs_state(scenario.static_obstacles)
init_state = [arr[0, :] for arr in optimizer.ref_traj]
init_state = np.vstack(init_state)
state_record = [init_state]
init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1
for t_step in range(int(optimizer.T/optimizer.dt - optimizer.N_horz)): 
    optimizer.update_obs_state(t_step, scenario.dynamic_obstacles)
    if t_step == 19:
        print('')
    optimizer.initialize(t_step, init_state, max_x=200, max_y=20, 
                         prob=if_comm_delay, min_dis=min_dis)
    optimizer.build_model() 
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
        
    # iterate to the next time step
    init_state = np.concatenate((x_opt[:, 1].reshape((-1, 1)), 
                                 y_opt[:, 1].reshape((-1, 1)), 
                                 v_opt[:, 1].reshape((-1, 1)),
                                 heading_opt[:, 1].reshape((-1, 1)),
                                 steer_opt[:, 1].reshape((-1, 1))), axis=1)
    state_record += [init_state]
    init_state = np.reshape(init_state, [optimizer.num_veh*optimizer.n_states, 1]) # 10 X 1

# get the vehicle states
v1x = [vec[0, 0] for vec in state_record]
v1y = [vec[0, 1] for vec in state_record]
v1v = [vec[0, 2] for vec in state_record]
v1theta = [vec[0, 3] for vec in state_record]

v2x = [vec[1, 0] for vec in state_record]
v2y = [vec[1, 1] for vec in state_record]
v2v = [vec[1, 2] for vec in state_record]
v2theta = [vec[1, 3] for vec in state_record]

# %% Visualization
from commonroad.visualization.draw_params import DynamicObstacleParams
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import PMState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3

# generate state list of the ego vehicle's trajectory -- No1
state_list_v1 = [initial_state_v1]
orientation_v1 = initial_state_v1.orientation
for i in range(1, len(v1x)):
    # compute new position
    # add new state to state_list
    state_list_v1.append(PMState(**{'position': np.array([v1x[i], v1y[i]]),
                               'time_step': i, 'velocity': v1v[i]*np.cos(v1theta[i]),
                               'velocity_y': v1v[i]*np.sin(v1theta[i])}))

# create the planned trajectory starting at time step 1
ego_v1_trajectory = Trajectory(initial_time_step=1, state_list=state_list_v1[1:])

# generate state list of the ego vehicle's trajectory -- No2
state_list_v2 = [initial_state_v2]
orientation_v2 = initial_state_v2.orientation
for i in range(1, len(v1x)):
    # compute new position
    # add new state to state_list
    state_list_v2.append(PMState(**{'position': np.array([v2x[i], v2y[i]]),
                               'time_step': i, 'velocity': v2v[i]*np.cos(v2theta[i]),
                               'velocity_y': v2v[i]*np.sin(v2theta[i])}))

# create the planned trajectory starting at time step 1
ego_v2_trajectory = Trajectory(initial_time_step=1, state_list=state_list_v2[1:])

# create the prediction using the planned trajectory and the shape of the ego vehicle
vehicle3 = parameters_vehicle3.parameters_vehicle3()
ego_v1_shape = Rectangle(length=optimizer.L, width=optimizer.W)
ego_v1_prediction = TrajectoryPrediction(trajectory=ego_v1_trajectory,
                                              shape=ego_v1_shape)

# the ego vehicle can be visualized by converting it into a DynamicObstacle
ego_v1_type = ObstacleType.CAR
ego_v1 = DynamicObstacle(obstacle_id=100, obstacle_type=ego_v1_type,
                              obstacle_shape=ego_v1_shape, initial_state=initial_state_v1,
                              prediction=ego_v1_prediction)

# create the prediction using the planned trajectory and the shape of the ego vehicle
vehicle3 = parameters_vehicle3.parameters_vehicle3()
ego_v2_shape = Rectangle(length=optimizer.L, width=optimizer.W)
ego_v2_prediction = TrajectoryPrediction(trajectory=ego_v2_trajectory,
                                              shape=ego_v2_shape)

# the ego vehicle can be visualized by converting it into a DynamicObstacle
ego_v2_type = ObstacleType.CAR
ego_v2 = DynamicObstacle(obstacle_id=100, obstacle_type=ego_v2_type,
                              obstacle_shape=ego_v2_shape, initial_state=initial_state_v2,
                              prediction=ego_v2_prediction)

# plot the scenario and the ego vehicle for each time step
ego_v1_params = DynamicObstacleParams()
ego_v1_params.vehicle_shape.occupancy.shape.facecolor = "g"
ego_v2_params = DynamicObstacleParams()
ego_v2_params.vehicle_shape.occupancy.shape.facecolor = "g"

for i in range(0, len(v1x)):
    plt.figure(figsize=(25, 10))
    rnd = MPRenderer()
    rnd.draw_params.time_begin = i
    scenario.draw(rnd)
    ego_v1_params.time_begin = i
    ego_v1.draw(rnd, draw_params=ego_v1_params)
    ego_v2_params.time_begin = i
    ego_v2.draw(rnd, draw_params=ego_v2_params)
    
    planning_problem_set.draw(rnd)
    rnd.render()


# # visualization
# fig, ax = plt.subplots(figsize=(20, 4))

# # plot the vehicle center trajectories


# ax.plot(v1x, v1y, 'go', ms=10, label='vehicle1 path')
# ax.plot(v2x, v2y, 'ro', ms=10, label='vehicle2 path')
# ax.set_xlim(0,100)

# # plot the vehicle polygons
# for i in range(len(state_record)):
#     v1_verts = centralized.generate_vehicle_vertices(state_record[i][0, :])
#     plot_polygon(v1_verts, fill=False, linewidth=5, color='b')
#     v2_verts = centralized.generate_vehicle_vertices(state_record[i][1, :])
#     plot_polygon(v2_verts, fill=False, linewidth=5, color='r')

# plt.show()

