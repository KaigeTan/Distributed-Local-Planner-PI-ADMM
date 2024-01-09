from .veh_config import VehicleConfig
import casadi as ca
import numpy as np
from math import *
from .util import compute_square_halfspaces_ca, generate_vehicle_vertices, \
    compute_square_halfspaces_ca_rot, compute_square_halfspaces_ca_prob
import copy

class OBCAOptimizer:
    def __init__(self, cfg: VehicleConfig = VehicleConfig()) -> None:
        self.L = cfg.length
        self.offset = cfg.length/2 - cfg.baselink_to_rear
        self.lf = cfg.lf
        self.lr = cfg.lr
        self.v_cfg = cfg
        self.n_controls = 2
        self.n_states = 5 # x, y, v, heading angle, steer angle
        self.n_loc_lambda = 4 # full dimension lambda variable
        self.n_dual_variable = 4 # dual variable dimension is same to the vertices
        self.n_s = 2
        self.num_veh = 2
        self.obstacles = []
        self.G = ca.DM([[1, 0],
                     [-1, 0],
                     [0, 1],
                     [0, -1], ])
        self.g = ca.vertcat(ca.SX([cfg.length/2, cfg.length/2,
                                   0.5*cfg.width, 0.5*cfg.width]))
        self.T = cfg.T
        self.dt = cfg.dt
        self.ref_traj = cfg.ref_traj_gen()
        self.N_horz = 10 # control horizon
        self.primal_thres = 0.01
        self.dual_thres = 0.01
        
    # %% below are the functions used at the local vehicle side
    def local_initialize(self, t_step, init_state, bar_state_prev, i_veh, max_x, max_y, prob, min_dis):
        self.constrains = []
        self.x0 = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.variable = []
        self.init_state = ca.SX(init_state) # the current state at each time step, numerical value, num_veh X n_state
        # assign the x0 with solver initialized values
        # state values
        for i_t in range(self.N_horz):
            x0_temp = self.ref_traj[i_veh][t_step+i_t]
            self.x0 += [x0_temp] # state variable initialization, N_horz X n_state
        self.x0 += [[0]*(self.n_controls*(self.N_horz-1))] # control variable initialization
        self.x0 += [[0]*(self.n_loc_lambda*(self.N_horz-1))] # local lambda values initialization
        self.max_x = max_x
        self.max_y = max_y
        self.veh_idx = i_veh
        self.bar_state = copy.deepcopy(bar_state_prev) # TODO: check isolation
        self.prob = prob
        self.min_dis = min_dis

    def local_build_model(self) -> bool:
        # state variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        theta = ca.SX.sym('theta')
        steering = ca.SX.sym('steering')
        # intermidiate varible
        self.LAMB_LOC = ca.SX.sym('lamb_loc', self.n_loc_lambda, self.N_horz-1)
        # control variables
        a = ca.SX.sym('a')
        steering_rate = ca.SX.sym('steering_rate')
        self.state = ca.vertcat(x, y, v, theta, steering) # 5 X 1
        self.control = ca.vertcat(a, steering_rate) # 2 X 1
        beta = ca.atan(self.lr*ca.tan(steering)/(self.lr+self.lf))
        self.rhs = ca.vertcat(ca.vertcat(v*ca.cos(theta+beta), v*ca.sin(theta+beta),
                                   a, v/self.lr*ca.sin(beta)), steering_rate) # 5 X 1
        self.f = ca.Function('f', [self.state, self.control], [self.rhs]) # delta part
        self.X = ca.SX.sym('X', self.state.size1(), self.N_horz) # 5 X N_horz
        self.U = ca.SX.sym('U', self.n_controls, self.N_horz-1)
        self.obj = 0
        return True

    def local_generate_constrain(self):
        # constraints for state transition functions
        
        # specify the initial position, 1 X N_horz
        self.constrains += [self.X[:, 0]-self.init_state]
        self.lbg += [0, 0, 0, 0, 0]
        self.ubg += [0, 0, 0, 0, 0]
        # state transition via Euler approximation, N_horz X 1
        for i in range(self.N_horz-1):
            st = self.X[:, i]
            con = self.U[:, i]
            f_value = self.f(st, con) # using ca.Function to represent the temp terms
            st_next_euler = st+self.dt*f_value
            st_next = self.X[:, i+1]
            self.constrains += [st_next-st_next_euler]
            self.lbg += [0, 0, 0, 0, 0]
            self.ubg += [0, 0, 0, 0, 0]
        
        # consider later how to implement multi-vehicle collsison avoidances
        # constraints for collision avoidance
        for i_tstep in range(1, self.N_horz): # iterate each predicted time horizon
            # generate polytopic set if vehicles at i_tstep
            # if consider the communication delay, calcuate the halfspace expression by rotational &
            # translational matrix instead of the edge points
            if self.prob:
                A, b = compute_square_halfspaces_ca_prob(self.X[:, i_tstep])
            else:
                veh_vertices = generate_vehicle_vertices(self.X[:, i_tstep]) # 4 x 2
                A, b = compute_square_halfspaces_ca(veh_vertices)
            # TODO: think about later how to formulate with multiple vehicles
            # for 2 vehicles: 1 pair
            self.constrains += [-b.T@self.LAMB_LOC[:, i_tstep-1] - 
                                np.dot(self.bar_state.b[1-self.veh_idx, i_tstep-1, :],\
                                       self.bar_state.lamb_ij[1-self.veh_idx, i_tstep-1, :])] # (5a)
            self.lbg += [self.min_dis] # minimal distance req.
            self.ubg += [1000] # TODO: check the range and without assignment
            self.constrains += [A.T@self.LAMB_LOC[:, i_tstep-1] + self.bar_state.A[1-self.veh_idx, i_tstep-1].T@\
                                self.bar_state.lamb_ij[1-self.veh_idx, i_tstep-1, :]] # (5b)
            self.lbg += [0, 0]
            self.ubg += [0, 0]
        
        for i_tstep in range(1, self.N_horz):
            self.constrains += [(A.T@self.LAMB_LOC[:, i_tstep-1]).T@(A.T@self.LAMB_LOC[:, i_tstep-1])]
            self.lbg += [0]
            self.ubg += [1]

    def local_generate_variable(self):
        # variables are states, 5 X N_horz
        for i in range(self.N_horz):
            self.variable += [self.X[:, i]]
            self.lbx += [0, -self.max_y, -self.v_cfg.max_v, -2*pi, -self.v_cfg.max_front_wheel_angle]
            self.ubx += [self.max_x, self.max_y, self.v_cfg.max_v, 2*pi, self.v_cfg.max_front_wheel_angle]
            
        # variables are inputs, 2 X N_horz-1
        for i in range(self.N_horz-1):
            self.variable += [self.U[:, i]]
            self.lbx += [-self.v_cfg.max_acc, -self.v_cfg.max_steer_rate]
            self.ubx += [self.v_cfg.max_acc, self.v_cfg.max_steer_rate]       
        
        # variables are local lambda, 4 X N_horz-1 (lambda_ij)
        for i in range(self.N_horz-1):
            self.variable += [self.LAMB_LOC[:, i]]
            self.lbx += [0]*self.n_loc_lambda
            self.ubx += [1000]*self.n_loc_lambda # TODO: ubx should not be specified? check the range and without assignment
    
    def local_generate_object(self, r, q, rho):
        R = ca.SX(r)
        Q = ca.SX(q)
        # objective is to minimize the input effort and 
        # deviation to reference trajectory in predicted horizon
        # also the augmented lagrangian term is added
        for i in range(1, self.N_horz):
            st = self.X[:, i]
            ref_st = self.x0[i]
            error = st - ref_st
            con = self.U[:, i-1]

            self.obj += (con.T@R@con) # control input effort
            self.obj += (error.T@Q@error) # deviation to reference trajectory
            temp_S = ca.vertcat(self.X[:, i], self.LAMB_LOC[:, i-1]) # 9 X 1
            temp_Z = self.bar_state.Z_bar[self.veh_idx, i-1, :] # 9 X 1
            temp_lambda = self.bar_state.lamb_bar[i-1, :]
            self.obj += ca.SX(temp_lambda).T@temp_S # TODO: assign lambda_bar
            self.obj += rho/2*(temp_S - temp_Z).T@(temp_S - temp_Z) # TODO: assign z_bar

    def local_solve(self):
        nlp_prob = {'f': self.obj, 'x': ca.vertcat(*self.variable),
                    'g': ca.vertcat(*self.constrains)}
        opts = {'ipopt.max_iter':100, 
                'ipopt.print_level':0, 
                'print_time':0, 
                'ipopt.acceptable_tol':1e-5, 
                'ipopt.acceptable_obj_change_tol':1e-5}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        sol = solver(x0=ca.vertcat(*self.x0), lbx=self.lbx, ubx=self.ubx,
                     ubg=self.ubg, lbg=self.lbg)
        u_opt = sol['x']
        # get optimized result
        state_num = self.n_states*self.N_horz
        action_num = self.n_controls*(self.N_horz-1)
        # state
        self.x_opt = u_opt[0: state_num: self.n_states]
        self.y_opt = u_opt[1: state_num: self.n_states]
        self.v_opt = u_opt[2: state_num: self.n_states]
        self.theta_opt = u_opt[3: state_num: self.n_states]
        self.steer_opt = u_opt[4: state_num: self.n_states]
        # control
        self.a_opt = u_opt[state_num: state_num+action_num: self.n_controls]
        self.steerate_opt = u_opt[state_num+1: state_num+action_num: self.n_controls]
        # lambda
        lambda_loc = u_opt[state_num+action_num: ]
        
        # update the bar_state
        self.bar_x = ca.horzcat(self.x_opt, self.y_opt, self.v_opt, self.theta_opt, self.steer_opt) # N_horz X 5
        self.bar_u = ca.horzcat(self.a_opt, self.steerate_opt) # N_horz-1 X 2
        self.bar_lambda_loc = ca.reshape(lambda_loc, 4, 9).T # resize to N_horz-1 X 4
        self.bar_fullx = ca.horzcat(self.bar_x[1:, :], self.bar_lambda_loc) # N_horz-1 X 9
        
    
    # update the A and b matrices based on the local calculation
    def bar_state_update(self, bar_fullx_vehs, bar_lambda):
        # update bar_A and bar_b for all predicted horizons, bar_x contains information for all vehicles
        for i_tstep in range(self.N_horz-1): # iterate each predicted time horizon
            for i_veh in range(self.num_veh):
                # update A and b
                bar_x = bar_fullx_vehs[i_veh][:, :5]
                # generate polytopic set if vehicles at i_tstep
                if self.prob:
                    A_temp, b_temp = compute_square_halfspaces_ca_prob(np.array(bar_x[i_tstep, :]).ravel())
                else:
                    veh_vertices_bar = generate_vehicle_vertices(np.array(bar_x[i_tstep, :]).ravel()) # 4 x 2
                    A_temp, b_temp = compute_square_halfspaces_ca(veh_vertices_bar)
                self.bar_state.A[i_veh, i_tstep, :, :] = np.array(A_temp) # TODO: check the horizon, 4 X 2
                self.bar_state.b[i_veh, i_tstep, :] = np.array(b_temp).ravel() # TODO: check the horizon, 4 X 1
                # update lamb_ij
                self.bar_state.lamb_ij[i_veh, i_tstep, :] = np.array(bar_fullx_vehs[i_veh][i_tstep, 5:]).ravel()
                # update local_x
                self.bar_state.local_x[i_veh, i_tstep, :] = bar_x[i_tstep, :]
    
    # %% below are functions used at the edge coordinator side
    def edge_initialize(self, max_x, max_y):
        self.constrains = []
        self.x0 = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.variable = []
        
        # TODO: check x0
        for i_t in range(1, self.N_horz):
            self.x0 += [[0]*self.num_veh*(self.n_states+self.n_loc_lambda)] # LAMBDA at a single time step, 1 X 2*self.n_dual_variable
        self.max_x = max_x
        self.max_y = max_y
    
    def edge_build_model(self) -> bool:
        self.Z = ca.SX.sym('Z', self.num_veh, self.n_states+self.n_loc_lambda, self.N_horz-1) # N_horz-1 X (2Xstate)
        self.obj = 0
        return True

    def edge_generate_constrain(self):
        # constraints for (6a) and (6b)
        # TODO: I think here A and b needs to be recalculated!!!!!
        edge_x = [vec[:, :5] for vec in self.Z] # 2X9X5
        edge_lambda = [vec[:, 5:] for vec in self.Z] # 2X9X4
        for i_tstep in range(self.N_horz-1):
            # calculate the A_temp, b_temp for both vehicles
            if self.prob:
                A_temp1, b_temp1 = compute_square_halfspaces_ca_prob(edge_x[i_tstep][0, :])
                A_temp2, b_temp2 = compute_square_halfspaces_ca_prob(edge_x[i_tstep][1, :])
            else:
                veh_vertices_bar1 = generate_vehicle_vertices(edge_x[i_tstep][0, :]) # 4 x 2
                veh_vertices_bar2 = generate_vehicle_vertices(edge_x[i_tstep][1, :]) # 4 x 2
                A_temp1, b_temp1 = compute_square_halfspaces_ca(veh_vertices_bar1) # 4x2 and 4x1
                A_temp2, b_temp2 = compute_square_halfspaces_ca(veh_vertices_bar2)
            # calculate the beta and alpha
            
            self.constrains += [A_temp1.T@edge_lambda[i_tstep][0, :].T +
                                A_temp2.T@edge_lambda[i_tstep][1, :].T] # 2 x 1
            self.lbg += [0, 0]
            self.ubg += [0, 0]
            self.constrains += [-b_temp1.T@edge_lambda[i_tstep][0, :].T
                                -b_temp2.T@edge_lambda[i_tstep][1, :].T] # (6c-1)
            self.lbg += [self.min_dis] # minimal distance req.
            self.ubg += [1000] # TODO: check the upper bound
            
    def edge_generate_variable(self):
        # variables are Z
        for i in range(self.N_horz-1):
            self.variable += [ca.reshape(self.Z[i], self.num_veh*(self.n_states+self.n_loc_lambda), 1)] # 2*(5+4) x 1
            # self.lbx += [0]*2*self.n_dual_variable # TODO: ***** check if dual vars has the lower and upper bound
            # self.ubx += [5]*2*self.n_dual_variable
            
    def edge_generate_object(self, rho):
        for i in range(self.N_horz-1):
            temp_lambda = self.bar_state.lamb_bar[i, :]
            self.obj += -self.Z[i][0, :]@temp_lambda - self.Z[i][1, :]@temp_lambda
            # TODO: double check correctness of bar_fullx
            local_fullx1 = np.concatenate((self.bar_state.local_x[0, i, :], self.bar_state.lamb_ij[0, i, :]))
            local_fullx2 = np.concatenate((self.bar_state.local_x[1, i, :], self.bar_state.lamb_ij[1, i, :]))
            temp_diff1 = local_fullx1 - self.Z[i][0, :].T # veh1
            temp_diff2 = local_fullx2 - self.Z[i][1, :].T # veh2
            self.obj += rho/2*(temp_diff1.T@temp_diff1 + temp_diff2.T@temp_diff2)
            
    def edge_solve(self):
        nlp_prob = {'f': self.obj, 'x': ca.vertcat(*self.variable),
                    'g': ca.vertcat(*self.constrains)}
        opts = {'ipopt.max_iter':100, 
                'ipopt.print_level':0, 
                'print_time':0, 
                'ipopt.acceptable_tol':1e-5, 
                'ipopt.acceptable_obj_change_tol':1e-5}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        sol = solver(x0=ca.vertcat(*self.x0), lbx=self.lbx, ubx=self.ubx,
                     ubg=self.ubg, lbg=self.lbg)
        u_opt = sol['x']
        # get optimized result
        Z_bar = np.array(np.round(u_opt, decimals=4))
        # TODO: double check the result
        # annoying, have to use two steps to split into 3d array
        Z_bar = np.reshape(Z_bar, (self.N_horz-1, self.num_veh*(self.n_states+self.n_loc_lambda))) # time-step wise
        Z_bar = np.reshape(Z_bar, (self.N_horz-1, self.num_veh, self.n_states+self.n_loc_lambda), 'F') # N_horz X num_veh X state+lambda
        # update the Z_bar for the iteration
        self.bar_state.Z_bar = np.transpose(Z_bar, (1, 0, 2))
    
    # %% define the initialized bar_state
    def create_bar_state(self):
        bar_state = self.mid_state(self)
        return bar_state
    
    class mid_state:
        def __init__(self, outer):
            self.Z_bar = np.zeros((outer.num_veh, outer.N_horz-1, outer.n_states+outer.n_loc_lambda)) # 2x10-1x9
            self.A = np.zeros((outer.num_veh, outer.N_horz-1, outer.n_dual_variable, 2)) # 2x9x4X2
            self.b = np.zeros((outer.num_veh, outer.N_horz-1, outer.n_dual_variable)) # 2x9x4
            self.lamb_bar = np.zeros((outer.N_horz-1, outer.n_states+outer.n_loc_lambda)) # 9x9
            self.lamb_ij = np.zeros((outer.num_veh, outer.N_horz-1, outer.n_dual_variable)) # 2x9x4
            self.local_x = np.zeros((outer.num_veh, outer.N_horz-1, outer.n_states)) # 2x9x5