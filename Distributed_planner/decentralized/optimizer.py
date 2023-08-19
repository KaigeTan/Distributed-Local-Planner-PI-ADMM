from .veh_config import VehicleConfig
import casadi as ca
import numpy as np
from math import *
from .util import compute_square_halfspaces_ca, generate_vehicle_vertices

class OBCAOptimizer:
    def __init__(self, cfg: VehicleConfig = VehicleConfig()) -> None:
        self.L = cfg.length
        self.offset = cfg.length/2 - cfg.baselink_to_rear
        self.lf = cfg.lf
        self.lr = cfg.lr
        self.v_cfg = cfg
        self.n_controls = 2
        self.n_states = 5 # x, y, v, heading angle, steer angle
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

    # %% below are the functions used at the local vehicle side
    def local_initialize(self, t_step, init_state, bar_state, i_veh, max_x, max_y):
        
        self.constrains = []
        self.x0 = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.variable = []
        self.init_state = ca.SX(init_state) # the current state at each time step, numerical value, num_veh X n_state
        
        for i_t in range(self.N_horz):
            x0_temp = self.ref_traj[i_veh][t_step+i_t]
            self.x0 += [x0_temp] # state variable initialization, N_horz X n_state
        self.x0 += [[0]*(self.n_controls*(self.N_horz-1))] # control variable initialization
        self.max_x = max_x
        self.max_y = max_y
        self.veh_idx = i_veh
        self.bar_state = bar_state

    def local_build_model(self) -> bool:
        
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        theta = ca.SX.sym('theta')
        steering = ca.SX.sym('steering')
        a = ca.SX.sym('a')
        steering_rate = ca.SX.sym('steering_rate')
        self.state = ca.vertcat(ca.vertcat(x, y, v, theta), steering) # 5 X 1
        self.control = ca.vertcat(a, steering_rate) # 2 X 1
        beta = ca.atan(self.lr*ca.tan(steering)/(self.lr+self.lf))
        self.rhs = ca.vertcat(ca.vertcat(v*ca.cos(theta+beta), v*ca.sin(theta+beta),
                                   a, v/self.lr*ca.sin(beta)), steering_rate) # 5 X 1, TODO: doublecheck the last-but-two term
        self.f = ca.Function('f', [self.state, self.control], [self.rhs])
        self.X = ca.SX.sym('X', self.n_states, self.N_horz) # state and ctrl variable indicates the global optimization
        self.U = ca.SX.sym('U', self.n_controls, self.N_horz-1)
        self.obj = 0

        return True

    def local_solve(self):
        nlp_prob = {'f': self.obj, 'x': ca.vertcat(*self.variable),
                    'g': ca.vertcat(*self.constrains)}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob)
        sol = solver(x0=ca.vertcat(*self.x0), lbx=self.lbx, ubx=self.ubx,
                     ubg=self.ubg, lbg=self.lbg)
        u_opt = sol['x']
        # get optimized result
        state_num = self.n_states*self.N_horz
        action_num = self.n_controls*(self.N_horz-1)
        self.x_opt = u_opt[0: state_num: self.n_states]
        self.y_opt = u_opt[1: state_num: self.n_states]
        self.v_opt = u_opt[2: state_num: self.n_states]
        self.theta_opt = u_opt[3: state_num: self.n_states]
        self.steer_opt = u_opt[4: state_num: self.n_states]
        self.a_opt = u_opt[state_num: state_num+action_num: self.n_controls]
        self.steerate_opt = u_opt[state_num+1: state_num+action_num: self.n_controls]
        self.bar_z = ca.horzcat(self.x_opt, self.y_opt, self.theta_opt)

    def local_generate_object(self, r, q):
        R = ca.SX(r)
        Q = ca.SX(q)
        # objective is to minimize the input effort and 
        # deviation to reference trajectory in predicted horizon
        for i in range(1, self.N_horz):
            st = self.X[:, i]
            ref_st = self.x0[i]
            error = st - ref_st
            con = self.U[:, i-1]

            self.obj += (con.T@R@con) # control input effort
            self.obj += (error.T@Q@error) # deviation to reference trajectory

    def local_generate_variable(self):
        
        # variables are states
        for i in range(self.N_horz):
            self.variable += [self.X[:, i]]
            self.lbx += [0, -self.max_y, -self.v_cfg.max_v, -2*pi, -self.v_cfg.max_front_wheel_angle]
            self.ubx += [self.max_x, self.max_y, self.v_cfg.max_v, 2*pi, self.v_cfg.max_front_wheel_angle]
            
        # variables are inputs
        for i in range(self.N_horz-1):
            self.variable += [self.U[:, i]]
            self.lbx += [-self.v_cfg.max_acc, -self.v_cfg.max_steer_rate]
            self.ubx += [self.v_cfg.max_acc, self.v_cfg.max_steer_rate]       

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
            f_value = self.f(st, con) # using ca.Function to represent the intermidiate terms
            st_next_euler = st+self.dt*f_value
            st_next = self.X[:, i+1]
            self.constrains += [st_next-st_next_euler]
            self.lbg += [0, 0, 0, 0, 0]
            self.ubg += [0, 0, 0, 0, 0]
        
        # consider later how to implement multi-vehicle collsison avoidances
        # constraints for collision avoidance
        for i_tstep in range(1, self.N_horz): # iterate each predicted time horizon
            # generate polytopic set if vehicles at i_tstep
            veh_vertices = generate_vehicle_vertices(self.X[:, i_tstep][0, 1, 3]) # 4 x 2
            A, b = compute_square_halfspaces_ca(veh_vertices)
            # TODO: think about later how to formulate with multiple vehicles
            # for 2 vehicles: 1 pair
            self.constrains += [-b.T@self.bar_state.lamb[self.veh_idx, i_tstep, :]
                                -np.dot(self.bar_state.b[1-self.veh_idx, i_tstep, :], 
                                        self.bar_state.lamb[1-self.veh_idx, i_tstep, :])] # (5a)
            self.lbg += [0.001] # minimal distance req.
            self.ubg += [1000]
            self.constrains += [A.T@self.bar_state.lamb[self.veh_idx, i_tstep, :]
                                +self.bar_state.s[i_tstep, :]] # (5b)
            self.lbg += [0, 0]
            self.ubg += [0, 0]


    # %% below are functions used at the edge coordinator side
    def edge_initialize(self, bar_z):
        
        self.constrains = []
        self.x0 = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.variable = []
        
        # get bar_A and bar_b for all predicted horizons
        self.A = []
        self.b = []
        for i_tstep in range(self.N_horz): # iterate each predicted time horizon
            for i_veh in range(self.num_veh):
                # generate polytopic set if vehicles at i_tstep
                veh_vertices = generate_vehicle_vertices(bar_z[i_veh][i_tstep, :]) # 4 x 2
                A_temp, b_temp = compute_square_halfspaces_ca(veh_vertices) 
                self.A += [A_temp]
                self.b += [b_temp]
        
        # TODO: check x0
        for i_t in range(self.N_horz):
            x0_temp = self.ref_traj[i_veh][t_step+i_t]
            self.x0 += [x0_temp] # state variable initialization, N_horz X n_state
        self.x0 += [[0]*(self.n_controls*(self.N_horz-1))] # control variable initialization
        self.max_x = max_x
        self.max_y = max_y
    
    def edge_build_model(self) -> bool:
        
        self.LAMBDA = ca.SX.sym('LAMBDA', self.num_veh, self.n_dual_variable, self.N_horz)
        self.s = ca.SX.sym('s', self.n_s, self.N_horz)
        self.obj = 0

        return True

    def edge_generate_constrain(self):
        # constraints for (6a) and (6b)
        for i_tstep in range(self.N_horz):
            self.constrains += self.A[self.N_horz*2].T@self.LAMBDA[i_tstep][0, :] + self.s[:, i_tstep] # 2 x 1
            self.lbg += [0, 0]
            self.ubg += [0, 0]
            self.constrains += self.A[1+self.N_horz*2].T@self.LAMBDA[i_tstep][1, :] - self.s[:, i_tstep]
            self.lbg += [0, 0]
            self.ubg += [0, 0]
            self.constrains += [-self.b[self.N_horz*2].T@self.LAMBDA[i_tstep][0, :]
                                -self.b[1+self.N_horz*2].T@self.LAMBDA[i_tstep][1, :]] # (6c-1)
            self.lbg += [0.001] # minimal distance req.
            self.ubg += [1000]
            self.constrains += [s[:, 0].T@s[:, 0]]
            self.lbg += [0]
            self.ubg += [1]
            
    def edge_generate_variable(self):
        
        # variables are lambda
        for i in range(self.N_horz):
            self.variable += [ca.reshape(self.LAMBDA[i], 2*self.n_dual_variable, 1)] # 8 x 1
            self.lbx += [0]*2*self.n_dual_variable
            self.ubx += [1000]*2*self.n_dual_variable
            
        # variables are s
        for i in range(self.N_horz):
            self.variable += [self.s[:, i]]
            self.lbx += [-1000000]*self.n_s
            self.ubx += [1000000]*self.n_s


    # %% define the initialized bar_state
    def create_bar_state(self):
        bar_state = self.mid_state(self)
        return bar_state
    
    class mid_state:
        def __init__(self, outer):
            self.lamb = np.zeros((outer.num_veh, outer.N_horz, outer.n_dual_variable)) # 2x10x4
            self.b = np.zeros((outer.num_veh, outer.N_horz, outer.n_dual_variable)) # 2x10x4
            self.s = np.zeros((outer.N_horz, outer.n_s)) # 10x2