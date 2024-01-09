from .veh_config import VehicleConfig
import casadi as ca
import numpy as np
from math import *
from .util import compute_square_halfspaces_ca, generate_vehicle_vertices, \
    compute_square_halfspaces_ca_rot, compute_square_halfspaces_ca_prob

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
        self.N_horz = 20 # control horizon

    def initialize(self, t_step, init_state, max_x, max_y, prob, min_dis=1):
        self.constrains = []
        self.x0 = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self.variable = []
        self.init_state = ca.SX(init_state) # the current state at each time step, numerical value, num_veh X n_state
        
        for i_t in range(self.N_horz):
            x0_temp = []
            for i_veh in range(self.num_veh):
                x0_temp = np.hstack([x0_temp, self.ref_traj[i_veh][t_step+i_t]])
            self.x0 += [x0_temp] # state variable initialization, N_horz X num_veh*n_state
        self.x0 += [[0]*(self.num_veh*self.n_controls*(self.N_horz-1))] # control variable initialization
        self.x0 += [[0.1]*(self.num_veh*self.n_dual_variable*(self.N_horz-1))] # TODO: check dual variable number, dual variable initialization
        self.max_x = max_x
        self.max_y = max_y
        self.prob = prob
        self.min_dis = min_dis

    def build_model(self) -> bool:
        # state function is based on 'a distributed multi-robot coordination ...' eq.(7)-(8)
        # state variable
        x = ca.SX.sym('x', self.num_veh)
        y = ca.SX.sym('y', self.num_veh)
        v = ca.SX.sym('v', self.num_veh)
        theta = ca.SX.sym('theta', self.num_veh)
        steering = ca.SX.sym('steering', self.num_veh)
        # control variable
        a = ca.SX.sym('a', self.num_veh)
        steering_rate = ca.SX.sym('steering_rate', self.num_veh)
        self.state = ca.vertcat(x, y, v, theta, steering) # 5*num_veh X 1
        self.control = ca.vertcat(a, steering_rate) # 2*num_veh X 1
        beta = ca.atan(self.lr*ca.tan(steering)/(self.lr+self.lf))
        # rhs is delta_state/delta_t
        self.rhs = ca.vertcat(ca.vertcat(v*ca.cos(theta+beta), v*ca.sin(theta+beta),
                                    a, v/(self.lr+self.lf)*ca.cos(beta)*ca.tan(steering)), steering_rate) # TODO: doublecheck the last-but-two term， 5*num_veh X 1 
#                                   a, v/self.lr*ca.sin(beta)), steering_rate)
        self.f = ca.Function('f', [self.state, self.control], [self.rhs])
        self.X = ca.SX.sym('X', self.num_veh, self.n_states, self.N_horz) # state and ctrl variable indicates the global optimization
        self.U = ca.SX.sym('U', self.num_veh, self.n_controls, self.N_horz-1)
        self.LAMBDA = ca.SX.sym('LAMBDA', self.num_veh, self.n_dual_variable, self.N_horz-1)
        self.obj = 0
        return True

    def solve(self):
        nlp_prob = {'f': self.obj, 'x': ca.vertcat(*self.variable),
                    'g': ca.vertcat(*self.constrains)}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob) # TODO: sqp
        sol = solver(x0=ca.vertcat(*self.x0), lbx=self.lbx, ubx=self.ubx,
                     ubg=self.ubg, lbg=self.lbg)
        u_opt = sol['x']
        # get optimized result
        state_num = self.num_veh*self.n_states*self.N_horz
        action_num = self.num_veh*self.n_controls*(self.N_horz-1)
        self.x_opt = ca.reshape(u_opt[0: state_num: self.n_states], 
                                self.num_veh, self.N_horz)
        self.y_opt = ca.reshape(u_opt[1: state_num: self.n_states], 
                                self.num_veh, self.N_horz)
        self.v_opt = ca.reshape(u_opt[2: state_num: self.n_states],
                                self.num_veh, self.N_horz)
        self.theta_opt = ca.reshape(u_opt[3: state_num: self.n_states],
                                    self.num_veh, self.N_horz)
        self.steer_opt = ca.reshape(u_opt[4: state_num: self.n_states], 
                                    self.num_veh, self.N_horz)
        self.a_opt = ca.reshape(u_opt[state_num: state_num+action_num: self.n_controls], 
                                    self.n_controls, self.N_horz-1)
        self.steerate_opt = ca.reshape(u_opt[state_num+1: state_num+action_num: self.n_controls],
                                       self.n_controls, self.N_horz-1)
        self.lambda_result = u_opt[state_num+action_num: ]

    def generate_object(self, r, q):
        R = ca.SX(r)
        Q = ca.SX(q)
        # objective is to minimize the input effort and 
        # deviation to reference trajectory in predicted horizon
        for i in range(1, self.N_horz):
            st = ca.horzcat(self.X[i][0, :], self.X[i][1, :]).T # TODO: think about later how to cat more vehicles
            ref_st = self.x0[i]
            error = st - ref_st # TODO: check dimension during test (10 X 1)
            con = ca.reshape(self.U[i-1], self.n_controls*self.num_veh, 1)
            self.obj += (con.T@R@con)
            self.obj += (error.T@Q@error)

    def generate_variable(self):
        # variables are states
        for i in range(self.N_horz):
            for j in range(self.num_veh):    
                self.variable += [self.X[i][j, :].T]
                self.lbx += [0, -self.max_y, -self.v_cfg.max_v, -2*pi, -self.v_cfg.max_front_wheel_angle]
                self.ubx += [self.max_x, self.max_y, self.v_cfg.max_v, 2*pi, self.v_cfg.max_front_wheel_angle]
            
        # variables are inputs
        for i in range(self.N_horz-1):
            for j in range(self.num_veh):
                self.variable += [self.U[i][j, :].T]
                self.lbx += [-self.v_cfg.max_acc, -self.v_cfg.max_steer_rate]
                self.ubx += [self.v_cfg.max_acc, self.v_cfg.max_steer_rate]       
        
        # %%
        # for i in range(self.N_horz-1):
        #     self.variable += [ca.reshape(self.U[i], self.num_veh*self.n_controls, 1)]
        #     lbx_temp = [-self.v_cfg.max_acc, -self.v_cfg.max_steer_rate]*self.num_veh
        #     ubx_temp = [self.v_cfg.max_acc, self.v_cfg.max_steer_rate]*self.num_veh
        #     self.lbx += [lbx_temp[row][col] for col in range(self.n_controls) for row in range(self.num_veh)]
        #     self.ubx += [ubx_temp[row][col] for col in range(self.n_controls) for row in range(self.num_veh)]
        
        # %% variables are dual variables
        for i in range(self.N_horz-1):
            for j in range(self.num_veh):
                self.variable += [self.LAMBDA[i][j, :].T]
                self.lbx += [0.0, 0.0, 0.0, 0.0]
                self.ubx += [100000, 100000, 100000, 100000]
                

    def generate_constrain(self):
        # constraints for state transition functions
        # specify the initial position, num_veh X N_horz
        self.constrains += [ca.reshape(self.X[0].T, (self.num_veh*self.n_states, 1))
                            -self.init_state]
        self.lbg += [0, 0, 0, 0, 0]*self.num_veh
        self.ubg += [0, 0, 0, 0, 0]*self.num_veh
        
        # state transition via Euler approximation, num_veh*N_horz X 1
        for i in range(self.N_horz-1):
            st = ca.reshape(self.X[i], (self.num_veh*self.n_states, 1))
            con = ca.reshape(self.U[i], (self.num_veh*self.n_controls, 1))
            f_value = self.f(st, con) # using ca.Function to represent the intermidiate terms
            st_next_euler = st+self.dt*f_value
            st_next = ca.reshape(self.X[i+1], (self.num_veh*self.n_states, 1))
            self.constrains += [st_next-st_next_euler]
            self.lbg += [0, 0, 0, 0, 0]*self.num_veh
            self.ubg += [0, 0, 0, 0, 0]*self.num_veh
        
        # constraints for collision avoidance
        for i_tstep in range(1, self.N_horz): # iterate each predicted time horizon
            A = []
            b = []
            lamb = []
            # generate polytopic set if vehicles at i_tstep
            for i_veh in range(self.num_veh):
                # if consider the communication delay, calcuate the halfspace expression by rotational &
                # translational matrix instead of the edge points
                if self.prob:
                    A_temp, b_temp = compute_square_halfspaces_ca_prob(self.X[i_tstep][i_veh, :])
                else:
                    veh_vertices = generate_vehicle_vertices(self.X[i_tstep][i_veh, :]) # 4 x 2
                    A_temp, b_temp = compute_square_halfspaces_ca(veh_vertices)
                A += [A_temp] # 4 X 2
                b += [b_temp] # 4 X 1
                lamb += [self.LAMBDA[i_tstep-1][i_veh, :].T] # 4 X 1
            # TODO: think about later how to formulate with multiple vehicles
            # for 2 vehicles: 1 pair
            self.constrains += [-b[0].T@lamb[0]-b[1].T@lamb[1]] # (4a)
            self.lbg += [self.min_dis] # minimal distance req.
            self.ubg += [1000]
            self.constrains += [A[0].T@lamb[0]+A[1].T@lamb[1]] # (4b, 4c)
            self.lbg += [0, 0]
            self.ubg += [0, 0]
            self.constrains += [ca.mtimes((A[0].T@lamb[0]).T, A[0].T@lamb[0])] # dual norm of \lambda.T@A <= 1
            self.lbg += [0]
            self.ubg += [1]