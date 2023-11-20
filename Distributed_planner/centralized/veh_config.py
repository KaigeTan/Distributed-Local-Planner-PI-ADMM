import math as m
import numpy as np


class VehicleConfig:   

    def __init__(self) -> None:
        self.length = 3.5
        self.width =2
        self.baselink_to_front = 3
        self.baselink_to_rear = self.length - self.baselink_to_front
        self.wheel_base = 2.5
        self.lf = 1.5
        self.lr = self.wheel_base - self.lf
        self.max_front_wheel_angle = 0.6  # rad
        self.min_radius = self.wheel_base/m.tan(self.max_front_wheel_angle)
        self.dt = 0.1  # discreate time
        self.T = 5  # period time
        self.max_acc = 5
        self.max_v = 20
        self.max_steer_rate = 20
        # consider the comm. delay follows normal distribution
        # reference: 'On ramp merging strategies of connected and 
        # automated vehicles considering communication delay' TABLE-V
        self.avg_delay = 0.05 # consider the time delay is 50ms on average
        self.var_delay = 0.025 # consider the time variance is 25ms on average
        self.prob = 0.95
        
    # vehicle reference trajectory configuration, default 2 vehicles overtaking
    def ref_traj_gen(self, num_veh = 2):
        v1 = 20 # m/s
        x1 = np.linspace(0, 0+v1*self.T, int(self.T/self.dt)+1) 
        v2 = 10 # m/s
        x2 = np.linspace(20, 20+v2*self.T, int(self.T/self.dt)+1) 
        # ref trajectory, T/dt X 5: x, y, v, head, steer
        ref_traj1 = np.vstack((x1,
                               np.zeros_like(x1), 
                               v1*np.ones_like(x1),
                               np.zeros_like(x1), 
                               np.zeros_like(x1))).T
        ref_traj2 = np.vstack((x2,
                               np.zeros_like(x2), 
                               v2*np.ones_like(x2),
                               np.zeros_like(x2), 
                               np.zeros_like(x2))).T
        ref_traj = [ref_traj1, ref_traj2]    
        return ref_traj  