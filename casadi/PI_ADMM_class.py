# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:37:17 2023

@author: ladmin
"""
import numpy as np
from bunch import Bunch
# from casadi import *
import casadi as ca

class PI_ADMM_CASADI(object):
    def __init__(self):
        # ADMM parameter setting
        self.param = Bunch(dt = 0.1,# 4 seconds, 5m/s
                           Nt = 5, 
                           L = 1, 
                           num_ho = 15, 
                           num_veh = 2,
                           dis_thres = 2, 
                           spd = np.array([4, 8]), 
                           beta = 10,
                           Pnorm = 1, 
                           Pcost = 1, 
                           iter_num = 100, 
                           rho = 2, 
                           eps_pri = 20,   # assign fixed threshold for now
                           eps_dual = 1)
        
        # reference trajectory of vehicles, each size: 2 * N_step
        Nt = self.param.Nt
        dt = self.param.dt
        x_A = np.linspace(-10, 10, int(Nt/dt))
        y_B = np.linspace(20, -20, int(Nt/dt))
        self.ref_traj_A = np.vstack((x_A, np.zeros_like(x_A)))
        self.ref_traj_B = np.vstack((np.zeros_like(y_B), y_B))
        self.ref_traj = np.vstack((self.ref_traj_A, self.ref_traj_B))
        
        
    # %% dynamic update function --- at vehicle
    # xt - current vehicles position (pos_x, pos_y, theta),     size: N_vehicle * 3
    # u - control input, steering angle,    size: N_vehicle * time_horizon
    # x_pred - prediction x trajectory of all vehicles in finite time
    # horizons,     size: N_vehicle * time_horizon+1 (consider current pos)
    def dynamic_update_local(self, xt, u, veh_index, if_SX):
        const_spd = self.param.spd[veh_index]
        if if_SX:   # if define a casadi symbolic value
            x_pred = ca.SX.zeros(self.param.num_ho+1)
            y_pred = ca.SX.zeros(self.param.num_ho+1)
            theta_pred = ca.SX.zeros(self.param.num_ho+1)
        else:   # if do the dynamic update in numerical value
            x_pred = np.zeros(self.param.num_ho+1)
            y_pred = np.zeros(self.param.num_ho+1)
            theta_pred = np.zeros(self.param.num_ho+1)
            
        x_pred[0] = xt[0]
        y_pred[0] = xt[1]
        theta_pred[0] = xt[2]
        for k in range(self.param.num_ho):
            # calculate linearized x_dot, y_dot and theta_dot, estimate the trajectory
            x_dot = -const_spd*np.sin(xt[2])*theta_pred[k] + \
                (const_spd*np.cos(xt[2]) + const_spd*xt[2]*np.sin(xt[2]))
            x_pred[k+1] = x_pred[k] + x_dot*self.param.dt
            y_dot = const_spd*np.cos(xt[2])*theta_pred[k] + \
                (const_spd*np.sin(xt[2]) - const_spd*xt[2]*np.cos(xt[2]))
            y_pred[k+1] = y_pred[k] + y_dot*self.param.dt
            # casadi is troublesome, matrix 1X5 regarded as vector by default
            theta_dot = const_spd/self.param.L*u[0][k] if if_SX==0 else const_spd/self.param.L*u[k]
            theta_pred[k+1] = theta_pred[k] + theta_dot*self.param.dt
        return x_pred, y_pred, theta_pred
    
    # %% dynamic update function --- at RSU
    # xt - current vehicles position (pos_x, pos_y, theta), size: N_vehicle * 3
    # u - control input, steering angle, size: N_vehicle * time_horizon
    # x_pred - prediction x trajectory of all vehicles in finite time horizons,
    # size: N_vehicle * time_horizon+1 (consider current pos)
    def dynamic_update_edge(self, xt, u, if_SX):
        
        if if_SX:   # if define a casadi symbolic value
            x_pred = ca.SX.zeros((self.param.num_veh, self.param.num_ho+1))
            y_pred = ca.SX.zeros((self.param.num_veh, self.param.num_ho+1))
            theta_pred = ca.SX.zeros((self.param.num_veh, self.param.num_ho+1))
        else:   # if do the dynamic update in numerical value
            x_pred = np.zeros((self.param.num_veh, self.param.num_ho+1))
            y_pred = np.zeros((self.param.num_veh, self.param.num_ho+1))
            theta_pred = np.zeros((self.param.num_veh, self.param.num_ho+1))
        
        x_pred[:, 0] = xt[:, 0]
        y_pred[:, 0] = xt[:, 1]
        theta_pred[:, 0] = xt[:, 2]
        for k in range(self.param.num_ho):
            for i_veh in range(self.param.num_veh):
                const_spd = self.param.spd[i_veh]  # velocity of vehicle
                # calculate linearized x_dot, y_dot and theta_dot, estimate the trajectory
                x_dot = -const_spd*np.sin(theta_pred[i_veh, k])*theta_pred[i_veh, k] + \
                    (const_spd*np.cos(theta_pred[i_veh, k]) +
                     const_spd*theta_pred[i_veh, k]*np.sin(theta_pred[i_veh, k]))
                x_pred[i_veh, k+1] = x_pred[i_veh, k] + x_dot*self.param.dt
                y_dot = const_spd*np.cos(theta_pred[i_veh, k])*theta_pred[i_veh, k] + \
                    (const_spd*np.sin(theta_pred[i_veh, k]) -
                     const_spd*theta_pred[i_veh, k]*np.cos(theta_pred[i_veh, k]))
                y_pred[i_veh, k+1] = y_pred[i_veh, k] + y_dot*self.param.dt
                theta_dot = const_spd/self.param.L*u[i_veh, k]
                theta_pred[i_veh, k+1] = theta_pred[i_veh, k] + theta_dot*self.param.dt
        return x_pred, y_pred, theta_pred
    
    
    # %% cost function for the primal variable
    # num_step - current time step number
    # u - control input, steering angle, size: 1 * time_horizon
    # prev_pred_pos - 2 * N_horizon
    # hat_pos - edge variables, size: 2 * time_horizon+1
    # dual_var - dual variables, size: 2 * time_horizon+1
    def cost_function_primal(self, num_step, u, xt, hat_pos, dual_var, veh_index):
        # ref_val_x: x reference trajectory of the vehicle-i in the future time horizon, size: 1 * time_horizon+1
        ref_val_x = ca.SX(self.ref_traj[2*veh_index, num_step: num_step+self.param.num_ho+1])
        ref_val_y = ca.SX(self.ref_traj[2*veh_index+1, num_step: num_step+self.param.num_ho+1])
        # x_pred - prediction x trajectory of all vehicles in finite time horizons, size: 1 * time_horizon+1 (consider current pos)
        x_pred, y_pred, _ = self.dynamic_update_local(xt[veh_index, :], u, veh_index, if_SX=1)
        # limit the lane change of each vehicle
        cost_norm = self.param.Pnorm*ca.sum1((ref_val_x - x_pred)**2 + (ref_val_y - y_pred)**2)
        # a smooth steering input requirement
        cost_smooth = ca.sumsqr(u[2: ] - 2*u[1: -1] + u[: -2])
        # augmented Lagrangian term
        cost_AL = ca.SX(0)
        for i in range(self.param.num_veh):
            if i != veh_index:
                cost_AL += self.param.rho/2*ca.sumsqr(ca.transpose(ca.horzcat(x_pred, y_pred)) 
                                                      - hat_pos[veh_index, i] + dual_var[veh_index, i])
        # add additional term to minimize u term for fast convergence
        cost_u = self.param.Pcost*ca.sumsqr(u)
        # total cost
        cost = cost_norm + cost_smooth + cost_AL + cost_u
        
        return cost
    
    # %% cost function for the dual variable
    # estimate the cost for the control input u - this is for the edge side, only the collision avoidance is considered
    # hat_u - control input, steering angle, size: N_vehicle * time_horizon
    # prev_pred_pos - 2 * N_horizon
    # pos_old - local variables, size: 2*N_vehicle * time_horizon+1
    # dual_var_old - dual variables, size: 2*N_vehicle * time_horizon+1
    # prev_pred_pos - predicted trajectory from the last time step iteration, size: N_vehicle * (2*time_horizon)
    # edge_pos - used for AL calculation, size: 2*N_vehicle * time_horizon+1
    def cost_function_edge(self, hat_u, xt, pos_old, dual_var_old, prev_pred_pos):
        # apply the dynamic constraint
        x_pred, y_pred, _ = self.dynamic_update_edge(xt, hat_u, if_SX=1)
        
        last_dis = prev_pred_pos[1] - prev_pred_pos[0]
        curr_dis = ca.vertcat(x_pred[1, 1:] - x_pred[0, 1:], y_pred[1, 1:] - y_pred[0, 1:])
        dis_temp = 2*(last_dis[0]*curr_dis[0, :] + last_dis[1]*curr_dis[1, :]) - np.sum(np.power(last_dis, 2))
        
        edge_pos = ca.SX.zeros(2*2, self.param.num_ho+1)
        # punishment for collision avoidance failure
        max_zero_vec = ca.fmax(0, self.param.dis_thres**2 - dis_temp)
        cost_punish = self.param.beta*ca.sum2(max_zero_vec)
        
        # update the edge_pos (4 X T+1)
        edge_pos[0, :] = x_pred[0, :]
        edge_pos[2, :] = x_pred[1, :]
        edge_pos[1, :] = y_pred[0, :]
        edge_pos[3, :] = y_pred[1, :]

        # add additional term to minimize u term for fast convergence
        cost_u = self.param.Pcost*ca.sumsqr(hat_u)
        cost_AL = self.param.rho/2*ca.sumsqr(pos_old - edge_pos + dual_var_old)
        cost = cost_punish + cost_u + cost_AL
        
        return cost
    
    # %% define constraints
    def nonlcon_function(self, u):
        ineq1 = u + np.pi/6*ca.SX.ones(u.shape)
        ineq2 = -u + np.pi/6*ca.SX.ones(u.shape)
        # control input limits, -30 to 30 degrees
        # ineq3 = ca.horzcat(u[:, 1: ]-u[:, : -1] - np.pi/9, ca.SX.zeros(u.shape[0], 1))
        # ineq4 = ca.horzcat(u[:, : -1]-u[:, 1: ] - np.pi/9, ca.SX.zeros(u.shape[0], 1))
        
        ineq3 = -(u[:, 1: ]-u[:, : -1]) + np.pi/9
        ineq4 = -(u[:, : -1]-u[:, 1: ]) + np.pi/9
        
        # equality constraints, and inequality constraints
        ceq = []
        
        # 
        reineq1 = ca.reshape(ineq1, -1, 1)
        reineq2 = ca.reshape(ineq2, -1, 1)
        reineq3 = ca.reshape(ineq3, -1, 1)
        reineq4 = ca.reshape(ineq4, -1, 1)
        cieq = ca.vertcat(reineq1, reineq2, reineq3, reineq4) # >= 0
        # cieq = ca.vertcat(ineq1, ineq2, ineq3, ineq4) >= 0
        return ceq, cieq
    
    
    
