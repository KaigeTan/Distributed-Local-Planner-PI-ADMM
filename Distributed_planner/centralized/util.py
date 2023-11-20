from math import *
import casadi as ca
from .veh_config import VehicleConfig
import numpy as np

def normalize_angle(angle):
    a = fmod(fmod(angle, 2.0*pi) + 2.0*pi, 2.0*pi)
    if a > pi:
        a -= 2.0 *pi
    return a

def generate_vehicle_vertices(state_vec, vehicle_config=VehicleConfig(), base_link=False):
    x = state_vec[0]
    y = state_vec[1]
    heading = state_vec[3]
    L = vehicle_config.length
    W = vehicle_config.width
    b_to_f = vehicle_config.baselink_to_front
    b_to_r = vehicle_config.baselink_to_rear

    vertice_x = []
    vertice_y = []
    if(base_link):
        vertice_x = [x + b_to_f*ca.cos(heading) - W/2*ca.sin(heading),
                     x + b_to_f*ca.cos(heading) + W/2*ca.sin(heading),
                     x - b_to_r*ca.cos(heading) + W/2*ca.sin(heading),
                     x - b_to_r*ca.cos(heading) - W/2*ca.sin(heading)]

        vertice_y = [y + b_to_f*ca.sin(heading) + W/2*ca.cos(heading),
                     y + b_to_f*ca.sin(heading) - W/2*ca.cos(heading),
                     y - b_to_r*ca.sin(heading) - W/2*ca.cos(heading),
                     y - b_to_r*ca.sin(heading) + W/2*ca.cos(heading)]
    else:
        vertice_x = [x + L/2*ca.cos(heading) - W/2*ca.sin(heading),
                     x + L/2*ca.cos(heading) + W/2*ca.sin(heading),
                     x - L/2*ca.cos(heading) + W/2*ca.sin(heading),
                     x - L/2*ca.cos(heading) - W/2*ca.sin(heading)]

        vertice_y = [y + L/2*ca.sin(heading) + W/2*ca.cos(heading),
                     y + L/2*ca.sin(heading) - W/2*ca.cos(heading),
                     y - L/2*ca.sin(heading) - W/2*ca.cos(heading),
                     y - L/2*ca.sin(heading) + W/2*ca.cos(heading)]

    V = np.vstack((vertice_x, vertice_y)).T

    return V

def compute_square_halfspaces_ca_prob(state_vec, vehicle_config=VehicleConfig(), base_link=False):
    x = state_vec[0]
    y = state_vec[1]
    v = state_vec[2]
    heading = state_vec[3]
    L = vehicle_config.length
    W = vehicle_config.width
    b_to_f = vehicle_config.baselink_to_front
    b_to_r = vehicle_config.baselink_to_rear
    prob = vehicle_config.prob
    
    delay_avg = vehicle_config.avg_delay
    delay_var = vehicle_config.var_delay
    delta_x_avg = delay_avg*v*ca.cos(heading)
    delta_y_avg = delay_avg*v*ca.sin(heading)
    delta_avg = ca.vertcat(delta_x_avg, delta_y_avg)
    delta_x_var = (delay_var*v*ca.cos(heading))*(delay_var*v*ca.cos(heading))
    delta_y_var = (delay_var*v*ca.sin(heading))*(delay_var*v*ca.sin(heading))
    delta_var = ca.vertcat(delta_x_var, delta_y_var)
    pos0 = ca.vertcat(x, y)
    
    # write the full-dimensional space by agent with A*R*(x-x0) <= b    
    if(~base_link):
        b0 = ca.DM([L/2, W/2, L/2, W/2])
    else:
        b0 = ca.DM([b_to_f, W/2, b_to_r, W/2])
    R_mat = ca.vertcat(ca.horzcat(ca.cos(heading), -ca.sin(heading)),
                       ca.horzcat(ca.sin(heading), ca.cos(heading)))
    A_mat = ca.vertcat(R_mat.T, -R_mat.T)
    b_mat = b0 + A_mat@(pos0 + delta_avg + sqrt(prob/(1-prob))*delta_var)
        
    return A_mat, b_mat

def compute_square_halfspaces_ca_rot(state_vec, vehicle_config=VehicleConfig(), base_link=False):
    x = state_vec[0]
    y = state_vec[1]
    v = state_vec[2]
    heading = state_vec[3]
    L = vehicle_config.length
    W = vehicle_config.width
    b_to_f = vehicle_config.baselink_to_front
    b_to_r = vehicle_config.baselink_to_rear
    prob = vehicle_config.prob
    
    pos0 = ca.vertcat(x, y)
    
    # write the full-dimensional space by agent with A*R*(x-x0) <= b    
    if(~base_link):
        b0 = ca.DM([L/2, W/2, L/2, W/2])
    else:
        b0 = ca.DM([b_to_f, W/2, b_to_r, W/2])
    # https://github.com/RoyaFiroozi/Centralized-Planning/blob/master/rotation_translation.m
    R_mat = ca.vertcat(ca.horzcat(ca.cos(heading), -ca.sin(heading)),
                       ca.horzcat(ca.sin(heading), ca.cos(heading)))
    A_mat = ca.vertcat(R_mat.T, -R_mat.T)
    b_mat = b0 + A_mat@(pos0)
        
    return A_mat, b_mat

def compute_square_halfspaces_ca(points):
    # Calculate the center of the square
    center = ca.sum1(points)/4
    # Compute the half-space representation for a square
    A = []
    b = []
    
    for i in range(4):
        p1 = points[i, :]
        p2 = points[(i + 1) % 4, :]

        A_temp = ca.vertcat(p1[1]-p2[1], p2[0]-p1[0])
        normal = ca.norm_2(ca.vertcat(p2[1]-p1[1], p1[0]-p2[0]))
        A_temp = A_temp/normal
        b_temp = (p2[0]*p1[1]-p2[1]*p1[0])/normal


        A.append(A_temp.T)
        b.append(b_temp)
    
    return ca.vertcat(*A), ca.vertcat(*b)