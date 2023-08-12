from math import *
import casadi as ca
from .search import VehicleConfig
import numpy as np

def normalize_angle(angle):
    a = fmod(fmod(angle, 2.0*pi) + 2.0*pi, 2.0*pi)
    if a > pi:
        a -= 2.0 *pi
    return a

def generate_vehicle_vertices(state_vec, vehicle_config=VehicleConfig(), base_link=False):
    x = state_vec[0]
    y = state_vec[1]
    heading = state_vec[2]
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