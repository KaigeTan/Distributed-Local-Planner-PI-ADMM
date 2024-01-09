# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:42:16 2023

@author: kaiget
"""

import casadi as ca
import numpy as np

def create_subproblem(fi_func, Ai, rho, hi_func):
    N_lambda, N_yi = np.shape(Ai)
    yi = ca.SX.sym("yi",N_yi)
    xi = ca.SX.sym("xi",N_yi)
    sigma_i = ca.SX.sym('sigma_i',N_yi,N_yi)
    lambda_ = ca.SX.sym("lambda",N_lambda)
    
    fi = fi_func(yi) + lambda_.T @ Ai @ yi + \
        rho/2 * (yi - xi).T @ sigma_i @ (yi - xi)
    p = ca.vertcat(lambda_, xi, ca.reshape(sigma_i, -1,1))
    g = hi_func(yi)
    # Define proximal solver
    solver_opt = {}
    solver_opt['print_time'] = False
    solver_opt['ipopt'] = {
        'max_iter': 500,
        'print_level': 1,
        'acceptable_tol': 1e-6,
        'acceptable_obj_change_tol': 1e-6
    }

    nlp = {'x':yi, 'g':g, 'f':fi, 'p': p}
#     print(nlp)
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opt)
    return solver

def constraint_jac_approx(yi, hi_func, hi_jac_func):
    constraint_res = hi_func(yi)    #  Residue
    Nh = np.shape(constraint_res)[0]
    Ny = np.shape(yi)[0]
    zero_row = ca.DM.zeros(1,Ny)
    hi_jac = hi_jac_func(yi)
    for i in range(Nh):
        if constraint_res[i] != 0:    #  TODO: deal with small value
            hi_jac[i,:] = zero_row
    return hi_jac

def modified_grad(fi_grad, hi_jac_approx, hi_jac_real, kappa_i):
    return fi_grad + (hi_jac_real - hi_jac_approx).T @ kappa_i
    

def create_QP_problem(A_list, b,  mu, N_hi_list):
    N = len(A_list)
    N_lambda = np.shape(A_list[0])[0]
    
    s = ca.SX.sym("s", N_lambda)
    lambda_ = ca.SX.sym("lambda_", N_lambda)
    
    delta_yi_list = []
    fkh_hess_col_list = [] 
    modiefied_grad_col_list = []
    Ci_col_list = []
    
    yi_list = []
    obj = 0
    sigma_Ai = 0

    g = []
    for i in range(N):
        Ai = A_list[i]
        N_delta_yi = np.shape(Ai)[1]
        Hi = ca.SX.sym("Hi" + str(i), N_delta_yi, N_delta_yi)
        gi = ca.SX.sym("gi" + str(i), N_delta_yi)
        yi = ca.SX.sym("yi" + str(i), N_delta_yi)
        Ci = ca.SX.sym("Ci" + str(i), N_hi_list[i], N_delta_yi)
        
        fkh_hess_col_list += [ca.reshape(Hi, -1, 1)]
        modiefied_grad_col_list += [ca.reshape(gi, -1, 1)]
        yi_list += [yi]
        
        delta_yi = ca.SX.sym("delta_yi" + str(i),N_delta_yi)
        delta_yi_list += [delta_yi]
    
        obj += 1/2 * delta_yi.T @ Hi @ delta_yi + gi.T @ delta_yi
        sigma_Ai += Ai @ (yi + delta_yi)
        
        Ci_col_list += [ca.reshape(Ci, -1, 1)]
        g += [Ci @ delta_yi]
    obj += lambda_.T @ s + mu/2 * s.T @ s
    x = ca.vertcat(*delta_yi_list, s)
    p = ca.vertcat(lambda_, *(fkh_hess_col_list + modiefied_grad_col_list + yi_list + Ci_col_list))

    g += [ sigma_Ai - b - s ]
    g = ca.vertcat(*g)
    # Define proximal solver
    solver_opt = {}
    solver_opt['print_time'] = False
    solver_opt['ipopt'] = {
        'max_iter': 500,
        'print_level': 1,
        'acceptable_tol': 1e-6,
        'acceptable_obj_change_tol': 1e-6
    }

    nlp = {'x':x, 'g':g, 'f':obj, 'p': p}
#     print(nlp)
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opt)
    return solver    


