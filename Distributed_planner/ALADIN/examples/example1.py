# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:43:48 2023

@author: kaiget
"""

import casadi as ca
import numpy as np
from ALADIN_fun import create_subproblem, constraint_jac_approx, modified_grad, create_QP_problem

x = ca.SX.sym("x",2)
f = x[0] * x[1]
ca.hessian(f, x)

# %%
eps = 1e-5
# rho = 0.75
rho = 1e5
N_itermax = 50
A_list = []
sigma_list = []
fi_func_list = []
hi_func_list = []
N_hi_list = []

A = ca.DM([[1,-1]])
A_list += [A]
N = len(A_list)
b = ca.DM([0])

sigma_i = ca.diag([1,1])
sigma_list += [sigma_i]

Nx = 2
x = ca.SX.sym("x",Nx)
fi = x[0] * x[1]
fi_func = ca.Function("fi_func", [x], [fi])
fi_func_list += [fi_func]

fi_grad_func_list = []
fi_grad = ca.gradient(fi, x)
fi_grad_func = ca.Function("fi_grad_func", [x], [fi_grad])
fi_grad_func_list += [fi_grad_func]


# hi = ca.vertcat(x[0]+x[1]) # To be modified
hi = ca.vertcat(x[0]+x[1] - 2) # To be modified
hi_func = ca.Function("hi_func", [x], [hi])
Nhi = np.shape(hi)[0]
N_hi_list = [Nhi]
hi_func_list = [hi_func]

kappa_i = ca.SX.sym("kappa_i",Nhi)


hi_jac_func_list = []
fkh_hess_func_list = []
hi_jac = ca.jacobian(hi,x)
hi_jac_func = ca.Function("hi_jac_func",[x],[hi_jac])
hi_jac_func_list +=  [hi_jac_func]
fkh_i = fi + kappa_i.T @ hi
fkh_hess_i = ca.hessian(fkh_i, x)[0]
fkh_hess_i_func = ca.Function("fkh_hess_i_func", [x, kappa_i], [fkh_hess_i])
fkh_hess_func_list += [fkh_hess_i_func]


# %%

subsolver_list = []
# Define subproblem solvers
for i in range(N):
    Ai = A_list[i]
    sigma_i = sigma_list[i]
    fi_func = fi_func_list[i]
    hi_func = hi_func_list[i]
    subsolver_list += [create_subproblem(fi_func, Ai, rho, hi_func)]    
mu = 1e5
QP_solver = create_QP_problem(A_list, b,  mu, N_hi_list)
# %% Initial guess
delta_yi_list = []
sigma_i_list = []
xi_list = []
yi_list = []
lbhi_list = []
ubhi_list = []
lbx_list = []
ubx_list = []
Nx = 0
N_hi_sum = 0
for i in range(N):
    Ai = A_list[i]
    N_lambda, N_xi = np.shape(Ai)
    Nx += N_xi
    N_hi = N_hi_list[i]
    N_hi_sum += N_hi
#     xi = np.random.randn(N_xi,1).flatten().tolist()
    xi = ca.DM.zeros(N_xi,1).full().flatten().tolist()
    xi_list += [xi]
    sigma_i_list += [ca.diag([1] * N_xi)]
    
    lbhi_list += [[-ca.inf] * N_hi]
    ubhi_list += [[0] * N_hi]
    yi_list += [[0] * N_xi]
    lbx_list += [[-ca.inf] * N_xi]
    ubx_list += [[ca.inf] * N_xi]
lambda_ = np.random.randn(N_lambda,1)
lambda_ = ca.DM.zeros(N_lambda,1)
print(lambda_)
s_list = [0] * N_lambda
delta_yi_list = sum(yi_list,[])
    
nl_sub = {}



nl_qp = {}
nl_qp['lbg'] = [0] * (N_hi_sum + N_lambda)
nl_qp['ubg'] = [0] * (N_hi_sum + N_lambda)
nl_qp['lbx'] = sum(lbx_list,[]) + [-np.inf] * N_lambda    # delta_y and s lower bound
nl_qp['ubx'] = sum(ubx_list,[]) + [+np.inf] * N_lambda    # delta_y and s upper bound

# %% Track solution
yi_sol_list = []
delta_y_sol_list = []
lambda_list = []
x_sol_list = []
x_sol_list += [xi_list.copy()]


# delta_y_sol_list += [sum(xi_list,[])]
lambda_list += lambda_.full().flatten().tolist()
for i in range(N_itermax):
    sum_Ay = 0
    kappa_list = []
    # %% Step 1: solve the subproblem      
    for j in range(N):
        Ai = A_list[j]
        N_lambda_i, N_xi = np.shape(Ai)
        sigma_i = sigma_i_list[j]
        nl_sub['lbg'] = lbhi_list[j]
        nl_sub['ubg'] = ubhi_list[j]    
        nl_sub['x0'] = yi_list[j]
        nl_sub['lbx'] = lbx_list[j]
        nl_sub['ubx'] = ubx_list[j]
        nl_sub['p'] = lambda_list + xi_list[j] + ca.reshape(sigma_i, -1, 1).full().flatten().tolist()

        solver_subproblem = subsolver_list[j]
        yi_sol = solver_subproblem(**nl_sub)
        yi_list[j] = yi_sol['x'].full().flatten().tolist()
#         print(yi_list)
        yi_sol_list += [yi_list[j].copy()]
        kappa_i = yi_sol['lam_g']
        kappa_list += [kappa_i]
        
        sum_Ay += Ai @ yi_sol['x']

    # %% Step 2: Check if the tolerance satisfied
    #TODO: modify
    N_flag = 0
    for j in range(N):
        if rho * ca.norm_1( sigma_i_list[j] @ ca.DM(yi_list[j])) <= eps:
            N_flag += 1
    if ca.norm_1(sum_Ay - b) <= eps and N_flag == N:
        break
    # %% Step3: update Jacobian approximations, calculate the modified gradient, and update Hessian
    Ci_list = []    #  constraint Jacobian
    g_list = []    #  modified gradient
    H_list = []    #  Hessian
    for j in range(N):
        # 3.1 Choose Jacobian approximations
        yi = yi_list[j]
        hi_func = hi_func_list[j]
        hi_jac_func = hi_jac_func_list[j]
        fkh_hess_func = fkh_hess_func_list[j]
        hi = hi_func(yi)
        kappa_i = kappa_list[j]
        fi_grad = fi_grad_func_list[j](yi)
        hi_jac_real = hi_jac_func(yi)
        
        hi_jac_approx = constraint_jac_approx(yi, hi_func, hi_jac_func)
        Ci_list += [ca.reshape(hi_jac_real, -1, 1)]
        gi = modified_grad(yi,fi_grad, hi_jac_approx, hi_jac_real, kappa_i)
        g_list += [ca.reshape(gi, -1, 1)]
        
        Hi = fkh_hess_func(x, kappa_i)
        H_list += [ca.reshape(Hi, -1, 1)]
#         print("hi", hi, "kappa_i",kappa_i, "fi_grad",fi_grad, "hi_jac_real",hi_jac_real, "hi_jac_approx",hi_jac_approx, "gi",gi, "Hi" ,Hi)
    # %% Step 4: Solve QP problem
    nl_qp['x0'] = delta_yi_list + s_list    #  Initial guess
    
    H_para = ca.vertcat(*H_list)
    modiefied_grad = ca.vertcat(*g_list)
    y = ca.vertcat(* sum(yi_list,[]))
    Ci = ca.vertcat(*Ci_list)
    lambda_ = ca.vertcat(lambda_list)
    p = ca.vertcat(lambda_, H_para, modiefied_grad, y, Ci)
    nl_qp['p'] = ca.DM(p)
#     print(nl_qp)
    QP_sol = QP_solver(**nl_qp)
    
    alpha1 = 1
    alpha2 = 1
    alpha3 = 1
    # %% Step 5: Update x and lambda
    pos = 0
    
    delta_y = QP_sol['x'][0:Nx,:]
    QP_list = QP_sol['x'].full().flatten().tolist()
    lambda_QP = QP_sol['lam_g'][:N_lambda]
#     print("lambda_QP", lambda_QP)
    x = ca.DM(sum(xi_list,[]))
    y = ca.DM(sum(yi_list,[]))
    x_plus = x + alpha1 * (y - x) + alpha2 * delta_y
    for j in range(N):
        list_len = len(xi_list[j])
        xi_list[j] = x_plus[pos:pos+list_len].full().flatten().tolist()
       
    lambda_ = lambda_ + alpha3 * (lambda_QP - lambda_)
    lambda_list = lambda_.full().flatten().tolist()
    x_sol_list += xi_list.copy()