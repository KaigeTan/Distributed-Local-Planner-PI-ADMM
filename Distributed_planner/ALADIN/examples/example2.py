# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:29:46 2023

@author: kaiget
"""

import casadi as ca
import numpy as np
from ALADIN_fun import create_subproblem, constraint_jac_approx, modified_grad, create_QP_problem

np.random.seed(2)
N = 2 # Should test with 25000
Nx = 4
eps = 1e-5
sigma_i = 10
sigma_bar_i = 10
# rho = 0.75
rho = 1
N_itermax = 15

# Define A matrix
A_list = []
NA_col = int(N * Nx / 2)
I = ca.diag([1,1])
for i in range(N):
    A = ca.DM.zeros(NA_col, Nx)
    if i == 0:
        A[NA_col-2: , :2] = -I
    else:
        A[(i-1)*2: (i-1)*2+2, :2] = -I        
    A[i*2:i*2+2,Nx-2:] = I
    A_list += [A]
# Define b
b = ca.DM.zeros(NA_col,1)

# %% Define parameter
eta_list = []
eta_bar_list = []

for i in range(N):
    eta = np.array([[N * np.cos(2*(i+1) * np.pi / N)],[N * np.sin(2*(i+1) * np.pi / N)]]) + sigma_i * np.random.randn(2,1)
    eta_list +=  [ca.DM(eta)]
    
    eta_bar = 2 * N * np.sin(np.pi / N) + sigma_i * np.random.randn(1)
    eta_bar_list += [ca.DM(eta_bar)]
    
# %% Define obejective function
fi_list = []
fi_func_list = []
x = ca.SX.sym("x",Nx)    #  $x_{i}=\left(\chi_{i}^{\top}, \zeta_{i}^{\top}\right)^{\top} \in \mathbb{R}^{4}$
for i in range(N):
    if i == N-1:
        fi = 1 / (4 * sigma_i ** 2) * (x[0:2] - eta_list[i]).T @ (x[0:2] - eta_list[i]) \
            + 1 / (4 * sigma_i**2) * (x[2:] - eta_list[0]).T @ (x[2:] - eta_list[0]) \
                + 1 / (2 * sigma_bar_i**2) * ( ca.norm_2(x[0:2] - x[2:]) - eta_bar_list[i] )**2
    else:
        fi = 1 / (4 * sigma_i ** 2) * (x[0:2] - eta_list[i]).T @ (x[0:2] - eta_list[i]) \
            + 1 / (4 * sigma_i**2) * (x[2:] - eta_list[i+1]).T @ (x[2:] - eta_list[i+1]) \
                + 1 / (2 * sigma_bar_i**2) * (ca.norm_2(x[0:2] - x[2:]) - eta_bar_list[i])**2
    fi_list += [fi]
    fi_func = ca.Function("fi_func", [x], [fi])
    fi_func_list += [fi_func]
    
# %% Define gradient function
fi_grad_list = []
fi_grad_func_list = []
for i in range(N):
    fi_grad = ca.gradient(fi_list[i], x)
    fi_grad_list += [fi_grad]
    fi_grad_func = ca.Function("fi_grad_func", [x], [fi_grad])
    fi_grad_func_list += [fi_grad_func]

# %%  Define inequality constraints
hi_list = []
hi_func_list = []
Nhi_list = []
for i in range(N):
    hi = (ca.norm_2(x[0:2] - x[2:]) - eta_bar_list[i])**2 - sigma_bar_i ** 2
    hi_list += [hi]
    hi_func = ca.Function("hi_func", [x], [hi])
    hi_func_list += [hi_func]
    # Deal with the number of inequality constraints for each i.
    Nhi = np.shape(hi)[0]
    Nhi_list += [Nhi]

# %% Define approximate jacobian, real jacobian and Hessian.
kappa_i_list = []
hi_jac_list = []
hi_jac_func_list = []
fkh_hess_i_list = []
fkh_hess_func_list = []
for i in range(N):
    # Kappa
    kappa_i = ca.SX.sym("kappa_i",Nhi)
    kappa_i_list += [kappa_i]
    # Jacobian function
    hi_jac = ca.jacobian(hi_list[i],x)
    hi_jac_list += [hi_jac]
    hi_jac_func = ca.Function("hi_jac_func",[x],[hi_jac])
    hi_jac_func_list +=  [hi_jac_func]
    # Hessian fucntion
    fi = fi_list[i]
    hi = hi_list[i]
    fkh_i = fi + kappa_i.T @ hi
    fkh_hess_i = ca.hessian(fkh_i, x)[0] # [1] is 4x1, first order?
    fkh_hess_i_func = ca.Function("fkh_hess_i_func", [x, kappa_i], [fkh_hess_i])
    fkh_hess_i_list += [fkh_hess_i]
    fkh_hess_func_list += [fkh_hess_i_func]

# %% Define parallel subproblem solvers -- only define the formulation of the problem
subsolver_list = []
for i in range(N):
    Ai = A_list[i]
    fi_func = fi_func_list[i]
    hi_func = hi_func_list[i]
    subsolver_list += [create_subproblem(fi_func, Ai, rho, hi_func)]
    
# %% Define QP problem
mu = 1e5
QP_solver = create_QP_problem(A_list, b,  mu, Nhi_list)

# %% Initial guess
delta_yi_list = []
Sigma_i_list = []
xi_list = []
yi_list = []
lbhi_list = []
ubhi_list = []
lbx_list = []
ubx_list = []
Nx = 0
Nhi_sum = 0
for i in range(N):
    Ai = A_list[i]
    N_lambda, N_xi = np.shape(Ai)
    Nx += N_xi
    Nhi = Nhi_list[i]
    Nhi_sum += Nhi
#     xi = np.random.randn(N_xi,1).flatten().tolist()
# #     xi = ca.DM.zeros(N_xi,1).full().flatten().tolist()
    if i == N-1:
        xi = ca.vertcat(eta_list[N-1],eta_list[0]).full().flatten().tolist()
    else:
        xi = ca.vertcat(eta_list[i],eta_list[i+1]).full().flatten().tolist()
    xi_list += [xi]

#     print(N_xi)
    Sigma_i_list += [ca.diag([1] * N_xi)]
    
    lbhi_list += [[-ca.inf] * Nhi]
    ubhi_list += [[0] * Nhi]
    yi = np.random.randn(N_xi,1).flatten().tolist()
    yi_list += [yi]   
#     yi_list += [[0] * N_xi]
    lbx_list += [[-ca.inf] * N_xi]
    ubx_list += [[ca.inf] * N_xi]
lambda_ = np.random.randn(N_lambda,1)
lambda_ = ca.DM(lambda_)
# lambda_ = ca.DM.zeros(N_lambda,1)
s_list = [0] * N_lambda
delta_yi_list = sum(yi_list,[])

# %% Define solver
nl_sub = {}

nl_qp = {}
nl_qp['lbg'] = [0] * (Nhi_sum + N_lambda)
nl_qp['ubg'] = [0] * (Nhi_sum + N_lambda)
nl_qp['lbx'] = sum(lbx_list,[]) + [-np.inf] * N_lambda    # delta_y and s lower bound
nl_qp['ubx'] = sum(ubx_list,[]) + [+np.inf] * N_lambda    # delta_y and s upper bound

# %% Track solution
yi_sol_list = []
delta_y_sol_list = []
lambda_list = []
x_sol_list = []
x_sol_list += [xi_list.copy()]
lambda_list += lambda_.full().flatten().tolist()


# %% solve problem

for i in range(N_itermax):
    sum_Ay = 0
    kappa_sol_list = []
    # Step 1: solve the subproblem      
    for j in range(N):
        Ai = A_list[j]
        N_lambda_i, N_xi = np.shape(Ai)
        Sigma_i = Sigma_i_list[j]
        nl_sub['lbg'] = lbhi_list[j]
        nl_sub['ubg'] = ubhi_list[j]    
        nl_sub['x0'] = yi_list[j]
        nl_sub['lbx'] = lbx_list[j]
        nl_sub['ubx'] = ubx_list[j]
        nl_sub['p'] = lambda_list + xi_list[j] + ca.reshape(Sigma_i, -1, 1).full().flatten().tolist()

        solver_subproblem = subsolver_list[j]
        yi_sol = solver_subproblem(**nl_sub)
        yi_list[j] = yi_sol['x'].full().flatten().tolist()
#         print(yi_sol['x'])
        yi_sol_list += [yi_list[j].copy()]
        kappa_i_sol = yi_sol['lam_g']
        kappa_sol_list += [kappa_i_sol]
        
        sum_Ay += Ai @ yi_sol['x']

    # Step 2: Check if the tolerance satisfied
    # TODO: modify
    N_flag = 0
    for j in range(N):
        if rho * ca.norm_1( Sigma_i_list[j] @ ca.DM(yi_list[j])) <= eps:
            N_flag += 1
    if ca.norm_1(sum_Ay - b) <= eps and N_flag == N:
        break
    # Step3: update Jacobian approximations, calculate the modified gradient, and update Hessian
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
        kappa_i_sol = kappa_sol_list[j]
        fi_grad = fi_grad_func_list[j](yi)
        hi_jac_real = hi_jac_func(yi)
        
        hi_jac_approx = constraint_jac_approx(yi, hi_func, hi_jac_func)
        Ci_list += [ca.reshape(hi_jac_real, -1, 1)]
        gi = modified_grad(fi_grad, hi_jac_approx, hi_jac_real, kappa_i_sol)
        g_list += [ca.reshape(gi, -1, 1)]
        
        Hi = fkh_hess_func(yi, kappa_i_sol)
        H_list += [ca.reshape(Hi, -1, 1)]
#         print("hi", hi, "kappa_i_sol",kappa_i_sol, "fi_grad",fi_grad, "hi_jac_real",hi_jac_real, "hi_jac_approx",hi_jac_approx, "gi",gi, "Hi" ,Hi)
    # Step 4: Solve QP problem
    nl_qp['x0'] = delta_yi_list + s_list    #  Initial guess
    
    H_para = ca.vertcat(*H_list)
    modified_grad_value = ca.vertcat(*g_list)
    y = ca.vertcat(* sum(yi_list,[]))
    Ci = ca.vertcat(*Ci_list)
    lambda_ = ca.vertcat(lambda_list)
    p = ca.vertcat(lambda_, H_para, modified_grad_value, y, Ci)
    nl_qp['p'] = ca.DM(p)
    print('current iteration:', i)
    print(lambda_)
    print(H_para)
    print(modified_grad_value)
    print(y)
    print(Ci)
    QP_sol = QP_solver(**nl_qp)
#     print(QP_sol)
    alpha1 = 1
    alpha2 = 1
    alpha3 = 1
    # Step 5: Update x and lambda
    pos = 0
#     print(QP_sol['x'][0:Nx,:])
    delta_y = QP_sol['x'][0:Nx,:]
    QP_list = QP_sol['x'].full().flatten().tolist()
    lambda_QP = QP_sol['lam_g'][:N_lambda]
#     print("lambda_QP", lambda_QP)
    x = ca.DM(sum(xi_list,[]))
    y = ca.DM(sum(yi_list,[]))
#     print(delta_y)
#     print(y,x)
    x_plus = x + alpha1 * (y - x) + alpha2 * delta_y
    for j in range(N):
        list_len = len(xi_list[j])
        xi_list[j] = x_plus[pos:pos+list_len].full().flatten().tolist()
        pos = pos+list_len
#     print(xi_list)
#     print(lambda_QP)
    lambda_ = lambda_ + alpha3 * (lambda_QP - lambda_)
    lambda_list = lambda_.full().flatten().tolist()
    x_sol_list += [xi_list.copy()]
