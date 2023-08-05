# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:32:33 2023

@author: kaiget
"""

import casadi as ca
import numpy as np

# Define optimization problem
x = ca.SX.sym('x', 2)  # decision variables
obj = ca.sumsqr(x)  # objective function
constraints = [x[0] + x[1] >= 1, x[0] >= 0, x[1] >= 0]  # constraints
nlp = {'x': x, 'f': obj, 'g': constraints}  # define the problem

# Set solver options
options = {'verbose': False, 'eps_abs': 1e-6, 'eps_rel': 1e-6}

# Choose solver
solver = ca.osqp(nlp, options)

# Solve problem
result = solver({'x0': np.array([1, 1])})  # initial guess for x

# Print solution
print('Solution: ', result['x'])
