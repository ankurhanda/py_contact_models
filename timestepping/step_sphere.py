import numpy as np
from integrate import *
from quaternions import *

import sys
sys.path.insert(0, '/home/ankur/workspace/code/py_contact_models/')

from solvers.solver_ncp import solver_ncp

def step_sphere(params, st, u):
    '''
    st = [x, y, z, q0, q1, q2, q3, .... ]
    params is dictionary and it has various meta-data related to sphere and optimisation 
    u is any other forces 
    '''

    h = params.h 
    mu = params.mu 
    m = params.m 
    r = params.r 

    step_fun = params.step_fun 

    M = m * np.diag([1, 1, 1, (2/5)*(r**2), (2/5)*(r**2), (2/5)*(r**2)])

    # Extract pose and velocity
    q = st[0:7]
    v = st[7:13]

    # Gravitational, external and other forces 
    omega = v[3:6]
    I = M[3:6, 3:6]
    Fext = np.hstack((np.array([0, 0, -9.81*m]), 
                    -np.cross(omega, np.matmul(I, omega)))) + u 

    # Contact normal distances 
    psi = q[2] - r 

    # Contact Jacobian 
    J = np.array([[0, 0, 1, 0, 0, 0], 
                  [1, 0, 0, 0, r, 0], 
                  [0, 1, 0, -r, 0, 0]])

    R = quat2mat(q[3:7])

    J[:, 3:6] = np.matmul(J[:, 3:6], R)

    if psi < 0.1:
        v_next, x = solver_ncp(v, Fext, M, J, np.array([mu]), psi, h)
    else:
        M_inv = np.linalg.inv(M)
        v_next = v + h * np.matmul(M_inv, Fext)

    q_next = integrate(q, v_next, h)

    st_next = np.hstack((q_next, v_next))

    return st_next


'''
    0.199019
    0.197057
    0.194114
    0.19019
    0.185285
    0.179399
    0.172532
    0.164684
    0.155855
    0.146045
    0.135254
    0.123482
    0.110729
    0.096995
    0.08228
    0.066584
'''