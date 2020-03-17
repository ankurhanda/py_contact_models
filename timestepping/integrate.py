import numpy as np 
from quaternions import *

def integrate(q, v, h):
    '''
    Input is
        q - pose [x, y, z, q0, q1, q2, q3]
        v - velocity [vx, vy, vz, wx, wy, wz]
        h - time step

    Output
        q_next - next pose
    '''

    u = np.linalg.norm(v[3:6])
    if u < 1e-8:
        p = np.array([1, 0, 0, 0])
    else:
        uhat = v[3:6]/u 
        p =np.array([np.cos(u*h/2), np.sin(u*h/2)*uhat])
    
    q_next = q
    q_next[0:3] = q[0:3] + h * v[0:3]

    r = q[3:7]

    q_next[3:7] = mult(p, r)

    return q_next