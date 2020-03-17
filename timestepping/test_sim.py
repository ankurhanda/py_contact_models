from easydict import EasyDict as edict
import numpy as np 

from step_sphere import * 

params = edict({'h': 0.01, 
                'mu': 0.3,
                'm': 0.2,
                'r': 0.05,
                'step_fun': []})

st0 = np.hstack((np.array([0, 0, params.r+0.15, 
                          1, 0, 0, 0]), 
                 np.zeros(6)))

u = np.zeros(6)

for i in range(1, 51):
    st_next = step_sphere(params, st0, u)
    st0 = st_next
    print(st0[2])

