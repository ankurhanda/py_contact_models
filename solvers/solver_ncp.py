import numpy as np

def project_circle(x, r):
    x_f = np.linalg.norm(x)

    x_proj = x
    if x_f > r:
        x_proj = x * r / x_f

    return x_proj
        
def solver_ncp(v_prev, Fext, M, J, mu, psi, h):

    nc = mu.shape[0]

    M_inv = np.linalg.inv(M)

    A = np.matmul(J, np.matmul(M_inv, J.T))

    A = (A + A.T)/2.0

    v_next = v_prev + h * np.matmul(M_inv, Fext)
    
    b = np.matmul(J, v_next)
    btilde = b + np.hstack((psi/h,
                            np.zeros(2*nc)))
    d = np.diagonal(A)
    x = np.zeros(3*nc)

    for _ in range(0, 30):
        for i in range(0, nc):

            #Normal 
            x_ni = x[i] - (np.dot(A[i, :], x) + btilde[i]) / d[i]
            x[i] = max(0, x_ni)

            min_di = min(d[i + nc], d[i + 2*nc])

            #Tangent 
            x_ti = x[i + nc] - (np.dot(A[i + nc, :], x) + btilde[i + nc])/min_di

            #Other 
            x_oi = x[i + 2*nc] - (np.dot(A[i + 2*nc, :], x) + btilde[i + 2*nc])/min_di

            #Project on circle 
            x_ti, x_oi = project_circle(np.array([x_ti, x_oi]), mu[i]*x[i])

            x[i + nc]   = x_ti
            x[i + 2*nc] = x_oi

    JxFext = np.matmul(J.T, x) + Fext*h
    v_next = v_prev + np.matmul(M_inv, JxFext)

    return v_next, x 


