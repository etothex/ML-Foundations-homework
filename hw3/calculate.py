from math import *
import numpy as np


def hw3_7_update(u,v,k):
    partial_u = exp(u) + v*exp(u*v) + 2*u - 2*v -3
    partial_v = 2*exp(2*v) + u*exp(u*v) -2*u + 4*v -2
    u_new = u - k*partial_u
    v_new = v - k*partial_v
    E_uv = exp(u_new) + exp(2*v_new) + exp(u_new*v_new) \
           + u_new*u_new -2*u_new*v_new + 2*v_new*v_new - 3*u_new - 2*v_new
    return u_new,v_new,E_uv

def hw3_7():
    u=0
    v=0
    for i in range(5):
        u,v,E_uv = hw3_7_update(u,v,0.01)
    print(u,v,E_uv)

def hw3_10_update(u,v):
    partial_u = exp(u) + v*exp(u*v) + 2*u - 2*v -3
    partial_v = 2*exp(2*v) + u*exp(u*v) -2*u + 4*v -2
    partial_uu = exp(u) + v*v*exp(u*v) + 2
    partial_uv = exp(u*v) + v*u*exp(u*v) - 2
    partial_vv = 4*exp(2*v) + u*u*exp(u*v) + 4

    HsM = np.zeros((2,2))
    HsM[0][0] = partial_uu
    HsM[0][1] = partial_uv
    HsM[1][0] = partial_uv
    HsM[1][1] = partial_vv
    HsM = np.mat(HsM)

    GrdM = np.zeros((2,1))
    GrdM[0][0] = partial_u
    GrdM[1][0] = partial_v
    GrdM = np.mat(GrdM)

    Ntd = np.dot(HsM.I,GrdM)
    u_new = u - Ntd[0,0]
    v_new = v - Ntd[1,0]
    E_uv = exp(u_new) + exp(2*v_new) + exp(u_new*v_new) \
           + u_new*u_new -2*u_new*v_new + 2*v_new*v_new - 3*u_new - 2*v_new
    return u_new,v_new,E_uv
    # return u_new, v_new

def hw3_10():
    u=0.0
    v=0.0
    for i in range(5):
        u,v,E_uv = hw3_10_update(u,v)
        # u, v = hw3_10_update(u, v)
    # E_uv = exp(u) + exp(2*v) + exp(u*v) + u*u -2*u*v + 2*v*v - 3*u - 2*v
    print(u,v,E_uv)

if __name__ == '__main__':
    # hw3_7()                     # 0.094 0.001789 2.825
    hw3_10()                      # 0.61181171726 0.0704995471016 2.36082334564