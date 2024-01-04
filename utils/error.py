import numpy as np
"""
Error metrics to assess the results
"""
def err_norm(u, u_p):
    err = np.linalg.norm(u - u_p, axis = (1, 2))**2/np.linalg.norm(u, axis = (1, 2))**2
    return 1 - err.mean(axis = 0)


def err(u, u_p):
    err = np.sum((u - u_p)**2, axis = (1, 2))/np.sum(u**2, axis = (1, 2))
    return 1 - err.mean(axis = 0)

def mse(u, u_p):
    err = np.mean((u - u_p)**2, axis = (1, 2))
    return err.mean(axis = 0)

def err_z(z_trn_out, zp_trn ):
    err = np.mean((z_trn_out - zp_trn)**2, axis=0)#/np.sum(z_trn_out**2, axis = 0)
    return  err.reshape(1, -1)