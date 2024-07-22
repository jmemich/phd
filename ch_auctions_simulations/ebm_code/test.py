import numpy as np
from ebm import ExclusiveBuyerMechanismApproximation as EBM

def f(x):
    return .25

for t in 9]:
    T = t+1
    ebm = EBM(
        N=2,
        X=[[6,8],[9,11]],
        c=[.9,5],
        T=T,
        f=f)
    
    max_i = np.inf
    max_r = -np.inf
    max_p = None
    max_ebm = None
#    for i, p in enumerate(ebm.Xj_ranges[0]):
#        p = [p,p]
    for i, p in enumerate(ebm.X_iter):
        r = ebm.obj(p)[0]
        # print(np.round(p, 4), np.round(r, 4))
        if r >= max_r:
            max_i = i
            max_r = r
            max_p = p
            max_ebm = ebm

    print("T=%s" % T,
          np.round(max_ebm.X_iter[max_i], 4).tolist(),
          # np.round(max_ebm.Xj_ranges[0][max_i], 4).tolist(),
          np.round(max_r, 6))

print("----\n")
Qs = max_ebm.obj(max_ebm.X_iter[max_i])[1]
# Qs = max_ebm.obj(max_p)[2]
print(np.round(np.array([Q[0]+Q[1] for Q in Qs]).reshape((max_ebm.T+1,max_ebm.T+1)),4)[:,:])
print(np.isclose(np.array([Q[0]+Q[1] for Q in Qs]),0).reshape((max_ebm.T+1,max_ebm.T+1)).astype(int)[:,:])
print(np.round(np.array([Q[0] + Q[1] for Q in Qs]).reshape((max_ebm.T+1,max_ebm.T+1)),4)[:,5:])




# this is obviously wrong:
print(np.round(np.array([ Q[0]+Q[1] for Q in Qs]).reshape((max_ebm.T+1,max_ebm.T+1)),4)[:,11:])
