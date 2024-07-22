import numpy as np
from ebm import ExclusiveBuyerMechanismApproximation as EBM

def f(x):
    return .25

for t in [19]:
    T = t+1
    ebm = EBM(
        N=2,
        X=[[6,8],[9,11]],
        c=[.9,5],
        T=T,
        f=f)

    p = [6, min(ebm.Xj_ranges[1][ebm.Xj_ranges[1] >= 9.9 - 1e-10])]
    res = ebm.obj(p)
    print("T=%s" % T, res[0])
   
Qs = res[1]
excl_region = np.isclose(np.array([Q[0]+Q[1] for Q in Qs]),0).reshape((ebm.T+1,ebm.T+1)).astype(int)[:,:]
"""
print(excl_region)

# this is obviously wrong
print(np.round(np.array([Q[0] + Q[1] for Q in Qs]).reshape((ebm.T+1,ebm.T+1)),4)[:,5:])




print(np.round(np.array([Q[0] for Q in Qs]).reshape((ebm.T+1,ebm.T+1)),4)[:,15:])
"""
# print(np.round(np.array([Q[1] for Q in Qs]).reshape((ebm.T+1,ebm.T+1)),4)[:,-5:])
