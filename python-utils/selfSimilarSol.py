def selfSimilarSol(gam=1.4,Ma=0.3,Re=300,expMu=1.5,Pr=0.72):

    import math as m
    import cmath as cm
    import numpy as np
    from numpy import linalg as npla
    
    from scipy.sparse.linalg import spsolve
    from scipy.sparse        import diags, hstack, vstack
    from scipy.integrate     import solve_bvp
    from scipy               import interpolate as intp
    
    from functools import partial
    
    # from CoolProp.CoolProp   import PropsSI
    
    import matplotlib.pyplot as plt
    from matplotlib import rc, rcParams
    
    
    def cBL_RHS(y, f):
        C1    = f[3]**(expMu-1)
        
        RHS = np.vstack(( f[1],                               # f[0] = F0
                          f[2]/C1,                            # f[1] = F1 = W
                         -f[0]*f[2]/C1,                       # f[2] = F2
                          f[4]*Pr_inf/C1,                     # f[3] = G0 = h
                         -f[0]*f[4]*Pr_inf/C1-Ec*f[2]**2/C1,  # f[4] = G1
                          np.sqrt(2.0)*(f[3])))               # f[5] = eta
        return RHS
    
    def cBL_BC(f0,finf):
        BC_res = np.array([ f0[0],
                            f0[1],
                            f0[4],
                            f0[5],
                            finf[1]-1,
                            finf[3]-1])
        return BC_res
    
    
    
    
    expMu   = 3.0/2.0
    Pr_inf  = Pr 
    Ec      = (gam-1)*Ma**2
    Rgas    = 1.0/(gam*Ma*Ma)
    
    # set eta and allocate f
    eta  = np.linspace(0,10,600)
    f    = np.zeros((6,eta.size))
    
    # set initial condition for f
    f[1] = np.sqrt(eta/eta[-1])
    f[3] = 1.0
    f[5] = eta[-1]
    
    # solve bvp 
    res = solve_bvp(cBL_RHS, cBL_BC, eta, f, verbose=2)
    F0,U,_,T,_,yBlasius = res.sol(eta)
    
    
    # expand arrays to y = 100 onto 1000 
    
    idx = np.where(U > 0.99)[0][0]
    deltaBlasius = yBlasius[idx]
    
    
    r = 1/T
    V = U*yBlasius/(4.0**0.5) - F0/(r*(2**0.5))
    V *= deltaBlasius
    
    
    y = yBlasius/deltaBlasius*5
    
    yfin = np.zeros(1000)
    rfin = np.zeros(1000)
    Ufin = np.zeros(1000)
    Vfin = np.zeros(1000)
    efin = np.zeros(1000)
   
    yfin[:y.size] = y
    rfin[:y.size] = r
    Ufin[:y.size] = U
    Vfin[:y.size] = V
    for i in range(0,y.size):
        efin[i] = T[i]*Rgas/(gam-1.0)
    
    for i in range(y.size,yfin.size):
        yfin[i] = yfin[i-1] + 0.15
    Ufin[y.size:] = Ufin[y.size-1]
    rfin[y.size:] = rfin[y.size-1]
    Vfin[y.size:] = Vfin[y.size-1]
    efin[y.size:] = efin[y.size-1]
    
    yfin.tofile('./blasius1D/xProf.bin')
    Ufin.tofile('./blasius1D/wProf.bin')
    rfin.tofile('./blasius1D/rProf.bin')
    efin.tofile('./blasius1D/eProf.bin')
    Vfin.tofile('./blasius1D/uProf.bin')
