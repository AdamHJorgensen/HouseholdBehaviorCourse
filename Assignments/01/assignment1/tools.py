import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from types import SimpleNamespace
from consav.linear_interp import interp_2d

def marshall_calc(modelbank, id, base_h,shock_h):
    ''' 
    Calculate Marshall elasticities for a given modelbank and id
    Input: modelbank (dict), id (key of modelbank)
    Output: plot of Marshall elasticities
    '''
    #a. Allocate
    marshall_total   = np.nan + np.zeros(modelbank[id]['baseline'].par.T)
    marshall_child   = np.nan + np.zeros(modelbank[id]['baseline'].par.T)
    marshall_nochild = np.nan + np.zeros(modelbank[id]['baseline'].par.T)

    #b. Calculate
    for t in range(modelbank[id]['baseline'].par.T):
        
        #i. Conditions (Index of child arrival is identical in baseline and tax_increase model)
        with_child    = modelbank[id]['baseline'].sim.n[:,t]>0
        without_child = modelbank[id]['baseline'].sim.n[:,t]==0
        
        #Calculate elasticities
        marshall_total[t]   = np.nanmean((shock_h[:,t]             - base_h[:,t])             / base_h[:,t])
        marshall_child[t]   = np.nanmean((shock_h[with_child,t]    - base_h[with_child,t])    / base_h[with_child,t])
        marshall_nochild[t] = np.nanmean((shock_h[without_child,t] - base_h[without_child,t]) / base_h[without_child,t])
        
    return marshall_total, marshall_child, marshall_nochild

def marshall_long(modelbank, id,dtau=0.01, print_figure = True, output=False):
    ''' 
    Calculate and plot Marshall elasticities for a given modelbank and id
    Input: modelbank (dict), id (key of modelbank)
    Output: plot of Marshall elasticities
    '''

    # a. Setup
    par = modelbank[id]['baseline'].par
    base_h = modelbank[id]['baseline'].sim.h
    shock_h = modelbank[id]['tax_increase'].sim.h
    
    #b. Calculate
    marshall_total, marshall_child, marshall_nochild = marshall_calc(modelbank, id, base_h,shock_h)
      
    #c. Plot  
    if print_figure:
        fig, ax = plt.subplots()
        ax.plot(range(par.simT),marshall_total*(1/dtau),label='total')
        ax.plot(range(par.simT),marshall_child*(1/dtau),label='child')
        ax.plot(range(par.simT),marshall_nochild*(1/dtau),label='nochild')
        ax.set(xlabel='period, t',ylabel=f'Marshall elasticities',xticks=range(par.simT))
        ax.legend()
        fig.show()
    
    #4. Output
    if output:
        return marshall_total, marshall_child, marshall_nochild
    
def marshall_short(modelbank, id, dtau=0.01, print_figure = True, output = False):
    ''' 
    Calculate and plot Marshall elasticities for a given modelbank and id
    Input: modelbank (dict), id (key of modelbank)
    Output: plot of Marshall elasticities
    '''

    # 1. Setup
    par = modelbank[id]['baseline'].par
    sim_base = modelbank[id]['baseline'].sim
    sol_shock= modelbank[id]['tax_increase'].sol
    sim_shock = SimpleNamespace()
    shape = (par.simN,par.simT)
    sim_shock.h = np.nan + np.zeros(shape)

    #2. Calculate        
    # a. Find optimal consumption and hours for unanticipated permanent shock
    for t in range(par.simT):
        for i in range(par.simN):
            #interpolate optimal consumption and hours
            idx_sol = (t,sim_base.n[i,t], sim_base.s[i,t])
            sim_shock.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol_shock.h[idx_sol],sim_base.a[i,t],sim_base.k[i,t])
        
    # b. Unpack
    base_h = sim_base.h
    shock_h = sim_shock.h 
            
    #c. calculate
    marshall_total, marshall_child, marshall_nochild = marshall_calc(modelbank, id, base_h,shock_h)

    #3. Plot  
    if print_figure:
        fig, ax = plt.subplots()
        ax.plot(range(par.simT),marshall_total*(1/dtau),label='total')
        ax.plot(range(par.simT),marshall_child*(1/dtau),label='child')
        ax.plot(range(par.simT),marshall_nochild*(1/dtau),label='nochild')
        ax.set(xlabel='period, t',ylabel=f'Marshall elasticities',xticks=range(par.simT))
        ax.legend()
        fig.show()
    
    #4. Output 
    if output:
        return marshall_total, marshall_child, marshall_nochild