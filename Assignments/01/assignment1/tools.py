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
        ax.plot(range(par.simT),marshall_total*(1/dtau),label='Unconditional')
        ax.plot(range(par.simT),marshall_child*(1/dtau),label='With child')
        ax.plot(range(par.simT),marshall_nochild*(1/dtau),label='Without child')
        ax.set(xlabel='period, t',ylabel=f'Marshall elasticities',xticks=range(par.simT))
        ax.legend()
        fig.savefig(f'figures/marshall_long_{id}.png')
    
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
        ax.plot(range(par.simT),marshall_total*(1/dtau),label='Unconditional')
        ax.plot(range(par.simT),marshall_child*(1/dtau),label='With child')
        ax.plot(range(par.simT),marshall_nochild*(1/dtau),label='Without child')
        ax.set(xlabel='period, t',ylabel=f'Marshall elasticities',xticks=range(par.simT))
        ax.legend()
        fig.savefig(f'figures/marshall_short_{id}.png')
    
    #4. Output 
    if output:
        return marshall_total, marshall_child, marshall_nochild
    
def plot_compare_behavior(modelbank, id_list, names):
    '''Plot the policy functions for a list of models'''
    
    #a. Setup figure
    ax = {}
    fig, ((ax['c'],ax['a']),(ax['h'],ax['n']))  = plt.subplots(2,2)
    fig.tight_layout()
    
    #b. Loop over models
    for id in id_list:
        #i. Unpack
        par = modelbank[id]['baseline'].par
        sim = modelbank[id]['baseline'].sim
        
        #ii. Loop over behavior
        for var in ('c','a','h','n'):
            ax[var].plot(range(par.simT),np.mean(getattr(sim,var),axis=0))
            ax[var].set(xlabel='period, t',ylabel=f'Avg. {var}',xticks=range(par.simT))
            
    #Add legend
    fig.legend(names, loc='upper left', bbox_to_anchor=(0.05, 0.97), ncol=1)
    fig.savefig(f'figures/behavior_{id_list[-1]}.png')
    
    
def plot_compare_marshall(modelbank, id_list, names, dtau):
    '''Plot the policy functions for a list of models'''
    
    #a. Setup figure
    ax = {}
    fig, ax = plt.subplots(1,1)
    fig.tight_layout()
    
    #b. Loop over models
    for id in id_list:
        #i. Unpack
        par = modelbank[id]['baseline'].par
        sim = modelbank[id]['baseline'].sim
        
        #ii. Get Marshall elasticities
        marshall_total, _, _ = marshall_long(modelbank, id, dtau=dtau, print_figure=False, output=True)
        
        ax.plot(range(par.simT),marshall_total)
            
    #Figure settings
    ax.set(xlabel='period, t',ylabel=f'Marshall elasticities',xticks=range(par.simT))
    fig.legend(names, loc='lower left', bbox_to_anchor=(0.07, 0.07), ncol=1)
    fig.savefig(f'figures/marshall_{id_list[-1]}.png')