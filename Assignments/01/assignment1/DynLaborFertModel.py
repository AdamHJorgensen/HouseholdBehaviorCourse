import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.interpolate import RegularGridInterpolator
import warnings
#warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

import matplotlib.pyplot as plt

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """
        self.savefolder = 'models'
        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 0.98 # discount factor

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.05 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax

        # children
        par.p_birth = 0.1
        par.childcost = 0.
        
        # spouse
        par.p_spouse = 1. #0.8
        par.spouse_base = 0. #0.1
        par.spouse_time = 0 #0.01

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 0. # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 100 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in capital grid
        par.Nk = 20 # number of grid points in capital grid    

        par.Nn = 2 # number of children
        
        par.Ns = 2 # number of spouses

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = - nonlinspace(-par.a_max,-par.a_min,par.Na,1.2)[::-1]

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. number of children grid
        par.n_grid = np.arange(par.Nn)
        
        # spouse grid
        par.s_grid = np.arange(par.Ns)

        # d. solution arrays
        shape = (par.T,par.Nn,par.Ns, par.Na,par.Nk)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)
        sim.s = np.zeros(shape,dtype=np.int_)

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform_child  = np.random.uniform(size=shape)
        sim.draws_uniform_spouse = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)
        sim.s_init = np.random.choice(par.s_grid, p=[1-par.p_spouse,par.p_spouse], size=par.simN)

        # h. vector of wages. Used for simulating elasticities
        par.w_vec = par.w * np.ones(par.T)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            #print('solving period',t,'of',par.T-1)

            # i. loop over state variables: number of children, human capital and wealth in beginning of period
            for i_n,kids in enumerate(par.n_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_k,capital in enumerate(par.k_grid):
                        for i_s, spouse in enumerate(par.s_grid):
                            #Set index
                            idx = (t,i_n,i_s,i_a,i_k)
                            
                            #Skip if deterministic spouse (always or never a spouse)
                            if (par.p_spouse == 1.) & (spouse == 0):
                                # store results
                                sol.c[idx] = 0.
                                sol.h[idx] = 0.
                                sol.V[idx] = 0.
                                continue

                            # ii. find optimal consumption and hours at this level of wealth in this period t.

                            if t==par.T-1: # last period
                                obj = lambda x: self.obj_last(x[0],assets,capital,kids, spouse)

                                constr = lambda x: self.cons_last(x[0],assets,capital, kids, spouse)
                                nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf, keep_feasible=True) #Question: Does this ensure we are on pareto frontier?

                                # call optimizer
                                hours_min = - (assets + self.spouse_inc(spouse, t) - par.childcost * (kids > 0)) / (self.wage_func(capital,t) + 1.0e-8) + 1.0e-5 # minimum amout of hours that ensures positive consumption
                                hours_min = np.maximum(hours_min,2.0)
                                init_h = np.array([hours_min]) if i_a==0 else np.array([sol.h[t,i_n,i_s,i_a-1,i_k]]) # initial guess on optimal hours

                                res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                                # store results
                                sol.c[idx] = self.cons_last(res.x[0],assets,capital, kids, spouse)
                                sol.h[idx] = res.x[0]
                                sol.V[idx] = -res.fun

                            else:
                                
                                # objective function: negative since we minimize
                                obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,kids, spouse, t)  

                                # bounds on consumption 
                                lb_c = 0.000001 # avoid dividing with zero
                                ub_c = np.inf

                                # bounds on hours
                                lb_h = 0.0
                                ub_h = np.inf 

                                bounds = ((lb_c,ub_c),(lb_h,ub_h))
                    
                                # call optimizer
                                init = np.array([lb_c,1.0]) if (i_n == 0 & i_s == 0 & i_a==0 & i_k==0) else res.x  # initial guess on optimal consumption and hours
                                res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 
                            
                                # store results
                                sol.c[idx] = res.x[0]
                                sol.h[idx] = res.x[1]
                                sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital, kids, spouse):
        par = self.par

        income = self.income_func(capital,hours, kids, spouse, par.T-1)
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital,kids, spouse):
        cons = self.cons_last(hours,assets,capital, kids, spouse)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids, spouse, t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. Utility and income
        # i. utility from consumption and income
        util = self.util(cons,hours,kids)
        
        # ii. Income today
        income = self.income_func(capital,hours, kids, spouse, t)
        a_next = (1.0+par.r)*(assets + income - cons)
        k_next = capital + hours
        
        # d. *expected* continuation value from savings
        # Only calculate no spouse options if spouse is not deterministic
        if par.p_spouse != 1.:
            # i. no birth and no spouse
            kids_next = kids
            spouse_next = 0
            V_next = sol.V[t+1,kids_next, spouse_next]
            V_next_no_birth_no_spouse = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)
            
            # ii. birth and no spouse - CAN'T HAPPEN
            V_next_birth_no_spouse = V_next_no_birth_no_spouse
        else: #Set to zero if deterministic, because probability is zero anyway
            V_next_no_birth_no_spouse   = 0.
            V_next_birth_no_spouse      = 0.
            
        # iii. no birth and spouse     
        kids_next = kids
        spouse_next = 1
        V_next = sol.V[t+1,kids_next, spouse_next]
        V_next_no_birth_spouse = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)
        
        # iv. birth and spouse
        if (kids>=(par.Nn-1)):
            # cannot have more children
            V_next_birth_spouse = V_next_no_birth_spouse

        else:
            kids_next = kids + 1
            spouse_next = 1
            V_next = sol.V[t+1,kids_next, spouse_next]
            V_next_birth_spouse = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # v. Expected value
        EV_next = par.p_birth * par.p_spouse        * V_next_birth_spouse \
                + (1-par.p_birth) * par.p_spouse    * V_next_no_birth_spouse \
                + par.p_birth * (1-par.p_spouse)    * V_next_birth_no_spouse \
                + (1-par.p_birth) * (1-par.p_spouse)* V_next_no_birth_no_spouse

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,kids):
        par = self.par

        beta = par.beta_0 + par.beta_1*kids

        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau )* par.w_vec[t] * (1.0 + par.alpha * capital)
    
    def spouse_inc(self, spouse, t):
        # Spause's income
        par = self.par
        
        if spouse == 1:
            return par.spouse_base + par.spouse_time * t
        else:
            return 0.0
    
    def income_func(self, capital, hours, kids, spouse, t):
        # Total income
        par = self.par
        
        return self.wage_func(capital, t) * hours + self.spouse_inc(spouse, t) - par.childcost * (kids > 0)
    
    def s_interp_2d(self,grid1,grid2,values,points1,points2, method='cubic'):
        # Interpolate 2d array
        par = self.par
        sol = self.sol
        
        # Set up interpolater
        interp = RegularGridInterpolator((grid1,grid2),values,method=method,bounds_error=False,fill_value=None)
        
        # Interpolate
        return interp(np.array([points1,points2]).T)
    
    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.n[i,0] = sim.n_init[i]
            sim.s[i,0] = sim.s_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.n[i,t], sim.s[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    #Income, asset and human capital
                    income = self.income_func(sim.k[i,t],sim.h[i,t], sim.n[i,t], sim.s[i,t], t)
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    #Children
                    birth = 0 
                    if ((sim.draws_uniform_child[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1)) & (sim.s[i,t]==1)):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                    
                    #Spouse
                    sim.s[i,t+1] = 0
                    if sim.draws_uniform_spouse[i,t] <= par.p_spouse:
                        sim.s[i,t+1] = 1
                        

                    
        sim.time_since_birth = self.get_birth_time()
                    
    def get_birth_time(self):
        
        # a. unpack
        par = self.par
        sim = self.sim
        
        # b. Calculate birth times
        birth = np.zeros(sim.n.shape,dtype=np.int_)
        birth[:,1:] = (sim.n[:,1:] - sim.n[:,:-1]) > 0

        # c. Calculate time since birth
        periods = np.tile([t for t in range(par.simT)],(par.simN,1))
        time_of_birth = np.max(periods * birth, axis=1)

        I = time_of_birth>0
        time_of_birth[~I] = 1000 # never has a child
        time_of_birth = np.transpose(np.tile(time_of_birth , (par.simT,1)))

        time_since_birth = periods - time_of_birth
        
        return time_since_birth
    
    def plot_event_study(self, min_time = -8, max_time = 8):
        
        # a. unpack
        sim = self.sim
        
        # b. Calculate effect of birth
        # i. Make grid
        event_grid = np.arange(min_time,max_time+1)

        # ii. calculate average outcome across time since birth
        event_hours = np.nan + np.zeros(event_grid.size)
        for t,time in enumerate(event_grid):
            event_hours[t] = np.mean(sim.h[sim.time_since_birth==time])

        # iii. relative to period before birth
        event_hours_rel = event_hours - event_hours[event_grid==-1]
        
        # c. Plot
        fig, ax = plt.subplots()
        ax.scatter(event_grid,event_hours_rel)
        ax.hlines(y=0,xmin=event_grid[0],xmax=event_grid[-1],color='gray')
        ax.vlines(x=-0.5,ymin=np.nanmin(event_hours_rel),ymax=np.nanmax(event_hours_rel),color='red')
        ax.set(xlabel='Time since birth',ylabel=f'Hours worked (rel. to -1)',xticks=event_grid)
        
        fig.savefig(f'figures/{self.name}_event.png')
                    
    def plot_behavior(self): 
        # a. unpack
        par = self.par
        sim = self.sim
        
        #Make conditions
        unconditional = np.ones(sim.n.shape,dtype=bool)
        with_child = sim.n>0
        without_child = sim.n==0
        
        # b. Plot
        ax = {}
        fig, ((ax['c'],ax['a']),(ax['h'],ax['n']))  = plt.subplots(2,2)
        for var in ('c','a','h','n'):
            ax[var].plot(range(par.simT),np.nanmean(np.where(unconditional  ,getattr(sim,var),np.nan),axis=0),label='Unconditional')
            ax[var].plot(range(par.simT),np.nanmean(np.where(with_child     ,getattr(sim,var),np.nan),axis=0),label='With child')
            ax[var].plot(range(par.simT),np.nanmean(np.where(without_child  ,getattr(sim,var),np.nan),axis=0),label='Without child')
            ax[var].set(xlabel='period, t',ylabel=f'Avg. {var}',xticks=range(par.simT))
            if var=='c':
                ax[var].legend()
        fig.tight_layout()
        fig.savefig(f'figures/{self.name}_behavior.png')
        
    def plot_policy(self,T,n=1,s=1):
        # a. unpack
        par = self.par
        sol = self.sol
        a_mesh, k_mesh = np.meshgrid(par.a_grid,par.k_grid, indexing='ij')
        
        # b. Plot
        fig = plt.figure()
        # ax1 = fig.add_subplot(121,projection='3d')
        # ax1.plot_surface(a_mesh,k_mesh,sol.c[T,n,s],cmap='viridis',edgecolor='none')
        # ax1.set_title('Consumption')
        
        # b. Plot
        ax2 = plt.axes(projection = '3d')
        #ax2 = fig.add_subplot(122,projection='3d')
        ax2.plot_surface(k_mesh,a_mesh,sol.h[T,n,s],cmap='viridis',edgecolor='none')
        ax2.set_xlabel('k')
        ax2.set_ylabel('a')
        ax2.set_title('Hours worked')
        
        #fig.tight_layout()
        fig.savefig(f'figures/{self.name}_policy_T{T}_n{n}_s{s}.png')
        
