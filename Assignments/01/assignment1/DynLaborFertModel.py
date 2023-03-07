import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

import matplotlib.pyplot as plt

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

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
        
        # spouse
        par.spouse_base = 0. #0.1
        par.spouse_time = 0 #0.01

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 50 #70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 #30 # number of grid points in wealth grid    

        par.Nn = 2 # number of children #Question: Is max children not 1? (from problem formulation)

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
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. number of children grid
        par.n_grid = np.arange(par.Nn)

        # d. solution arrays
        shape = (par.T,par.Nn,par.Na,par.Nk)
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

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)

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

            # i. loop over state variables: number of children, human capital and wealth in beginning of period
            for i_n,kids in enumerate(par.n_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_k,capital in enumerate(par.k_grid):
                        idx = (t,i_n,i_a,i_k)

                        # ii. find optimal consumption and hours at this level of wealth in this period t.

                        if t==par.T-1: # last period
                            obj = lambda x: self.obj_last(x[0],assets,capital,kids)

                            constr = lambda x: self.cons_last(x[0],assets,capital)
                            nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf,keep_feasible=True) #Question: Does this ensure we are on pareto frontier?

                            # call optimizer
                            hours_min = - assets / self.wage_func(capital,t) + 1.0e-5 # minimum amout of hours that ensures positive consumption
                            hours_min = np.maximum(hours_min,2.0)
                            init_h = np.array([hours_min]) if i_a==0 else np.array([sol.h[t,i_n,i_a-1,i_k]]) # initial guess on optimal hours

                            res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                            # store results
                            sol.c[idx] = self.cons_last(res.x[0],assets,capital)
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = -res.fun

                        else:
                            
                            # objective function: negative since we minimize
                            obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,kids,t)  

                            # bounds on consumption 
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.0
                            ub_h = np.inf 

                            bounds = ((lb_c,ub_c),(lb_h,ub_h))
                
                            # call optimizer
                            init = np.array([lb_c,1.0]) if (i_n == 0 & i_a==0 & i_k==0) else res.x  # initial guess on optimal consumption and hours
                            res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 
                        
                            # store results
                            sol.c[idx] = res.x[0]
                            sol.h[idx] = res.x[1]
                            sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital):
        par = self.par

        income = self.income_func(capital,hours, par.T-1)
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital,kids):
        cons = self.cons_last(hours,assets,capital)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids,t):

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

        # c. utility from consumption
        util = self.util(cons,hours,kids)
        
        # d. *expected* continuation value from savings
        income = self.income_func(capital,hours, t)
        a_next = (1.0+par.r)*(assets + income - cons)
        k_next = capital + hours

        # no birth
        kids_next = kids
        V_next = sol.V[t+1,kids_next]
        V_next_no_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # birth
        if (kids>=(par.Nn-1)): #Question: why -1
            # cannot have more children
            V_next_birth = V_next_no_birth

        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,kids_next]
            V_next_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        EV_next = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth

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
    
    def spouse(self, t):
        # Spause's income
        par = self.par
        
        return par.spouse_base + par.spouse_time * t
    
    def income_func(self, capital, hours, t):
        # Total income
        par = self.par
        
        return self.wage_func(capital, t) * hours + self.spouse(t)

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
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.n[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.income_func(sim.k[i,t],sim.h[i,t], t)
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                    
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
        fig.show()
                    
    def plot_behavior(self): 
        # a. unpack
        par = self.par
        sim = self.sim
        
        # b. Plot
        for var in ('c','a','h','n'):
            fig, ax = plt.subplots()
            ax.scatter(range(par.simT),np.mean(getattr(sim,var),axis=0),label='Simulated')
            ax.set(xlabel='period, t',ylabel=f'Avg. {var}',xticks=range(par.simT))
