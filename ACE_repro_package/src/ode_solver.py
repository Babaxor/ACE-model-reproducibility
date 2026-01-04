
ODE solver for the ACE model replicator-resource system.

Implements the coupled ODE system described in Section 3.4 of the manuscript.


import numpy as np
from scipy.integrate import solve_ivp


class ACEModel
    ACE (Architect in Coupled Ecosystem) Model ODE solver.
    
    def __init__(self, a_tilde=1.0, b=0.2, gamma=0.5, c=1.0, 
                 p_star=0.7, r=1.0, lam=0.3, K=1.0)
        
        Initialize ACE model parameters.
        
        Parameters
        ----------
        a_tilde  float
            Effective extraction rate (must be  0)
        b  float
            Cooperation cost coefficient (≥ 0)
        gamma  float
            Responsibility benefit coefficient (≥ 0)
        c  float
            Self-cost intensity (must be in (L, U) for stability)
        p_star  float
            Architect symmetry parameter (0  p_star  1)
        r  float
            Resource intrinsic growth rate ( 0)
        lam  float
            Resource maintenance threshold ( 0)
        K  float
            Resource carrying capacity ( 0)
        
        self.a_tilde = a_tilde
        self.b = b
        self.gamma = gamma
        self.c = c
        self.p_star = p_star
        self.r = r
        self.lam = lam
        self.K = K
        
        # Store all parameters in a tuple for ODE function
        self.params = (a_tilde, b, gamma, c, p_star, r, lam, K)
        
        # Compute equilibrium values
        self.V_star = self.compute_V_star()
        
        # Compute stability bounds
        self.L, self.U = self.compute_stability_bounds()
    
    def compute_V_star(self)
        Compute resource equilibrium value.
        if self.p_star  self.lam  self.r
            return self.K  (self.p_star - self.lam  self.r)
        else
            return 0.0
    
    def compute_stability_bounds(self)
        Compute L and U bounds for stability (Theorem 3.3).
        p = self.p_star
        numerator_L = (self.a_tilde  (1 - p)2 + 
                      self.b  p2 + 
                      self.gamma  p  (1 - p))
        denominator_L = 1 - p + p2
        L = numerator_L  denominator_L
        
        U = (self.a_tilde + 
             self.b  (1 - p) + 
             self.gamma  p  (1 - p))
        
        return L, U
    
    def ode_system(self, t, state)
        
        ODE system for the ACE model.
        
        Parameters
        ----------
        t  float
            Time (not used explicitly as system is autonomous)
        state  array-like
            State vector [x_T, x_V, V]
            
        Returns
        -------
        dstate_dt  array-like
            Derivatives [dx_Tdt, dx_Vdt, dVdt]
        
        xT, xV, V = state
        a_tilde, b, gamma, c, p_star, r, lam, K = self.params
        
        # Architect frequency (completes the simplex)
        xA = 1.0 - xT - xV
        
        # Payoff matrix entries (from Appendix D)
        # Against Architect
        Pi_TA = a_tilde  (1 - p_star) + gamma  p_star - c
        Pi_VA = -b  p_star
        Pi_AA = (a_tilde  p_star  (1 - p_star) - 
                b  p_star2 + 
                gamma  p_star2 - 
                c  p_star  (1 - p_star))
        
        # Against Tyrant
        Pi_TT = -c
        Pi_VT = 0
        Pi_AT = -c  p_star  (1 - p_star)
        
        # Against Victim
        Pi_TV = a_tilde + gamma - c
        Pi_VV = -b
        Pi_AV = (a_tilde  p_star - 
                b  p_star + 
                gamma  p_star - 
                c  p_star  (1 - p_star))
        
        # Fitness functions
        f_T = xT  Pi_TT + xV  Pi_TV + xA  Pi_TA
        f_V = xT  Pi_VT + xV  Pi_VV + xA  Pi_VA
        f_A = xT  Pi_AT + xV  Pi_AV + xA  Pi_AA
        
        # Average fitness
        f_bar = xT  f_T + xV  f_V + xA  f_A
        
        # Replicator equations
        dxT_dt = xT  (f_T - f_bar)
        dxV_dt = xV  (f_V - f_bar)
        
        # Resource dynamics
        # Average responsibility
        R_bar = xV + p_star  xA
        dV_dt = V  (r  R_bar - lam - (r  K)  V)
        
        return [dxT_dt, dxV_dt, dV_dt]
    
    def simulate(self, t_span=(0, 200), y0=None, 
                 method='RK45', rtol=1e-8, atol=1e-10)
        
        Run simulation of the ACE model.
        
        Parameters
        ----------
        t_span  tuple
            Time interval (start, end)
        y0  array-like, optional
            Initial conditions [x_T0, x_V0, V0]. If None, uses default.
        method  str
            ODE solver method ('RK45' or 'BDF')
        rtol, atol  float
            Relative and absolute tolerances
            
        Returns
        -------
        sol  scipy.integrate.OdeResult
            Solution object with time points and state values
        
        if y0 is None
            # Default initial conditions small mutant frequencies
            y0 = [0.01, 0.01, 0.5]
        
        # Solve ODE
        sol = solve_ivp(
            fun=lambda t, y self.ode_system(t, y),
            t_span=t_span,
            y0=y0,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
        
        return sol
    
    def check_stability(self, c=None)
        
        Check if the Architect strategy is stable for given c.
        
        Parameters
        ----------
        c  float, optional
            Self-cost parameter. If None, uses model's c.
            
        Returns
        -------
        stable  bool
            True if c is between L and U
        stability_margin  float
            Minimum distance to bounds
        
        if c is None
            c = self.c
        
        L, U = self.L, self.U
        stable = L  c  U
        stability_margin = min(c - L, U - c) if stable else 0
        
        return stable, stability_margin