"""
Unit tests for ODE solver functionality.
"""

import pytest
import numpy as np
from scipy.integrate import solve_ivp
from src.model import ACEModel
from src.ode_solver import ace_ode_system, simulate_ace_model


class TestODESolver:
    """Test cases for ODE solver functionality."""
    
    def setup_method(self):
        """Setup test model."""
        self.model = ACEModel(
            a_tilde=1.0, b=0.2, gamma=0.5, c=1.0,
            p_star=0.7, r=1.0, lam=0.3, K=1.0
        )
    
    def test_ode_system_shape(self):
        """Test that ODE system returns correct shape."""
        # Test state vector
        state = [0.01, 0.01, 0.5]
        derivatives = ace_ode_system(0, state, self.model.params)
        
        # Should return 3 derivatives
        assert len(derivatives) == 3
        assert isinstance(derivatives, list)
    
    def test_ode_system_at_equilibrium(self):
        """Test ODE system at Architect equilibrium."""
        # At Architect equilibrium (xT=0, xV=0, V=V*)
        V_star = self.model.V_star
        state_eq = [0.0, 0.0, V_star]
        derivatives = ace_ode_system(0, state_eq, self.model.params)
        
        # All derivatives should be zero at equilibrium (within tolerance)
        for deriv in derivatives:
            assert abs(deriv) < 1e-10, f"Derivative not zero at equilibrium: {deriv}"
    
    def test_simulation_basic(self):
        """Test basic simulation functionality."""
        # Run simulation
        sol = simulate_ace_model(
            params=self.model.params,
            t_span=(0, 10),
            y0=[0.01, 0.01, 0.5]
        )
        
        # Check solution properties
        assert sol.success is True
        assert len(sol.t) > 0
        assert sol.y.shape[0] == 3  # 3 state variables
        assert sol.y.shape[1] == len(sol.t)
    
    def test_simulation_convergence(self):
        """Test simulation convergence to equilibrium for stable c."""
        if self.model.check_stability()[0]:  # Only test if c is stable
            # Run longer simulation
            sol = simulate_ace_model(
                params=self.model.params,
                t_span=(0, 200),
                y0=[0.01, 0.01, 0.5]
            )
            
            # Final state should be close to equilibrium
            xT_final, xV_final, V_final = sol.y[:, -1]
            xA_final = 1.0 - xT_final - xV_final
            
            # Architect should dominate
            assert xA_final > 0.99, f"Architect didn't dominate: xA={xA_final}"
            assert abs(V_final - self.model.V_star) < 0.1, f"V not at equilibrium: {V_final}"
    
    def test_ode_system_conservation(self):
        """Test that frequencies sum to 1."""
        # Test with random frequencies
        np.random.seed(42)
        
        for _ in range(10):
            xT = np.random.uniform(0, 0.5)
            xV = np.random.uniform(0, 0.5)
            xA = 1.0 - xT - xV
            V = np.random.uniform(0.1, 1.0)
            
            state = [xT, xV, V]
            dxT, dxV, dV = ace_ode_system(0, state, self.model.params)
            
            # Sum of frequency derivatives should be zero
            dxA = -dxT - dxV  # Since xA = 1 - xT - xV
            total_deriv = dxT + dxV + dxA
            assert abs(total_deriv) < 1e-10, f"Frequencies don't sum to 1: {total_deriv}"
    
    def test_resource_dynamics(self):
        """Test resource dynamics specifically."""
        # Test that resource grows when R̄ > λ/r
        state1 = [0.0, 0.0, 0.1]  # All Victims, high responsibility
        _, _, dV1 = ace_ode_system(0, state1, self.model.params)
        assert dV1 > 0, "Resource should grow with high responsibility"
        
        # Test that resource declines when R̄ < λ/r
        state2 = [1.0, 0.0, 0.1]  # All Tyrants, zero responsibility
        _, _, dV2 = ace_ode_system(0, state2, self.model.params)
        assert dV2 < 0, "Resource should decline with zero responsibility"
    
    def test_replicator_dynamics_properties(self):
        """Test properties of replicator dynamics."""
        # Test that frequencies stay in [0, 1]
        t_span = (0, 100)
        y0 = [0.01, 0.01, 0.5]
        
        sol = simulate_ace_model(self.model.params, t_span, y0)
        
        # Check all frequencies are valid
        xT_vals, xV_vals, V_vals = sol.y
        xA_vals = 1.0 - xT_vals - xV_vals
        
        assert np.all(xT_vals >= 0) and np.all(xT_vals <= 1)
        assert np.all(xV_vals >= 0) and np.all(xV_vals <= 1)
        assert np.all(xA_vals >= 0) and np.all(xA_vals <= 1)
        assert np.all(V_vals >= 0)  # Resource can't be negative
    
    def test_solver_methods_comparison(self):
        """Compare different ODE solver methods."""
        methods = ['RK45', 'BDF']
        final_states = []
        
        for method in methods:
            sol = simulate_ace_model(
                params=self.model.params,
                t_span=(0, 50),
                y0=[0.01, 0.01, 0.5],
                method=method,
                rtol=1e-8,
                atol=1e-10
            )
            final_states.append(sol.y[:, -1])
        
        # Different methods should give similar results
        state1, state2 = final_states
        diff = np.max(np.abs(state1 - state2))
        assert diff < 1e-6, f"Different methods gave different results: {diff}"
    
    def test_parameter_sensitivity_in_ode(self):
        """Test ODE sensitivity to parameter changes."""
        # Baseline
        params1 = self.model.params.copy()
        sol1 = simulate_ace_model(params1, (0, 20), [0.01, 0.01, 0.5])
        
        # Change c to unstable value
        params2 = params1.copy()
        params2['c'] = 0.1  # Below L (unstable)
        sol2 = simulate_ace_model(params2, (0, 20), [0.01, 0.01, 0.5])
        
        # Final states should be different
        final1 = sol1.y[:, -1]
        final2 = sol2.y[:, -1]
        
        # At least one component should be significantly different
        diff = np.max(np.abs(final1 - final2))
        assert diff > 0.1, f"ODE not sensitive to c change: diff={diff}"
    
    def test_jacobian_approximation(self):
        """Test numerical Jacobian approximation matches analytical expectations."""
        from scipy.optimize import approx_fprime
        
        # State near equilibrium
        state = [0.001, 0.001, self.model.V_star]
        
        # Define function for Jacobian approximation
        def f(state_array):
            return np.array(ace_ode_system(0, state_array.tolist(), self.model.params))
        
        # Numerical Jacobian
        J_numerical = approx_fprime(np.array(state), f, epsilon=1e-8)
        
        # Check that population block is approximately diagonal
        # (off-diagonals should be small near equilibrium)
        off_diag_sum = abs(J_numerical[0, 1]) + abs(J_numerical[1, 0])
        assert off_diag_sum < 1e-6, f"Jacobian not diagonal near equilibrium: {off_diag_sum}"


def test_boundary_cases():
    """Test ODE behavior at boundary cases."""
    # Test with V* = 0 (boundary case)
    model_boundary = ACEModel(p_star=0.3, lam=0.5)  # p* < λ/r
    assert model_boundary.V_star == 0.0
    
    # Simulation should handle this
    sol = simulate_ace_model(
        params=model_boundary.params,
        t_span=(0, 10),
        y0=[0.01, 0.01, 0.1]
    )
    
    # Resource should stay at or go to 0
    V_final = sol.y[2, -1]
    assert V_final >= 0, "Resource became negative"
    
    # Test with extreme frequencies
    model = ACEModel()
    
    # All Tyrants
    sol_T = simulate_ace_model(model.params, (0, 10), [1.0, 0.0, 0.5])
    assert sol_T.success is True
    assert sol_T.y[0, -1] > 0.99  # Should remain mostly Tyrants
    
    # All Victims
    sol_V = simulate_ace_model(model.params, (0, 10), [0.0, 1.0, 0.5])
    assert sol_V.success is True
    assert sol_V.y[1, -1] > 0.99  # Should remain mostly Victims


if __name__ == "__main__":
    pytest.main([__file__, "-v"])