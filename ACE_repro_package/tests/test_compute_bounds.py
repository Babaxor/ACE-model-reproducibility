"""
Unit tests for stability bounds computation.
"""

import pytest
import numpy as np
from src.compute_bounds import compute_LU, check_stability_for_c


class TestComputeBounds:
    """Test cases for stability bounds computation."""
    
    def test_baseline_parameters(self):
        """Test with baseline parameters from manuscript."""
        L, U, info = compute_LU(a_tilde=1.0, b=0.2, gamma=0.5, p_star=0.7)
        
        # Check values match manuscript (within tolerance)
        assert pytest.approx(L, rel=1e-3) == 0.371
        assert pytest.approx(U, rel=1e-3) == 2.833
        
        # Check width is positive
        assert U > L
        assert info['nonempty'] is True
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid a_tilde
        with pytest.raises(ValueError, match="a_tilde must be > 0"):
            compute_LU(a_tilde=0, b=0.2, gamma=0.5, p_star=0.7)
        
        # Invalid p_star
        with pytest.raises(ValueError, match="p_star must be in"):
            compute_LU(a_tilde=1.0, b=0.2, gamma=0.5, p_star=1.5)
        
        # Negative b or gamma
        with pytest.raises(ValueError):
            compute_LU(a_tilde=1.0, b=-0.1, gamma=0.5, p_star=0.7)
    
    def test_boundary_limits(self):
        """Test boundary behavior as p→0 and p→1."""
        # As p→0
        L_small, U_small, _ = compute_LU(a_tilde=1.0, b=0.2, gamma=0.5, p_star=1e-6)
        assert pytest.approx(L_small, rel=1e-2) == 1.0  # a_tilde
        assert pytest.approx(U_small, rel=1e-2) == 1.2  # a_tilde + b
        
        # As p→1 (U should diverge)
        L_large, U_large, _ = compute_LU(a_tilde=1.0, b=0.2, gamma=0.5, p_star=0.9999)
        assert pytest.approx(L_large, rel=1e-2) == 0.2  # b
        assert U_large > 1000  # Should be large
    
    def test_stability_check(self):
        """Test stability checking for different c values."""
        # Baseline parameters
        params = {'a_tilde': 1.0, 'b': 0.2, 'gamma': 0.5, 'p_star': 0.7}
        
        # c below L (should be unstable)
        result_below = check_stability_for_c(c=0.1, **params)
        assert result_below['stable'] is False
        assert result_below['c'] < result_below['L']
        
        # c between L and U (should be stable)
        result_between = check_stability_for_c(c=1.0, **params)
        assert result_between['stable'] is True
        assert result_between['L'] < result_between['c'] < result_between['U']
        
        # c above U (should be unstable)
        result_above = check_stability_for_c(c=5.0, **params)
        assert result_above['stable'] is False
        assert result_above['c'] > result_above['U']
    
    def test_symmetry_properties(self):
        """Test mathematical properties of the bounds."""
        # L and U should be linear in a_tilde
        L1, U1, _ = compute_LU(a_tilde=1.0, b=0.2, gamma=0.5, p_star=0.5)
        L2, U2, _ = compute_LU(a_tilde=2.0, b=0.2, gamma=0.5, p_star=0.5)
        
        # Check linear scaling
        assert pytest.approx(L2 - L1, rel=1e-10) == 1.0  # Δa_tilde
        assert pytest.approx(U2 - U1, rel=1e-10) == 1.0  # Δa_tilde
        
        # Width should be positive for all valid parameters
        for a in np.linspace(0.1, 2.0, 5):
            for b_val in np.linspace(0, 1.0, 5):
                for g in np.linspace(0, 1.0, 5):
                    for p in np.linspace(0.1, 0.9, 5):
                        L, U, info = compute_LU(a, b_val, g, p)
                        assert U > L, f"Failed for a={a}, b={b_val}, γ={g}, p={p}"
                        assert info['nonempty'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])