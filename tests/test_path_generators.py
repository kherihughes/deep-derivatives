"""
Tests for path generation utilities.
"""

import numpy as np
import pytest
from numpy import log, exp, sqrt

from src.utils.path_generators import generate_gbm_paths, generate_heston_paths

def test_gbm_paths_shape():
    """Test GBM path generation shape."""
    nsim = 1000
    nstep = 252
    paths = generate_gbm_paths(
        nsim=nsim,
        nstep=nstep,
        t1=0,
        t2=1,
        s_0=100,
        r=0.05,
        q=0.02,
        v=0.2
    )
    assert paths.shape == (nsim, nstep + 1)
    assert np.all(paths[:, 0] == 100)

def test_gbm_paths_statistics():
    """Test GBM path statistics."""
    nsim = 10000
    nstep = 1
    s_0 = 100
    r = 0.05
    q = 0.02
    v = 0.2
    t2 = 1
    
    paths = generate_gbm_paths(
        nsim=nsim,
        nstep=nstep,
        t1=0,
        t2=t2,
        s_0=s_0,
        r=r,
        q=q,
        v=v
    )
    
    # Test log returns
    log_returns = log(paths[:, -1] / s_0)
    
    # Expected mean and variance of log returns
    expected_mean = (r - q - v*v/2) * t2
    expected_var = v*v * t2
    
    # Check statistics with 3 standard deviations
    assert abs(np.mean(log_returns) - expected_mean) < 3 * sqrt(expected_var / nsim)
    assert abs(np.var(log_returns) - expected_var) < 3 * expected_var * sqrt(2/(nsim-1))

def test_heston_paths_shape():
    """Test Heston path generation shape."""
    nsim = 1000
    nstep = 252
    stock_paths, var_paths = generate_heston_paths(
        nsim=nsim,
        nstep=nstep,
        t1=0,
        t2=1,
        s_0=100,
        r=0.05,
        q=0.02,
        v_0=0.04,
        theta=0.04,
        rho=-0.7,
        kappa=5.0,
        sigma=0.3
    )
    # Check stock price paths
    assert stock_paths.shape == (nsim, nstep + 1)
    assert np.all(stock_paths[:, 0] == 100)
    # Check variance paths
    assert var_paths.shape == (nsim, nstep + 1)
    assert np.all(var_paths[:, 0] == 0.04)

def test_heston_feller_condition():
    """Test Heston Feller condition check."""
    with pytest.raises(ValueError, match='Feller condition not met in generate_heston_paths'):
        generate_heston_paths(
            nsim=100,
            nstep=252,
            t1=0,
            t2=1,
            s_0=100,
            r=0.05,
            q=0.02,
            v_0=0.04,
            theta=0.04,
            rho=-0.7,
            kappa=0.1,  # Low kappa
            sigma=0.3   # High sigma
        )

def test_heston_paths_positivity():
    """Test Heston paths remain positive."""
    nsim = 1000
    nstep = 252
    stock_paths, var_paths = generate_heston_paths(
        nsim=nsim,
        nstep=nstep,
        t1=0,
        t2=1,
        s_0=100,
        r=0.05,
        q=0.02,
        v_0=0.04,
        theta=0.04,
        rho=-0.7,
        kappa=5.0,
        sigma=0.2
    )
    # Check stock price paths positivity
    assert np.all(stock_paths > 0)
    # Check variance paths positivity
    assert np.all(var_paths >= 0)