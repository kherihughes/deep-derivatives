"""
Path generation utilities for option pricing models.
Includes implementations of:
1. Geometric Brownian Motion (GBM)
2. Heston Stochastic Volatility Model
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union


def generate_gbm_paths(
    nsim: int,
    nstep: int,
    t1: float,
    t2: float,
    s_0: float,
    r: float,
    q: float,
    v: float,
    **kwargs
) -> np.ndarray:
    """
    Generate paths using Geometric Brownian Motion.
    
    Args:
        nsim: Number of simulations
        nstep: Number of steps
        t1: Start time
        t2: End time
        s_0: Initial stock price
        r: Risk-free rate
        q: Dividend yield
        v: Volatility
        
    Returns:
        Array of shape (nsim, nstep+1) containing simulated paths
    """
    dt = (t2 - t1) / nstep
    paths = np.zeros((nsim, nstep + 1))
    paths[:, 0] = s_0
    
    # Generate random normal variables
    z = np.random.normal(size=(nsim, nstep))
    
    # Simulate paths
    for t in range(nstep):
        paths[:, t + 1] = paths[:, t] * np.exp(
            (r - q - 0.5 * v ** 2) * dt + v * np.sqrt(dt) * z[:, t]
        )
    
    return paths


def generate_heston_paths(
    nsim: int,
    nstep: int,
    t1: float,
    t2: float,
    s_0: float,
    r: float,
    q: float,
    v_0: float,
    theta: float,
    rho: float,
    kappa: float,
    sigma: float,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate paths using Heston Stochastic Volatility Model.
    
    Args:
        nsim: Number of simulations
        nstep: Number of steps
        t1: Start time
        t2: End time
        s_0: Initial stock price
        r: Risk-free rate
        q: Dividend yield
        v_0: Initial variance
        theta: Long-term variance
        rho: Correlation between stock and variance processes
        kappa: Rate of mean reversion
        sigma: Volatility of variance
        
    Returns:
        Tuple of (price_paths, variance_paths) each of shape (nsim, nstep+1)
    """
    # Check Feller condition
    if 2 * kappa * theta <= sigma ** 2:
        raise ValueError('Feller condition not met in generate_heston_paths.')
    
    dt = (t2 - t1) / nstep
    price_paths = np.zeros((nsim, nstep + 1))
    variance_paths = np.zeros((nsim, nstep + 1))
    
    # Initialize paths
    price_paths[:, 0] = s_0
    variance_paths[:, 0] = v_0
    
    # Generate correlated random variables
    z1 = np.random.normal(size=(nsim, nstep))
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=(nsim, nstep))
    
    # Simulate paths using full truncation scheme
    for t in range(nstep):
        # Ensure variance stays positive
        variance_paths_t = np.maximum(variance_paths[:, t], 0)
        
        # Update variance
        variance_paths[:, t + 1] = variance_paths[:, t] + kappa * (
            theta - variance_paths_t
        ) * dt + sigma * np.sqrt(variance_paths_t * dt) * z2[:, t]
        
        # Update price
        price_paths[:, t + 1] = price_paths[:, t] * np.exp(
            (r - q - 0.5 * variance_paths_t) * dt + 
            np.sqrt(variance_paths_t * dt) * z1[:, t]
        )
    
    return price_paths, variance_paths


def generate_paths(
    nsim: int,
    nstep: int,
    t1: float,
    t2: float,
    s_0: float,
    r: float,
    q: float,
    path_kwargs: Dict,
    use_gbm: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate paths using either GBM or Heston model.
    
    Args:
        nsim: Number of simulations
        nstep: Number of steps
        t1: Start time
        t2: End time
        s_0: Initial stock price
        r: Risk-free rate
        q: Dividend yield
        path_kwargs: Model-specific parameters
        use_gbm: Whether to use GBM (True) or Heston (False)
        
    Returns:
        For GBM: Array of shape (nsim, nstep+1)
        For Heston: Tuple of (price_paths, variance_paths) each of shape (nsim, nstep+1)
    """
    if use_gbm:
        return generate_gbm_paths(
            nsim=nsim,
            nstep=nstep,
            t1=t1,
            t2=t2,
            s_0=s_0,
            r=r,
            q=q,
            **path_kwargs
        )
    else:
        return generate_heston_paths(
            nsim=nsim,
            nstep=nstep,
            t1=t1,
            t2=t2,
            s_0=s_0,
            r=r,
            q=q,
            **path_kwargs
        ) 