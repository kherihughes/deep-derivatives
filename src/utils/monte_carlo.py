"""Monte Carlo pricing utilities for option valuation."""

import numpy as np
from typing import Callable, Optional, Union, Tuple
from .path_generators import generate_gbm_paths, generate_heston_paths
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def compute_mc_price(
    prices: np.ndarray,
    t1: float,
    t2: float,
    payoff_func: Callable,
    strike: float,
    risk_free_rate: float,
    regression_order: int = 3
) -> float:
    """
    Compute option price using Monte Carlo simulation with Longstaff-Schwartz method.
    
    Args:
        prices: Array of simulated price paths
        t1: Initial time
        t2: Final time
        payoff_func: Option payoff function
        strike: Strike price
        risk_free_rate: Risk-free rate
        regression_order: Order of polynomial regression for continuation value
        
    Returns:
        Option price
    """
    nstep = prices.shape[1]
    dt = (t2 - t1) / nstep
    
    # Initialize with terminal payoff
    values = payoff_func(prices[:, -1], strike)
    
    # Backward induction
    for t in range(nstep-2, -1, -1):
        # Identify in-the-money paths
        itm = payoff_func(prices[:, t], strike) > 0
        if not np.any(itm):
            continue
        
        # Compute continuation value using polynomial regression
        X = np.vander(prices[itm, t], regression_order + 1)
        coeff = np.linalg.lstsq(X, np.exp(-risk_free_rate * dt) * values[itm], rcond=None)[0]
        
        continuation_values = np.dot(np.vander(prices[:, t], regression_order + 1), coeff)
        exercise_values = payoff_func(prices[:, t], strike)
        
        # Update values based on optimal exercise decision
        values = np.where(itm, np.maximum(exercise_values, continuation_values), values)
    
    return np.exp(-risk_free_rate * dt) * values.mean()

def price_american_option(
    paths: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    s0: Optional[float] = None,
    strike: Optional[float] = None,
    t1: Optional[float] = None,
    t2: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    volatility: Optional[float] = None,
    payoff_func: Optional[Callable] = None,
    nsim: Optional[int] = None,
    nstep: Optional[int] = None,
    dt: Optional[float] = None,
    r: Optional[float] = None,
    regression_order: int = 3,
    use_gbm: bool = True
) -> float:
    """
    Price American option using Longstaff-Schwartz regression method.
    
    Can be called in two ways:
    1. With pre-generated paths:
        price_american_option(paths=paths, dt=dt, payoff_func=payoff_func, r=r)
    2. With parameters to generate paths:
        price_american_option(s0=s0, strike=strike, t1=t1, t2=t2, ...)
    
    Args:
        paths: Pre-generated price paths (optional)
        s0: Initial stock price (optional)
        strike: Strike price (optional)
        t1: Start time (optional)
        t2: End time (optional)
        risk_free_rate: Risk-free rate (optional)
        dividend_yield: Dividend yield (optional)
        volatility: Volatility (optional)
        payoff_func: Option payoff function (optional)
        nsim: Number of simulations (optional)
        nstep: Number of time steps (optional)
        dt: Time step size (optional)
        r: Risk-free rate (optional)
        regression_order: Order of polynomial regression
        use_gbm: Whether to use GBM model
    
    Returns:
        Option price
    """
    # Handle both function signatures
    if paths is not None:
        # Use pre-generated paths
        if dt is None or r is None or payoff_func is None:
            raise ValueError("When using pre-generated paths, must provide dt, r, and payoff_func")
        r_val = r
        dt_val = dt
        # Handle both array and tuple return from path generators
        paths_val = paths[0] if isinstance(paths, tuple) else paths
        payoff_func_val = payoff_func
    else:
        # Generate paths
        if any(x is None for x in [s0, strike, t1, t2, risk_free_rate, dividend_yield, volatility, payoff_func, nsim, nstep]):
            raise ValueError("When generating paths, must provide all parameters")
        
        dt_val = (t2 - t1) / nstep
        r_val = risk_free_rate
        
        if use_gbm:
            paths_val = generate_gbm_paths(
                nsim=nsim,
                nstep=nstep,
                t1=t1,
                t2=t2,
                s_0=s0,
                r=risk_free_rate,
                q=dividend_yield,
                v=volatility
            )
        else:
            paths_val = generate_heston_paths(
                nsim=nsim,
                nstep=nstep,
                t1=t1,
                t2=t2,
                s_0=s0,
                r=risk_free_rate,
                q=dividend_yield,
                v_0=volatility**2,
                theta=volatility**2,
                rho=-0.7,
                kappa=2.0,
                sigma=volatility
            )
        
        payoff_func_val = lambda x: payoff_func(x, strike)
    
    n_paths, n_steps = paths_val.shape[0], paths_val.shape[1] - 1
    
    # Initialize arrays
    exercise_values = np.zeros((n_paths, n_steps + 1))
    continuation_values = np.zeros((n_paths, n_steps + 1))
    exercise_flags = np.zeros((n_paths, n_steps + 1), dtype=bool)
    
    # Terminal payoffs
    exercise_values[:, -1] = payoff_func_val(paths_val[:, -1])
    exercise_flags[:, -1] = True
    
    # Store optimal stopping values
    stopping_values = np.zeros(n_paths)
    stopping_values[:] = exercise_values[:, -1]
    
    # Backward induction
    for t in range(n_steps-1, -1, -1):
        # Current exercise values
        exercise_values[:, t] = payoff_func_val(paths_val[:, t])
        
        # Identify in-the-money paths
        itm = exercise_values[:, t] > 0
        
        if np.sum(itm) > 0:
            # Prepare regression data
            X = paths_val[itm, t].reshape(-1, 1)
            poly = PolynomialFeatures(degree=regression_order)
            X = poly.fit_transform(X)
            
            # Future discounted cash flows
            future_values = stopping_values[itm] * np.exp(-r_val * dt_val)
            
            # Fit regression
            reg = LinearRegression()
            reg.fit(X, future_values)
            
            # Predict continuation values
            continuation_values[itm, t] = reg.predict(X)
            
            # Update optimal stopping values where exercise is better
            exercise_better = exercise_values[itm, t] > continuation_values[itm, t]
            stopping_values[itm] = np.where(
                exercise_better,
                exercise_values[itm, t],
                stopping_values[itm]
            )
            
            # Record exercise decisions
            exercise_flags[itm, t] = exercise_better
            
            # Clear future exercise opportunities for paths where we exercise
            exercise_flags[itm & exercise_flags[:, t], t+1:] = False
    
    # Calculate option value using first optimal exercise
    option_values = np.zeros(n_paths)
    for i in range(n_paths):
        exercise_times = np.where(exercise_flags[i])[0]
        if len(exercise_times) > 0:
            t = exercise_times[0]  # Take first exercise opportunity
            option_values[i] = exercise_values[i, t] * np.exp(-r_val * t * dt_val)
    
    return np.mean(option_values)

def price_european_option(
    paths: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    payoff_func: Callable,
    r: float,
    t: float
) -> float:
    """Price a European option using Monte Carlo simulation.
    
    Args:
        paths: Array of simulated price paths or tuple of (price_paths, variance_paths)
        payoff_func: Function that takes price and returns payoff
        r: Risk-free rate
        t: Time to maturity
    
    Returns:
        float: Option price
    """
    # Handle both array and tuple return from path generators
    price_paths = paths[0] if isinstance(paths, tuple) else paths
    
    # For European options, we only need the terminal values
    terminal_values = price_paths[:, -1]
    
    # Calculate payoffs at maturity
    payoffs = payoff_func(terminal_values)
    
    # Discount payoffs back to present value
    option_price = np.exp(-r * t) * np.mean(payoffs)
    
    return option_price 