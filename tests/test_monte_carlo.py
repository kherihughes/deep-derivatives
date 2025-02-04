"""
Tests for Monte Carlo pricing utilities.
"""

import numpy as np
import pytest
from numpy import log, exp, sqrt

from src.utils.monte_carlo import compute_mc_price, price_american_option, price_european_option
from src.utils.path_generators import generate_gbm_paths
from src.utils.option_utils import call_payoff, put_payoff

def test_compute_mc_price():
    """Test Monte Carlo price computation."""
    # Generate paths
    paths = generate_gbm_paths(
        nsim=10000,
        nstep=100,
        t1=0,
        t2=1,
        s_0=100,
        r=0.05,
        q=0.02,
        v=0.2
    )
    
    # Price put option
    put_price = compute_mc_price(
        prices=paths,
        t1=0,
        t2=1,
        payoff_func=put_payoff,
        strike=100,
        risk_free_rate=0.05,
        regression_order=3
    )
    assert put_price > 0
    
    # Price call option
    call_price = compute_mc_price(
        prices=paths,
        t1=0,
        t2=1,
        payoff_func=call_payoff,
        strike=100,
        risk_free_rate=0.05,
        regression_order=3
    )
    assert call_price > 0

def test_american_option_pricing():
    """Test American option pricing."""
    # Price American put
    put_price = price_american_option(
        s0=100,
        strike=100,
        t1=0,
        t2=1,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        volatility=0.2,
        payoff_func=put_payoff,
        nsim=10000,
        nstep=100,
        regression_order=3,
        use_gbm=True
    )
    assert put_price > 0
    
    # Price American call
    call_price = price_american_option(
        s0=100,
        strike=100,
        t1=0,
        t2=1,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        volatility=0.2,
        payoff_func=call_payoff,
        nsim=10000,
        nstep=100,
        regression_order=3,
        use_gbm=True
    )
    assert call_price > 0

def test_early_exercise_premium():
    """Test early exercise premium for American put."""
    # Parameters for deep ITM put with high dividend
    s0 = 100
    strike = 300  # Very deep ITM
    t1 = 0
    t2 = 10  # Very long maturity
    risk_free_rate = 0.02
    dividend_yield = 0.2  # Very high dividend
    volatility = 0.8  # Very high volatility
    dt = (t2 - t1) / 200  # Match nstep
    
    # Generate paths for American option
    paths = generate_gbm_paths(
        nsim=200000,  # Many more paths for stability
        nstep=200,    # Keep same number of steps
        t1=t1,
        t2=t2,
        s_0=s0,
        r=risk_free_rate,
        q=dividend_yield,
        v=volatility
    )
    
    # Price American put
    american_price = price_american_option(
        paths=paths,
        dt=dt,
        payoff_func=lambda x: put_payoff(x, strike),
        r=risk_free_rate,
        regression_order=3
    )
    
    # Generate paths for European price
    paths = generate_gbm_paths(
        nsim=200000,  # Many more paths for stability
        nstep=1,      # Single step for European
        t1=t1,
        t2=t2,
        s_0=s0,
        r=risk_free_rate,
        q=dividend_yield,
        v=volatility
    )
    
    # Price European put
    european_price = price_european_option(
        paths=paths,
        payoff_func=lambda x: put_payoff(x, strike),
        r=risk_free_rate,
        t=t2
    )
    
    # American put should be worth more than European
    assert american_price > european_price
    # Premium should be significant
    assert american_price - european_price > 0.5

def test_regression_order():
    """Test impact of regression order on price."""
    # Generate paths
    paths = generate_gbm_paths(
        nsim=10000,
        nstep=50,
        t1=0,
        t2=1,
        s_0=100,
        r=0.05,
        q=0.02,
        v=0.2
    )
    
    # Price with different regression orders
    prices = []
    for order in [2, 3, 4]:
        price = compute_mc_price(
            prices=paths,
            t1=0,
            t2=1,
            payoff_func=put_payoff,
            strike=100,
            risk_free_rate=0.05,
            regression_order=order
        )
        prices.append(price)
    
    # Prices should be similar across regression orders
    assert max(prices) - min(prices) < 0.5 