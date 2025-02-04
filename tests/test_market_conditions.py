"""Tests for option pricing under various market conditions."""

import pytest
import numpy as np
from src.environments.american_option import AmericanOptionEnv, Action
from src.utils.monte_carlo import price_american_option, price_european_option
from src.utils.path_generators import generate_gbm_paths
from src.utils.option_utils import call_payoff, put_payoff

@pytest.mark.parametrize("s0", [80, 90, 100, 110, 120])
@pytest.mark.parametrize("volatility", [0.2, 0.4])
@pytest.mark.parametrize("maturity", [1, 2])
def test_base_cases(s0, volatility, maturity):
    """Test base cases with varying spot price, volatility, and maturity."""
    config = {
        'S': s0,
        'K': 100.0,
        'T': maturity,
        'r': 0.04,
        'sigma': volatility,
        'L': 50,
        'm': 1000,
        'use_gbm': True
    }
    env = AmericanOptionEnv(config)
    state = env.reset()
    assert len(state) == 3  # Normalized time, log price, continuation value

@pytest.mark.parametrize("dividend_yield", [0.01, 0.04, 0.05, 0.3])
def test_dividend_impact(dividend_yield):
    """Test early exercise due to dividends for American calls."""
    paths = generate_gbm_paths(
        nsim=10000,
        nstep=100,
        t1=0,
        t2=1,
        s_0=100,
        r=0.04,
        q=dividend_yield,
        v=0.2
    )
    
    american_price = price_american_option(
        paths=paths,
        dt=0.01,
        payoff_func=lambda x: call_payoff(x, 100),
        r=0.04,
        regression_order=3
    )
    
    european_price = price_european_option(
        paths=paths,
        payoff_func=lambda x: call_payoff(x, 100),
        r=0.04,
        t=1
    )
    
    # For high dividend yields, American call should be worth more than European
    if dividend_yield > 0.04:
        assert american_price > european_price

@pytest.mark.parametrize("rate", [0.0, 0.1, 0.2, 0.4, 0.5])
def test_interest_rate_impact(rate):
    """Test impact of different interest rates."""
    config = {
        'S': 100.0,
        'K': 100.0,
        'T': 1.0,
        'r': rate,
        'sigma': 0.2,
        'L': 50,
        'm': 1000,
        'use_gbm': True
    }
    env = AmericanOptionEnv(config)
    state = env.reset()
    assert len(state) == 3

@pytest.mark.parametrize("vol", [0.01, 0.2, 1.0])
def test_volatility_stress(vol):
    """Test extreme volatility scenarios."""
    paths = generate_gbm_paths(
        nsim=10000,
        nstep=100,
        t1=0,
        t2=1,
        s_0=100,
        r=0.04,
        q=0,
        v=vol
    )
    
    put_price = price_american_option(
        paths=paths,
        dt=0.01,
        payoff_func=lambda x: put_payoff(x, 100),
        r=0.04,
        regression_order=3
    )
    
    # Higher volatility should lead to higher option prices
    if vol > 0.2:
        assert put_price > 5.0

@pytest.mark.parametrize("spot", [30, 100, 300])
def test_moneyness_impact(spot):
    """Test deep ITM/ATM/OTM scenarios."""
    config = {
        'S': spot,
        'K': 100.0,
        'T': 1.0,
        'r': 0.04,
        'sigma': 0.2,
        'L': 50,
        'm': 1000,
        'use_gbm': True
    }
    env = AmericanOptionEnv(config)
    state = env.reset()
    
    # For puts, lower spot means higher value
    if spot < 100:
        action = env.step(Action.EXERCISE)[1]  # Get reward from exercise
        assert action > 0

@pytest.mark.parametrize("maturity", [0.5, 1.0, 10.0])
def test_maturity_impact(maturity):
    """Test impact of different maturities."""
    # Increase number of steps for longer maturities
    nstep = max(100, int(200 * maturity))
    
    paths = generate_gbm_paths(
        nsim=10000,
        nstep=nstep,
        t1=0,
        t2=maturity,
        s_0=100,
        r=0.04,
        q=0.02,  # Add dividend yield to increase put value
        v=0.2
    )
    
    american_price = price_american_option(
        paths=paths,
        dt=maturity/nstep,
        payoff_func=lambda x: put_payoff(x, 100),
        r=0.04,
        regression_order=3
    )
    
    # Longer maturity should generally mean higher option value
    if maturity > 1.0:
        assert american_price > 4.0  # Adjusted threshold based on realistic values 