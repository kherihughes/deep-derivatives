"""Tests for option pricing models and path generation."""

import pytest
import numpy as np
from src.utils.path_generators import generate_gbm_paths, generate_heston_paths
from src.utils.monte_carlo import price_american_option, price_european_option
from src.utils.option_utils import call_payoff, put_payoff

# Common test parameters
@pytest.fixture
def common_params():
    """Common parameters for testing."""
    return {
        'k': 100,
        'nsim': 10000,
        'nstep': 100,
        'q': 0,
        't1': 0,
        't2': 1,
        's_0': 100,
        'r': 0.04
    }

@pytest.fixture
def heston_params():
    """Heston model parameters for testing."""
    return {
        'v_0': 0.04,
        'theta': 0.04,
        'rho': -0.7,
        'kappa': 2.0,
        'sigma': 0.2
    }

def test_gbm_path_properties(common_params):
    """Test statistical properties of GBM paths."""
    v = 0.2  # volatility
    paths = generate_gbm_paths(v=v, **common_params)
    
    # Test shape
    assert paths.shape == (common_params['nsim'], common_params['nstep'] + 1)
    
    # Test initial value
    np.testing.assert_allclose(paths[:, 0], common_params['s_0'])
    
    # Test log-normality
    log_returns = np.log(paths[:, -1] / common_params['s_0'])
    mu = (common_params['r'] - common_params['q'] - v*v/2) * common_params['t2']
    sigma = v * np.sqrt(common_params['t2'])
    
    # Test mean and std of log returns (within 3 standard errors)
    assert abs(np.mean(log_returns) - mu) < 3 * sigma / np.sqrt(common_params['nsim'])
    assert abs(np.std(log_returns) - sigma) < 3 * sigma / np.sqrt(2*common_params['nsim'])

def test_heston_path_properties(common_params, heston_params):
    """Test statistical properties of Heston paths."""
    price_paths, variance_paths = generate_heston_paths(**common_params, **heston_params)
    
    # Test shape
    assert price_paths.shape == (common_params['nsim'], common_params['nstep'] + 1)
    assert variance_paths.shape == (common_params['nsim'], common_params['nstep'] + 1)
    
    # Test initial values
    np.testing.assert_allclose(price_paths[:, 0], common_params['s_0'])
    np.testing.assert_allclose(variance_paths[:, 0], heston_params['v_0'])
    
    # Test mean of returns (should be close to risk-free rate)
    returns = price_paths[:, -1] / common_params['s_0'] - 1
    expected_return = (common_params['r'] - common_params['q']) * common_params['t2']
    assert abs(np.mean(returns) - expected_return) < 0.1

@pytest.mark.parametrize("model", ["gbm", "heston"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_vs_european(common_params, heston_params, model, option_type):
    """Test that American options are worth at least as much as European options."""
    # Generate paths
    if model == "gbm":
        paths = generate_gbm_paths(v=0.2, **common_params)
    else:
        paths = generate_heston_paths(**common_params, **heston_params)
    
    payoff_func = call_payoff if option_type == "call" else put_payoff
    
    # Price both options
    american_price = price_american_option(
        paths=paths,
        dt=common_params['t2']/common_params['nstep'],
        payoff_func=lambda x: payoff_func(x, common_params['k']),
        r=common_params['r']
    )
    
    european_price = price_european_option(
        paths=paths,
        payoff_func=lambda x: payoff_func(x, common_params['k']),
        r=common_params['r'],
        t=common_params['t2']
    )
    
    # American options should be worth at least as much as European
    assert american_price >= european_price * 0.99  # Allow for small numerical errors

@pytest.mark.parametrize("model", ["gbm", "heston"])
def test_put_call_parity_european(common_params, heston_params, model):
    """Test put-call parity for European options."""
    # Use more simulations for better accuracy
    params = common_params.copy()
    params['nsim'] = 50000  # Increased from 10000
    
    # Generate paths
    if model == "gbm":
        # Generate paths with fixed random seed for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(42)
        
        # Generate first set of paths
        paths = generate_gbm_paths(v=0.2, **params)
        
        # Restore random state
        np.random.set_state(rng_state)
    else:
        # For Heston, use more paths
        paths = generate_heston_paths(**params, **heston_params)
    
    # Price call and put
    call_price = price_european_option(
        paths=paths,
        payoff_func=lambda x: call_payoff(x, params['k']),
        r=params['r'],
        t=params['t2']
    )
    
    put_price = price_european_option(
        paths=paths,
        payoff_func=lambda x: put_payoff(x, params['k']),
        r=params['r'],
        t=params['t2']
    )
    
    # Put-call parity
    s0 = params['s_0']
    k = params['k']
    r = params['r']
    q = params['q']
    t = params['t2']
    
    lhs = call_price - put_price
    rhs = s0 * np.exp(-q*t) - k * np.exp(-r*t)
    
    # Increase tolerance for numerical errors
    assert abs(lhs - rhs) < 0.2  # Increased tolerance and using more paths

@pytest.mark.parametrize("model", ["gbm", "heston"])
def test_early_exercise_dividend(common_params, heston_params, model):
    """Test early exercise premium for American calls with dividends."""
    params = common_params.copy()
    params['q'] = 0.06  # High dividend yield
    
    # Generate paths
    if model == "gbm":
        paths = generate_gbm_paths(v=0.2, **params)
    else:
        paths = generate_heston_paths(**params, **heston_params)
    
    # Price American and European calls
    american_price = price_american_option(
        paths=paths,
        dt=params['t2']/params['nstep'],
        payoff_func=lambda x: call_payoff(x, params['k']),
        r=params['r']
    )
    
    european_price = price_european_option(
        paths=paths,
        payoff_func=lambda x: call_payoff(x, params['k']),
        r=params['r'],
        t=params['t2']
    )
    
    # With high dividends, American calls should be worth more than European
    assert american_price > european_price

@pytest.mark.parametrize("model", ["gbm", "heston"])
@pytest.mark.parametrize("moneyness", [0.8, 1.0, 1.2])
def test_option_moneyness(common_params, heston_params, model, moneyness):
    """Test option prices for different moneyness levels."""
    params = common_params.copy()
    params['s_0'] = 100 * moneyness  # Adjust spot price for moneyness
    
    # Generate paths
    if model == "gbm":
        paths = generate_gbm_paths(v=0.2, **params)
    else:
        paths = generate_heston_paths(**params, **heston_params)
    
    # Price put options
    put_price = price_american_option(
        paths=paths,
        dt=params['t2']/params['nstep'],
        payoff_func=lambda x: put_payoff(x, params['k']),
        r=params['r']
    )
    
    # ITM puts should be worth more than OTM puts
    if moneyness < 1:
        assert put_price > 5.0
    elif moneyness > 1:
        assert put_price < 5.0

@pytest.mark.parametrize("model", ["gbm", "heston"])
@pytest.mark.parametrize("vol", [0.1, 0.3, 0.5])
def test_volatility_sensitivity(common_params, heston_params, model, vol):
    """Test sensitivity to volatility changes."""
    if model == "gbm":
        paths = generate_gbm_paths(v=vol, **common_params)
    else:
        params = heston_params.copy()
        # Adjust Heston parameters to maintain Feller condition
        params['v_0'] = vol * vol
        params['theta'] = vol * vol
        params['kappa'] = max(2.0, (params['sigma'] ** 2) / (2 * params['theta']) + 0.1)
        paths = generate_heston_paths(**common_params, **params)
    
    # Price options
    call_price = price_american_option(
        paths=paths,
        dt=common_params['t2']/common_params['nstep'],
        payoff_func=lambda x: call_payoff(x, common_params['k']),
        r=common_params['r']
    )
    
    # Higher volatility should lead to higher option prices
    if vol == 0.1:
        assert call_price < 10.0
    elif vol == 0.5:
        assert call_price > 15.0

@pytest.mark.parametrize("model", ["gbm", "heston"])
@pytest.mark.parametrize("maturity", [0.25, 1.0, 2.0])
def test_term_structure(common_params, heston_params, model, maturity):
    """Test term structure of option prices."""
    params = common_params.copy()
    params['t2'] = maturity
    params['nstep'] = max(100, int(200 * maturity))  # More steps for longer maturities
    
    if model == "gbm":
        paths = generate_gbm_paths(v=0.2, **params)
    else:
        paths = generate_heston_paths(**params, **heston_params)
    
    # Price ATM calls
    call_price = price_american_option(
        paths=paths,
        dt=params['t2']/params['nstep'],
        payoff_func=lambda x: call_payoff(x, params['k']),
        r=params['r']
    )
    
    # Longer maturity should generally mean higher option value
    if maturity == 0.25:
        assert call_price < 10.0
    elif maturity == 2.0:
        assert call_price > 14.0  # Adjusted threshold 