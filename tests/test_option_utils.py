"""Tests for option utilities."""

import pytest
import numpy as np
from numpy import log, exp, sqrt
from src.utils.option_utils import (
    call_payoff,
    put_payoff,
    black_scholes_call,
    black_scholes_put,
    get_payoff_func,
    OptionType,
    compute_option_price,
    compute_greeks,
    check_early_exercise
)

class TestPayoffFunctions:
    """Tests for option payoff functions."""
    
    @pytest.mark.parametrize("spot,strike,expected", [
        (110, 100, 10),  # ITM call
        (90, 100, 0),    # OTM call
        (100, 100, 0),   # ATM call
        (0, 100, 0),     # Zero spot
        (1000, 100, 900) # Deep ITM
    ])
    def test_call_payoff(self, spot, strike, expected):
        """Test call option payoff function."""
        assert call_payoff(spot, strike) == expected
        # Test array input
        spots = np.array([spot, spot])
        np.testing.assert_array_equal(call_payoff(spots, strike), np.array([expected, expected]))

    @pytest.mark.parametrize("spot,strike,expected", [
        (90, 100, 10),   # ITM put
        (110, 100, 0),   # OTM put
        (100, 100, 0),   # ATM put
        (0, 100, 100),   # Zero spot
        (1000, 100, 0)   # Deep OTM
    ])
    def test_put_payoff(self, spot, strike, expected):
        """Test put option payoff function."""
        assert put_payoff(spot, strike) == expected
        # Test array input
        spots = np.array([spot, spot])
        np.testing.assert_array_equal(put_payoff(spots, strike), np.array([expected, expected]))

class TestBlackScholes:
    """Tests for Black-Scholes pricing functions."""
    
    @pytest.mark.parametrize("spot,strike,time,rate,div,vol", [
        (100, 100, 1, 0.05, 0.02, 0.2),  # ATM
        (110, 100, 1, 0.05, 0.02, 0.2),  # ITM
        (90, 100, 1, 0.05, 0.02, 0.2),   # OTM
        (100, 100, 0, 0.05, 0.02, 0.2),  # Zero time
        (100, 100, 10, 0.05, 0.02, 0.2), # Long time
    ])
    def test_black_scholes_values(self, spot, strike, time, rate, div, vol):
        """Test Black-Scholes values under different conditions."""
        call = black_scholes_call(spot, strike, time, rate, div, vol)
        put = black_scholes_put(spot, strike, time, rate, div, vol)
        
        # Basic value checks
        assert call >= 0
        assert put >= 0
        
        # Put-call parity
        parity_diff = call - put - spot * exp(-div*time) + strike * exp(-rate*time)
        assert abs(parity_diff) < 1e-10
        
        # At expiry
        if time == 0:
            assert abs(call - max(spot - strike, 0)) < 1e-10
            assert abs(put - max(strike - spot, 0)) < 1e-10

    @pytest.mark.parametrize("vol", [0.1, 0.2, 0.3, 0.4])
    def test_volatility_impact(self, vol):
        """Test impact of volatility on option prices."""
        call = black_scholes_call(100, 100, 1, 0.05, 0.02, vol)
        put = black_scholes_put(100, 100, 1, 0.05, 0.02, vol)
        
        call_high = black_scholes_call(100, 100, 1, 0.05, 0.02, vol + 0.1)
        put_high = black_scholes_put(100, 100, 1, 0.05, 0.02, vol + 0.1)
        
        # Higher volatility should lead to higher prices
        assert call_high > call
        assert put_high > put

class TestOptionPricing:
    """Tests for general option pricing functions."""
    
    @pytest.mark.parametrize("option_type", list(OptionType))
    def test_option_types(self, option_type):
        """Test pricing for all option types."""
        price = compute_option_price(
            S=100, K=100, t=1, r=0.05, q=0.02, sigma=0.2,
            option_type=option_type
        )
        assert price > 0
        
        # Test at expiry
        price_expiry = compute_option_price(
            S=100, K=100, t=0, r=0.05, q=0.02, sigma=0.2,
            option_type=option_type
        )
        if 'CALL' in option_type.name:
            assert abs(price_expiry - max(100 - 100, 0)) < 1e-10
        else:
            assert abs(price_expiry - max(100 - 100, 0)) < 1e-10

    def test_american_vs_european(self):
        """Test that American options are worth at least as much as European."""
        params = dict(S=100, K=100, t=1, r=0.05, q=0.02, sigma=0.2)
        
        # Calls with no dividends
        euro_call = compute_option_price(option_type=OptionType.EUROPEAN_CALL, **params)
        amer_call = compute_option_price(option_type=OptionType.AMERICAN_CALL, **params)
        assert abs(euro_call - amer_call) < 1e-10  # Should be equal with no dividends
        
        # Puts (American worth more)
        euro_put = compute_option_price(option_type=OptionType.EUROPEAN_PUT, **params)
        amer_put = compute_option_price(option_type=OptionType.AMERICAN_PUT, **params)
        assert amer_put > euro_put

class TestGreeks:
    """Tests for option Greeks computation."""
    
    @pytest.mark.parametrize("option_type,expected_sign", [
        (OptionType.EUROPEAN_CALL, 1),
        (OptionType.EUROPEAN_PUT, -1),
        (OptionType.AMERICAN_CALL, 1),
        (OptionType.AMERICAN_PUT, -1)
    ])
    def test_delta_signs(self, option_type, expected_sign):
        """Test that deltas have correct signs."""
        delta, _ = compute_greeks(
            S=100, K=100, t=1, r=0.05, q=0.02, sigma=0.2,
            option_type=option_type
        )
        assert (delta > 0) == (expected_sign > 0)

    @pytest.mark.parametrize("option_type", list(OptionType))
    def test_gamma_positivity(self, option_type):
        """Test that gamma is always positive."""
        _, gamma = compute_greeks(
            S=100, K=100, t=1, r=0.05, q=0.02, sigma=0.2,
            option_type=option_type
        )
        assert gamma > 0

class TestEarlyExercise:
    """Tests for early exercise conditions."""
    
    @pytest.mark.parametrize("option_type,div,expected", [
        (OptionType.AMERICAN_CALL, 0.0, False),
        (OptionType.AMERICAN_CALL, 0.1, True),
        (OptionType.AMERICAN_PUT, 0.0, True),
        (OptionType.EUROPEAN_CALL, 0.1, False),
        (OptionType.EUROPEAN_PUT, 0.1, False)
    ])
    def test_early_exercise_conditions(self, option_type, div, expected):
        """Test early exercise conditions for different options."""
        result = check_early_exercise(
            S=100, K=100, t=1, r=0.05, q=div, sigma=0.2,
            option_type=option_type
        )
        assert result == expected 