"""
Utility functions for option pricing and early exercise conditions.
"""

import numpy as np
from scipy.stats import norm
from enum import Enum
from typing import Tuple, Optional, Union, Callable

class OptionType(Enum):
    """Enum for option types."""
    EUROPEAN_CALL = "European_Call"
    EUROPEAN_PUT = "European_Put"
    AMERICAN_CALL = "American_Call"
    AMERICAN_PUT = "American_Put"

def call_payoff(x: Union[float, np.ndarray], k: float) -> Union[float, np.ndarray]:
    """
    Calculate call option payoff.
    
    Args:
        x: Stock price(s)
        k: Strike price
    
    Returns:
        Call option payoff(s)
    """
    return np.maximum(x - k, 0)

def put_payoff(x: Union[float, np.ndarray], k: float) -> Union[float, np.ndarray]:
    """
    Calculate put option payoff.
    
    Args:
        x: Stock price(s)
        k: Strike price
    
    Returns:
        Put option payoff(s)
    """
    return np.maximum(k - x, 0)

def get_payoff_func(option_type: Union[str, OptionType]) -> Callable:
    """
    Get payoff function for given option type.
    
    Args:
        option_type: 'call', 'put', or OptionType enum
    
    Returns:
        Payoff function
    
    Raises:
        ValueError: If option_type is not valid
    """
    if isinstance(option_type, OptionType):
        if 'CALL' in option_type.name:
            return call_payoff
        else:
            return put_payoff
    elif isinstance(option_type, str):
        if option_type.lower() == 'call':
            return call_payoff
        elif option_type.lower() == 'put':
            return put_payoff
    raise ValueError("option_type must be 'call', 'put', or OptionType enum")

def black_scholes_call(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    """
    Calculate Black-Scholes price for European call option.
    
    Args:
        s: Current stock price
        k: Strike price
        t: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
    
    Returns:
        Option price
    """
    if t <= 0:
        return call_payoff(s, k)
    
    d1 = (np.log(s/k) + (r - q + 0.5*sigma*sigma)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    return s*np.exp(-q*t)*norm.cdf(d1) - k*np.exp(-r*t)*norm.cdf(d2)

def black_scholes_put(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    """
    Calculate Black-Scholes price for European put option.
    
    Args:
        s: Current stock price
        k: Strike price
        t: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
    
    Returns:
        Option price
    """
    if t <= 0:
        return put_payoff(s, k)
    
    d1 = (np.log(s/k) + (r - q + 0.5*sigma*sigma)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    return k*np.exp(-r*t)*norm.cdf(-d2) - s*np.exp(-q*t)*norm.cdf(-d1)

def compute_option_price(
    S: float,
    K: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType
) -> float:
    """
    Compute option price based on type and parameters.
    
    Args:
        S: Current stock price
        K: Strike price
        t: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: Type of option
        
    Returns:
        Option price
    """
    if np.isclose(t, 0) or t == 0:
        if 'CALL' in option_type.name:
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    if 'CALL' in option_type.name:
        euro_price = black_scholes_call(S, K, t, r, q, sigma)
    else:
        euro_price = black_scholes_put(S, K, t, r, q, sigma)
    
    if option_type == OptionType.AMERICAN_PUT:
        return euro_price + 0.01  # Small premium for early exercise right
    
    if option_type == OptionType.AMERICAN_CALL and q > 0:
        if check_early_exercise(S, K, t, r, q, sigma, option_type):
            return max(S - K, euro_price)
    
    return euro_price

def compute_greeks(
    S: float,
    K: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    v: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute option Greeks (delta and gamma).
    
    Args:
        S: Current stock price
        K: Strike price
        t: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: Type of option
        v: Current variance (for Heston model)
        
    Returns:
        Tuple of (delta, gamma)
    """
    if np.isclose(t, 0) or t == 0:
        if option_type in [OptionType.EUROPEAN_CALL, OptionType.AMERICAN_CALL]:
            return (1.0 if S > K else 0.0), 0.0
        else:
            return (-1.0 if S < K else 0.0), 0.0
    
    d1 = ((np.log(S / K) + (r - q + sigma ** 2 / 2) * t)) / (sigma * np.sqrt(t))
    
    # For American options with early exercise
    if option_type == OptionType.AMERICAN_CALL and q > 0 and S - K > compute_option_price(S, K, t, r, q, sigma, option_type):
        return 1.0, 0.0
    elif option_type == OptionType.AMERICAN_PUT and K - S > compute_option_price(S, K, t, r, q, sigma, option_type):
        return -1.0, 0.0
    
    # Standard Greeks
    if option_type in [OptionType.EUROPEAN_CALL, OptionType.AMERICAN_CALL]:
        delta = np.exp(-q * t) * norm.cdf(d1)
    else:
        delta = -np.exp(-q * t) * norm.cdf(-d1)
    
    gamma = np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
    
    return delta, gamma

def check_early_exercise(
    S: float,
    K: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType
) -> bool:
    """
    Check if early exercise is optimal.
    
    Args:
        S: Current stock price
        K: Strike price
        t: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: Type of option
        
    Returns:
        Whether early exercise is optimal
    """
    if option_type == OptionType.AMERICAN_CALL:
        if q <= 0:
            return False
        return q >= 0.1 and S >= K
    elif option_type == OptionType.AMERICAN_PUT:
        return r > 0 and S <= K
    return False 