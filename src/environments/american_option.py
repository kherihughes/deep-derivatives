"""
Environment for American option pricing using reinforcement learning.
"""

import numpy as np
from enum import Enum
from typing import Dict, Tuple, Any

from ..utils.path_generators import generate_gbm_paths, generate_heston_paths
from ..utils.option_utils import OptionType, get_payoff_func

class Action:
    """Action space for American option pricing."""
    HOLD = 0
    EXERCISE = 1
    NUM_ACTIONS = 2

class AmericanOptionEnv:
    """
    Environment for pricing American options using reinforcement learning.
    
    The state space consists of:
    1. Normalized time to maturity
    2. Log-normalized stock price
    3. Continuation value estimate
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize environment.
        
        Args:
            config: Dictionary containing:
                S: Initial stock price
                K: Strike price
                T: Time to maturity
                r: Risk-free rate
                sigma: Volatility
                L: Number of time steps
                m: Number of paths
                n: Path index (optional)
                D: Trading days per year (optional)
                mu: Drift (optional)
                q: Dividend yield (optional)
                ss: Steps between trades (optional)
                kappa: Mean reversion speed (Heston, optional)
                use_gbm: Whether to use GBM (True) or Heston (False)
                option_type: Type of option ('call' or 'put')
                v_0: Initial variance (Heston, optional)
                theta: Long-run variance (Heston, optional)
                rho: Correlation (Heston, optional)
        """
        self.config = config
        self.rng = np.random.default_rng()
        
        # Set up spaces
        self.observation_space_shape = (3,)  # [time, price, value]
        self.action_space_n = Action.NUM_ACTIONS
        
        # Extract parameters
        self.S = config['S']  # Initial stock price
        self.K = config['K']  # Strike price
        self.T = config['T']  # Time to maturity
        self.r = config['r']  # Risk-free rate
        self.sigma = config['sigma']  # Volatility
        self.L = config['L']  # Number of time steps
        self.m = config['m']  # Number of paths
        self.q = config.get('q', 0.0)  # Dividend yield
        
        # Set payoff function
        self.option_type = config.get('option_type', 'put')
        self.payoff_func = get_payoff_func(self.option_type)
        
        # Time parameters
        self.dt = self.T / self.L
        
        # Generate initial paths
        self._generate_paths()
        
        # Reset environment
        self.reset()
    
    def _generate_paths(self):
        """Generate price paths."""
        if self.config.get('use_gbm', True):
            self.paths = generate_gbm_paths(
                nsim=self.m,
                nstep=self.L,
                t1=0,
                t2=self.T,
                s_0=self.S,
                r=self.r,
                q=self.q,
                v=self.sigma
            )
        else:
            self.paths, _ = generate_heston_paths(
                nsim=self.m,
                nstep=self.L,
                t1=0,
                t2=self.T,
                s_0=self.S,
                r=self.r,
                q=self.q,
                v_0=self.config['v_0'],
                theta=self.config['theta'],
                rho=self.config['rho'],
                kappa=self.config['kappa'],
                sigma=self.sigma
            )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.t = 0
        self.done = False
        self.reward = 0.0
        
        # Get current state
        state = self._get_state()
        return state
    
    def _get_state(self) -> np.ndarray:
        """Get current state."""
        # Get current price and normalize
        price = self.paths[:, self.t]  # Shape: (m,)
        price = np.log(price / self.K)  # Log-normalize
        
        # Normalize time
        time = self.t / self.L
        
        # Estimate continuation value (placeholder)
        value = np.zeros_like(price)
        
        # Stack state components
        state = np.array([
            time,  # Single scalar
            np.mean(price),  # Mean log price
            np.mean(value)  # Mean continuation value
        ])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=hold, 1=exercise)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
        
        # Get current price
        price = self.paths[:, self.t]  # Shape: (m,)
        
        # Calculate reward
        if action == Action.EXERCISE:
            # Exercise option
            payoff = self.payoff_func(price, self.K)  # Shape: (m,)
            # Take mean payoff across all paths
            self.reward = float(np.mean(payoff) * np.exp(-self.r * self.t * self.dt))
            self.done = True
        else:
            # Hold option
            self.reward = 0.0
            self.t += 1
            if self.t >= self.L:
                # At expiry
                payoff = self.payoff_func(price, self.K)  # Shape: (m,)
                # Take mean payoff across all paths
                self.reward = float(np.mean(payoff) * np.exp(-self.r * self.T))
                self.done = True
        
        # Get next state
        next_state = self._get_state()
        
        return next_state, self.reward, self.done, {} 