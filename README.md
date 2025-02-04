# Deep Derivatives

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning and reinforcement learning framework for derivative pricing and hedging.

## Overview

Deep Derivatives is a Python framework that combines deep learning and reinforcement learning techniques for pricing and hedging financial derivatives. The framework currently focuses on American-style options but is designed to be extensible to other derivatives.

## Features

### Core Functionality
- American option pricing using Deep Q-Learning
- European option pricing with Monte Carlo methods
- Support for multiple underlying models:
  - Geometric Brownian Motion (GBM)
  - Heston Stochastic Volatility
- Longstaff-Schwartz method implementation
- Parallel processing support for large-scale simulations

### Deep Learning Components
- DQN agent with experience replay
- Configurable neural network architectures
- Checkpoint saving and loading
- Comprehensive logging and metrics
- Regular evaluation during training

### Market Conditions
- Varying spot prices (30-300)
- Different volatility regimes (0.01-1.0)
- Multiple maturities (0.25-10 years)
- Dividend impacts (0-30%)
- Interest rate scenarios (0-50%)
- Deep ITM/ATM/OTM scenarios

## Project Structure

```
.
├── src/
│   ├── agents/           # RL agent implementations
│   │   ├── __init__.py
│   │   └── dqn_agent.py
│   ├── environments/     # Option pricing environments
│   │   ├── __init__.py
│   │   └── american_option.py
│   └── utils/           # Utility functions
│       ├── monte_carlo.py
│       ├── option_utils.py
│       ├── parallel.py
│       ├── path_generators.py
│       └── visualizers.py
├── tests/              # Test suite
├── scripts/           # Command-line tools
├── data/             # Configuration and test data
└── docs/            # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/syphinx/deep-derivatives.git
cd deep-derivatives
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.environments.american_option import AmericanOptionEnv
from src.agents.dqn_agent import DQNAgent

# Configure environment
env_config = {
    'S': 100.0,  # Initial stock price
    'K': 100.0,  # Strike price
    'T': 1.0,    # Time to maturity
    'r': 0.04,   # Risk-free rate
    'sigma': 0.2,  # Volatility
    'L': 50,     # Number of time steps
    'm': 1000,   # Number of Monte Carlo paths
    'q': 0.02,   # Dividend yield
    'use_gbm': True,  # Use GBM model
    'option_type': 'put'  # Put option
}

# Create environment and agent
env = AmericanOptionEnv(env_config)
agent_config = {
    'hidden_dims': [128, 128],
    'learning_rate': 1e-3,
    'buffer_size': 1000,
    'batch_size': 32,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'target_update_freq': 10
}
agent = DQNAgent(env=env, config=agent_config)

# Train agent
rewards, losses, fig = agent.train(num_episodes=1000, notebook=False)
```

## References

1. Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. The review of financial studies, 14(1), 113-147.

2. Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. The review of financial studies, 6(2), 327-343.

3. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 