# API Documentation

## Core Components

### Agents

#### DQNAgent
The main reinforcement learning agent for option pricing.

```python
from src.agents.dqn_agent import DQNAgent

agent = DQNAgent(env, config={
    'hidden_dims': [128, 128],
    'learning_rate': 0.001,
    'buffer_size': 10000,
    'batch_size': 32,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'target_update_freq': 100
})
```

**Methods**:
- `step(state, action, reward, next_state, done)`: Process a single environment step
- `act(state)`: Select an action using the current policy
- `train(num_episodes)`: Train the agent for specified number of episodes
- `save(path)`: Save model weights and parameters
- `load(path)`: Load model weights and parameters

### Environments

#### AmericanOptionEnv
Environment for American option pricing simulation.

```python
from src.environments.american_option import AmericanOptionEnv

env = AmericanOptionEnv({
    'S': 100.0,  # Initial stock price
    'K': 100.0,  # Strike price
    'T': 1.0,    # Time to maturity
    'r': 0.05,   # Risk-free rate
    'sigma': 0.2 # Volatility
})
```

**Methods**:
- `reset()`: Reset environment to initial state
- `step(action)`: Take action and return (next_state, reward, done, info)

### Utils

#### Monte Carlo
```python
from src.utils.monte_carlo import price_american_option

price = price_american_option(
    s0=100,
    strike=100,
    t1=0,
    t2=1,
    risk_free_rate=0.05,
    dividend_yield=0.02,
    volatility=0.2,
    nsim=10000,
    nstep=100
)
```

#### Path Generators
```python
from src.utils.path_generators import generate_gbm_paths, generate_heston_paths

# Generate GBM paths
gbm_paths = generate_gbm_paths(
    nsim=1000,
    nstep=100,
    t1=0,
    t2=1,
    s_0=100,
    r=0.05,
    q=0.02,
    v=0.2
)

# Generate Heston paths
heston_paths = generate_heston_paths(
    nsim=1000,
    nstep=100,
    t1=0,
    t2=1,
    s_0=100,
    r=0.05,
    q=0.02,
    v_0=0.04,
    theta=0.04,
    kappa=2.0,
    sigma=0.2,
    rho=-0.7
)
```

#### Option Utils
```python
from src.utils.option_utils import (
    compute_option_price,
    compute_delta,
    compute_gamma,
    OptionType
)

# Price European option
price = compute_option_price(
    S=100,
    K=100,
    t=1,
    r=0.05,
    q=0.02,
    sigma=0.2,
    option_type=OptionType.EUROPEAN_CALL
)

# Compute Greeks
delta = compute_delta(...)
gamma = compute_gamma(...)
```

## Configuration

### Training Configuration
```python
config = {
    # Network architecture
    'hidden_dims': [128, 128],
    'learning_rate': 0.001,
    
    # Experience replay
    'buffer_size': 10000,
    'batch_size': 32,
    
    # Training parameters
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'target_update_freq': 100
}
```

### Environment Configuration
```python
env_config = {
    'S': 100.0,     # Initial stock price
    'K': 100.0,     # Strike price
    'T': 1.0,       # Time to maturity
    'r': 0.05,      # Risk-free rate
    'sigma': 0.2,   # Volatility
    'L': 50,        # Number of time steps
    'm': 1000       # Number of paths
}
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Basic option pricing
- Training DQN agents
- Analyzing results
- Comparing with traditional methods 