"""
Tests for DQN agent.
"""

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from src.environments.american_option import AmericanOptionEnv, Action
from src.agents.dqn_agent import DQNAgent, QNetwork, ReplayBuffer

@pytest.fixture
def env_config():
    """Environment configuration for testing."""
    return {
        'S': 100.0,
        'K': 100.0,
        'T': 1.0,
        'r': 0.05,
        'sigma': 0.2,
        'L': 10,
        'm': 100,
        'q': 0.0,
        'use_gbm': True,
        'option_type': 'put'
    }

@pytest.fixture
def agent_config():
    """Agent configuration for testing."""
    return {
        'hidden_dims': [32, 32],
        'learning_rate': 1e-3,
        'buffer_size': 1000,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.9,  # More aggressive decay for testing
        'epsilon_min': 0.01,
        'target_update_freq': 10
    }

def test_replay_buffer():
    """Test replay buffer functionality."""
    buffer = ReplayBuffer(buffer_size=100, batch_size=5)
    
    # Test adding experiences
    state = np.array([0.5, 1.0, 0.0])
    next_state = np.array([0.4, 1.1, 0.1])
    buffer.add(state, Action.HOLD, 0.0, next_state, False)
    
    assert len(buffer) == 1
    
    # Test sampling
    for _ in range(9):  # Add more experiences
        buffer.add(state, Action.HOLD, 0.0, next_state, False)
    
    batch = buffer.sample()
    assert len(batch) == 5  # Should match batch_size

def test_q_network():
    """Test Q-network functionality."""
    network = QNetwork(state_dim=3, action_dim=2, hidden_dim=32, depth=2)
    
    # Test forward pass
    batch_size = 10
    state = torch.randn(batch_size, 3)
    output = network(state)
    
    assert output.shape == (batch_size, 2)
    assert not torch.any(torch.isnan(output))

def test_dqn_agent(env_config, agent_config):
    """Test DQN agent functionality."""
    env = AmericanOptionEnv(env_config)
    agent = DQNAgent(env=env, config=agent_config)
    
    # Test action selection
    state = env.reset()
    action = agent.act(state)
    assert action in [Action.HOLD, Action.EXERCISE]
    
    # Test update
    next_state, reward, done, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    
    # Test deterministic action selection
    action = agent.act(state, eps=0.0)  # eps=0.0 for deterministic selection
    assert action in [Action.HOLD, Action.EXERCISE]
    
    # Test save and load
    with tempfile.NamedTemporaryFile() as tmp:
        agent.save(tmp.name)
        agent.load(tmp.name)

def test_agent_training(env_config, agent_config):
    """Test agent training loop."""
    env = AmericanOptionEnv(env_config)
    agent = DQNAgent(env=env, config=agent_config)
    
    # Store initial epsilon
    initial_epsilon = agent.epsilon
    
    # Run enough episodes to ensure learning
    num_episodes = 10
    min_steps = agent_config['batch_size'] * 2  # Reduced to ensure we hit this threshold
    
    # First fill the replay buffer
    state = env.reset()
    for _ in range(agent_config['batch_size']):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        if done:
            state = env.reset()
        else:
            state = next_state
    
    # Now run episodes for learning
    rewards, losses, _ = agent.train(num_episodes=num_episodes)
    
    # Check that epsilon decayed
    assert agent.epsilon < initial_epsilon
    assert len(rewards) == num_episodes
    assert len(losses) == num_episodes 