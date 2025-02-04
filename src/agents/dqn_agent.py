"""
DQN agent for American option pricing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, buffer_size: int, batch_size: int):
        """Initialize buffer with given capacity and batch size."""
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add transition to buffer."""
        state = np.asarray(state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        next_state = np.asarray(next_state, dtype=np.float32)
        done = bool(done)
        self.buffer.append((state, action, reward, next_state, done))
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Alias for push to maintain compatibility with tests."""
        self.push(state, action, reward, next_state, done)
    
    def sample(self, batch_size: Optional[int] = None) -> List[Tuple]:
        """Sample random batch of transitions."""
        batch_size = batch_size or self.batch_size
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class QNetwork(nn.Module):
    """Q-network for DQN."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, depth: int):
        """Initialize network with configurable architecture."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build layers
        layers = []
        prev_dim = state_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    """DQN agent for option pricing."""
    
    def __init__(self, env: Any, config: Dict[str, Any]):
        """
        Initialize agent.
        
        Args:
            env: Environment to interact with
            config: Dictionary containing:
                hidden_dims: List of hidden layer dimensions
                learning_rate: Learning rate for optimizer
                buffer_size: Size of replay buffer
                batch_size: Batch size for training
                gamma: Discount factor
                epsilon: Initial exploration rate
                epsilon_decay: Exploration decay rate
                epsilon_min: Minimum exploration rate
                target_update_freq: Steps between target network updates
        """
        self.env = env
        self.config = config
        
        # Get dimensions
        self.state_dim = env.observation_space_shape[0]
        self.action_dim = env.action_space_n
        
        # Create networks
        self.q_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=config['hidden_dims'][0],
            depth=len(config['hidden_dims'])
        )
        self.target_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=config['hidden_dims'][0],
            depth=len(config['hidden_dims'])
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=config['learning_rate']
        )
        
        # Create replay buffer
        self.buffer = ReplayBuffer(
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size']
        )
        
        # Set hyperparameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.target_update_freq = config['target_update_freq']
        
        # Initialize counters
        self.steps = 0
        self.episodes = 0
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Take a step in the environment and store the transition."""
        # Store transition in replay buffer
        self.buffer.push(state, action, reward, next_state, done)
        
        # Update step counter
        self.steps += 1
        
        # Update networks if enough samples
        if len(self.buffer) >= self.batch_size:
            return self._update_networks()
        return 0.0
    
    def select_action(self, state: np.ndarray, eps: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
    
    def train(self, num_episodes: int, notebook: bool = True, verbose: bool = True) -> Tuple[List[float], List[float], Any]:
        """
        Train agent.
        
        Args:
            num_episodes: Number of episodes to train for
            notebook: Whether running in notebook (for plotting)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (rewards, losses, training figure)
        """
        rewards = []
        losses = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            
            while True:
                # Select and take action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition
                self.buffer.push(state, action, reward, next_state, done)
                
                # Update networks
                if len(self.buffer) >= self.batch_size:
                    loss = self._update_networks()
                    episode_loss += loss
                
                # Update counters and state
                self.steps += 1
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )
            
            # Store metrics
            rewards.append(episode_reward)
            losses.append(episode_loss)
            
            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                print(f'Episode {episode + 1}/{num_episodes} | '
                      f'Reward: {episode_reward:.2f} | '
                      f'Loss: {episode_loss:.2f} | '
                      f'Epsilon: {self.epsilon:.2f}')
        
        # Create training figure
        from ..utils.visualizers import plot_training_metrics
        fig = plot_training_metrics(rewards, losses)
        
        return rewards, losses, fig
    
    def eval(self, num_episodes: int, notebook: bool = True) -> Tuple[float, Any]:
        """
        Evaluate agent.
        
        Args:
            num_episodes: Number of episodes to evaluate for
            notebook: Whether running in notebook (for plotting)
            
        Returns:
            Tuple of (mean reward, evaluation figure)
        """
        rewards = []
        paths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            path = []
            
            while True:
                # Select and take action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_net(state_tensor)
                    action = q_values.argmax().item()
                
                next_state, reward, done, _ = self.env.step(action)
                
                # Store metrics
                episode_reward += reward
                path.append((state[1], action))  # Store price and action
                
                if done:
                    break
                
                state = next_state
            
            rewards.append(episode_reward)
            paths.append(path)
        
        # Create evaluation figure
        from ..utils.visualizers import plot_exercise_boundary
        fig = plot_exercise_boundary(paths)
        
        return np.mean(rewards), fig
    
    def _update_networks(self) -> float:
        """Update Q-networks and return loss."""
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        state_batch = torch.FloatTensor([s for s, _, _, _, _ in batch])
        action_batch = torch.LongTensor([a for _, a, _, _, _ in batch])
        reward_batch = torch.FloatTensor([r for _, _, r, _, _ in batch])
        next_state_batch = torch.FloatTensor([s for _, _, _, s, _ in batch])
        done_batch = torch.FloatTensor([d for _, _, _, _, d in batch])
        
        # Compute current Q values
        current_q_values = self.q_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

    def save(self, path: str):
        """Save the model state to a file."""
        torch.save({
            'q_network_state_dict': self.q_net.state_dict(),
            'target_network_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        """Load the model state from a file."""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def act(self, state: np.ndarray, eps: Optional[float] = None) -> int:
        """Alias for select_action to match test interface."""
        return self.select_action(state, eps) 