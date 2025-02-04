"""
Visualization utilities for option pricing analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

def plot_training_metrics(rewards: List[float], losses: List[float],
                         save_dir: Optional[str] = None):
    """
    Plot training metrics.
    
    Args:
        rewards: List of episode rewards
        losses: List of training losses
        save_dir: Directory to save plots (optional)
    """
    plt.figure(figsize=(12, 4))
    
    # Plot rewards
    plt.subplot(121)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot losses
    plt.subplot(122)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(Path(save_dir) / 'training_metrics.png')
    
    plt.close('all')

def plot_price_paths(paths: np.ndarray, exercise_points: Optional[np.ndarray] = None,
                    save_dir: Optional[str] = None):
    """
    Plot simulated price paths with optional exercise points.
    
    Args:
        paths: Array of shape (n_paths, n_steps+1) containing price paths
        exercise_points: Boolean array of shape (n_paths, n_steps+1) indicating exercise points
        save_dir: Directory to save plots (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot paths
    for i in range(min(100, paths.shape[0])):  # Plot at most 100 paths
        plt.plot(paths[i], alpha=0.1, color='blue')
    
    # Plot exercise points if provided
    if exercise_points is not None:
        exercise_times, path_indices = np.where(exercise_points)
        plt.scatter(exercise_times, paths[path_indices, exercise_times],
                   color='red', alpha=0.5, label='Exercise Points')
        plt.legend()
    
    plt.title('Simulated Price Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    
    if save_dir:
        plt.savefig(Path(save_dir) / 'price_paths.png')
    plt.close()

def plot_exercise_boundary(paths: List[List[Tuple[float, int]]],
                         save_dir: Optional[str] = None):
    """
    Plot exercise boundary for American options.
    
    Args:
        paths: List of paths, where each path is a list of (price, action) tuples
        save_dir: Directory to save plots (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Collect exercise points
    exercise_prices = []
    exercise_times = []
    
    for path in paths:
        for t, (price, action) in enumerate(path):
            if action == 1:  # Exercise action
                exercise_prices.append(price)
                exercise_times.append(t)
    
    if exercise_prices:
        # Plot scatter of exercise points
        plt.scatter(exercise_times, exercise_prices, color='red', alpha=0.5, label='Exercise Points')
        
        # Plot approximate exercise boundary
        unique_times = sorted(set(exercise_times))
        boundary_prices = []
        for t in unique_times:
            t_prices = [p for p, t2 in zip(exercise_prices, exercise_times) if t2 == t]
            if t_prices:
                boundary_prices.append(np.mean(t_prices))
        
        plt.plot(unique_times, boundary_prices, 'r-', label='Exercise Boundary')
    
    plt.title('Option Exercise Boundary')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.legend()
    
    if save_dir:
        plt.savefig(Path(save_dir) / 'exercise_boundary.png')
    
    # Return the figure for testing purposes
    return plt.gcf()

def plot_value_surface(stock_prices: np.ndarray, times: np.ndarray,
                      values: np.ndarray, save_dir: Optional[str] = None):
    """
    Plot option value surface.
    
    Args:
        stock_prices: Array of stock prices
        times: Array of times to maturity
        values: 2D array of option values
        save_dir: Directory to save plots (optional)
    """
    plt.figure(figsize=(12, 8))
    
    X, Y = np.meshgrid(times, stock_prices)
    
    # Surface plot
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, values, cmap='viridis')
    
    plt.colorbar(surf)
    ax.set_title('Option Value Surface')
    ax.set_xlabel('Time to Maturity')
    ax.set_ylabel('Stock Price')
    ax.set_zlabel('Option Value')
    
    if save_dir:
        plt.savefig(Path(save_dir) / 'value_surface.png')
    plt.close()

def plot_comparison(predictions: Dict[str, np.ndarray],
                   true_values: Optional[np.ndarray] = None,
                   save_dir: Optional[str] = None):
    """
    Plot comparison of different pricing methods.
    
    Args:
        predictions: Dictionary mapping method names to arrays of predicted values
        true_values: Array of true values (optional)
        save_dir: Directory to save plots (optional)
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(next(iter(predictions.values()))))
    width = 0.8 / len(predictions)
    
    for i, (method, values) in enumerate(predictions.items()):
        plt.bar(x + i*width, values, width, label=method, alpha=0.7)
    
    if true_values is not None:
        plt.plot(x + 0.4, true_values, 'k--', label='True Values')
    
    plt.title('Pricing Method Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Option Value')
    plt.legend()
    
    if save_dir:
        plt.savefig(Path(save_dir) / 'method_comparison.png')
    plt.close() 