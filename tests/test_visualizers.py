"""
Tests for visualization utilities.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.utils.visualizers import (
    plot_training_metrics,
    plot_price_paths,
    plot_exercise_boundary,
    plot_value_surface,
    plot_comparison
)

@pytest.mark.skip(reason="Visualization tests temporarily disabled")
def test_plot_training_metrics():
    """Test training metrics plotting."""
    # Create sample data
    rewards = [1.0, 2.0, 1.5, 3.0, 2.5]
    losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Plot and verify file creation
        plot_training_metrics(rewards, losses, tmp_dir)
        assert (Path(tmp_dir) / 'training_metrics.png').exists()

@pytest.mark.skip(reason="Visualization tests temporarily disabled")
def test_plot_price_paths():
    """Test price paths plotting."""
    paths = np.random.randn(10, 10).cumsum(axis=1) + 100
    exercise_points = np.zeros_like(paths, dtype=bool)
    exercise_points[:, -1] = True
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test without exercise points
        plot_price_paths(paths, save_dir=tmp_dir)
        assert (Path(tmp_dir) / 'price_paths.png').exists()
        
        # Test with exercise points
        plot_price_paths(paths, exercise_points, tmp_dir)
        assert (Path(tmp_dir) / 'price_paths.png').exists()

@pytest.mark.skip(reason="Visualization tests temporarily disabled")
def test_plot_exercise_boundary():
    """Test exercise boundary plotting."""
    paths = np.random.randn(10, 10).cumsum(axis=1) + 100
    exercise_points = np.zeros_like(paths, dtype=bool)
    exercise_points[:, -1] = True
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_exercise_boundary(paths, exercise_points, tmp_dir)
        assert (Path(tmp_dir) / 'exercise_boundary.png').exists()

@pytest.mark.skip(reason="Visualization tests temporarily disabled")
def test_plot_value_surface():
    """Test value surface plotting."""
    stock_prices = np.linspace(80, 120, 10)
    times = np.linspace(0, 1, 10)
    values = np.random.randn(10, 10)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_value_surface(stock_prices, times, values, tmp_dir)
        assert (Path(tmp_dir) / 'value_surface.png').exists()

@pytest.mark.skip(reason="Visualization tests temporarily disabled")
def test_plot_comparison():
    """Test comparison plotting."""
    predictions = {
        'DQN': np.random.randn(10),
        'LSM': np.random.randn(10)
    }
    true_values = np.random.randn(10)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test without true values
        plot_comparison(predictions, save_dir=tmp_dir)
        assert (Path(tmp_dir) / 'method_comparison.png').exists()
        
        # Test with true values
        plot_comparison(predictions, true_values, tmp_dir)
        assert (Path(tmp_dir) / 'method_comparison.png').exists() 