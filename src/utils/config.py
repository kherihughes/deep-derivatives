"""
Configuration management utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration manager for option pricing project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
                'default.yaml'
            )
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        env_config = {
            'S': self.config['environment']['initial_price'],
            'K': self.config['environment']['strike_price'],
            'T': self.config['environment']['time_to_maturity'],
            'r': self.config['environment']['risk_free_rate'],
            'sigma': self.config['environment']['volatility'],
            'L': self.config['environment']['num_time_steps'],
            'm': self.config['environment']['num_paths']
        }
        return env_config
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.config['agent']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config['training']
    
    def get_monte_carlo_config(self) -> Dict[str, Any]:
        """Get Monte Carlo configuration."""
        return self.config['monte_carlo']
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """Get parallel processing configuration."""
        return self.config['parallel']
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config['visualization']
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def update_recursive(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_recursive(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_recursive(self.config, updates)
    
    def save_config(self, path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses current config path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False) 