"""
Parallel processing utilities for option pricing.
"""

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
from typing import Dict, List, Tuple, Any

from ..environments.american_option import AmericanOptionEnv
from ..agents.dqn_agent import DQNAgent
from .option_utils import get_payoff_func, OptionType

def read_test_cases(data_path: str) -> np.ndarray:
    """Read test cases from CSV file."""
    try:
        # Read CSV with specific dtypes to prevent automatic conversion
        dtype_dict = {
            's_0': float,
            't2': float,
            'q': float,
            'r': float,
            'option_type': str,
            'use_gbm': bool,
            'sigma': float,
            'v_0': float,
            'theta': float,
            'rho': float,
            'kappa': float
        }
        df = pd.read_csv(data_path, dtype=dtype_dict)
        
        # Verify required columns
        required_cols = ['s_0', 't2', 'q', 'r', 'option_type', 'use_gbm', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert option type strings to OptionType enum
        def convert_option_type(x):
            if isinstance(x, str):
                return OptionType[x.upper()]  # Convert to uppercase to match enum names
            elif isinstance(x, bool):
                return OptionType.AMERICAN_CALL if x else OptionType.EUROPEAN_CALL
            else:
                raise ValueError(f"Invalid option type: {x}")
        
        df['option_type'] = df['option_type'].apply(convert_option_type)
        
        # Create path kwargs
        def create_path_kwargs(row):
            if row['use_gbm']:
                return {'sigma': float(row['sigma'])}
            return {
                'v_0': float(row.get('v_0', 0.04)),
                'theta': float(row.get('theta', 0.04)),
                'rho': float(row.get('rho', -0.7)),
                'kappa': float(row.get('kappa', 2.0)),
                'sigma': float(row['sigma'])
            }
        
        df['path_kwargs'] = df.apply(create_path_kwargs, axis=1)
        
        # Select final columns
        result_cols = ['s_0', 't2', 'q', 'r', 'option_type', 'path_kwargs', 'use_gbm']
        return df[result_cols].values
        
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Empty CSV file")
    except Exception as e:
        raise ValueError(f"Error reading test cases: {str(e)}")

def run_single_simulation(
    args: Tuple,
    nsim: int,
    nstep: int,
    t1: float,
    strike: float,
    num_episodes_train: int,
    num_episodes_eval: int
) -> Tuple[float, Any, Any]:
    """Run a single simulation with given parameters."""
    try:
        s_0, t2, q, r, option_type, path_kwargs, use_gbm = args
        
        # Create environment
        env_config = {
            'S': float(s_0),
            'K': float(strike),
            'T': float(t2 - t1),
            'r': float(r),
            'sigma': float(path_kwargs.get('sigma', 0.2)),
            'L': int(nstep),
            'm': int(nsim),
            'q': float(q),
            'use_gbm': bool(use_gbm),
            'option_type': option_type
        }
        
        if not use_gbm:
            env_config.update({
                'v_0': float(path_kwargs['v_0']),
                'theta': float(path_kwargs['theta']),
                'rho': float(path_kwargs['rho']),
                'kappa': float(path_kwargs['kappa'])
            })
        
        env = AmericanOptionEnv(env_config)
        
        # Create agent
        agent_config = {
            'hidden_dims': [64, 64],
            'learning_rate': 1e-3,
            'buffer_size': 1024,
            'batch_size': 64,
            'gamma': 1.0,
            'epsilon': 0.99,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'target_update_freq': 16
        }
        
        agent = DQNAgent(env=env, config=agent_config)
        
        # Train agent
        _, _, train_fig = agent.train(
            num_episodes=num_episodes_train,
            notebook=True,
            verbose=False
        )
        
        # Evaluate agent
        mean_reward, eval_fig = agent.eval(
            num_episodes=num_episodes_eval,
            notebook=True
        )
        
        return mean_reward, train_fig, eval_fig
        
    except Exception as e:
        raise RuntimeError(f"Error in simulation: {str(e)}")

def run_parallel_simulations(
    params_path: str,
    test_cases_path: str,
    results_path: str
) -> None:
    """Run simulations in parallel using all available CPU cores."""
    try:
        # Load parameters
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Set default values if not provided
        nsim = int(params.get('nsim', 10000))
        num_episodes_train = int(params.get('num_episodes_train', 3000))
        num_episodes_eval = int(params.get('num_episodes_eval', 500))
        nstep = int(params.get('nstep', 365))
        t1 = float(params.get('t1', 0))
        strike = float(params.get('strike', 100))
        
        # Read test cases
        test_cases = read_test_cases(test_cases_path)
        print(f'Running {test_cases.shape[0]} simulations...')
        
        # Set up multiprocessing
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
        print(f'Using {num_workers} workers')
        
        start_time = time.time()
        
        # Run simulations in parallel
        with mp.Pool(processes=num_workers) as pool:
            # Create list of arguments for each simulation
            sim_args = [
                (
                    test_case,
                    nsim,
                    nstep,
                    t1,
                    strike,
                    num_episodes_train,
                    num_episodes_eval
                )
                for test_case in test_cases
            ]
            results = pool.starmap(run_single_simulation, sim_args)
        
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {elapsed_time/60:.2f} minutes')
        
        # Save results
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f'Results saved to {results_path}')
        
    except json.JSONDecodeError:
        raise json.JSONDecodeError("Invalid JSON file", params_path, 0)
    except Exception as e:
        raise RuntimeError(f"Error in parallel simulations: {str(e)}") 