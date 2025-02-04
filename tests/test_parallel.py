"""Tests for parallel processing functionality."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path

from src.utils.parallel import (
    read_test_cases,
    run_single_simulation,
    run_parallel_simulations
)
from src.utils.option_utils import OptionType

@pytest.fixture
def sample_test_cases_file():
    """Create a sample test cases CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("""s_0,t2,q,r,option_type,use_gbm,sigma,v_0,theta,rho,kappa
100.0,1.0,0.02,0.05,AMERICAN_CALL,True,0.2,0.0,0.0,0.0,0.0
90.0,1.0,0.02,0.05,AMERICAN_PUT,True,0.2,0.0,0.0,0.0,0.0
110.0,1.0,0.02,0.05,AMERICAN_CALL,False,0.2,0.04,0.04,-0.7,2.0
80.0,1.0,0.02,0.05,AMERICAN_PUT,False,0.2,0.04,0.04,-0.7,2.0""")
        f.flush()  # Ensure all data is written
        path = Path(f.name)
    return path

@pytest.fixture
def sample_params_file():
    """Create a sample parameters JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        params = {
            "nsim": 100,
            "num_episodes_train": 10,
            "num_episodes_eval": 5,
            "nstep": 10,
            "t1": 0,
            "strike": 100
        }
        json.dump(params, f)
        f.flush()  # Ensure all data is written
        path = Path(f.name)
    return path

def test_read_test_cases(sample_test_cases_file):
    """Test reading test cases from CSV."""
    test_cases = read_test_cases(str(sample_test_cases_file))
    
    # Check basic properties
    assert len(test_cases) == 4
    assert isinstance(test_cases, np.ndarray)
    
    # Check first row
    assert test_cases[0][0] == 100.0  # s_0
    assert test_cases[0][1] == 1.0    # t2
    assert test_cases[0][2] == 0.02   # q
    assert test_cases[0][3] == 0.05   # r
    assert test_cases[0][4] == OptionType.AMERICAN_CALL  # option_type
    assert isinstance(test_cases[0][5], dict)  # path_kwargs
    assert test_cases[0][6] == True   # use_gbm
    
    # Check GBM vs Heston parameters
    gbm_case = test_cases[0]
    heston_case = test_cases[2]
    
    assert isinstance(gbm_case[5], dict)  # path_kwargs
    assert 'sigma' in gbm_case[5]
    assert gbm_case[5]['sigma'] == 0.2
    
    assert isinstance(heston_case[5], dict)
    assert all(k in heston_case[5] for k in ['v_0', 'theta', 'rho', 'kappa', 'sigma'])
    assert heston_case[5]['v_0'] == 0.04
    assert heston_case[5]['theta'] == 0.04
    assert heston_case[5]['rho'] == -0.7
    assert heston_case[5]['kappa'] == 2.0
    assert heston_case[5]['sigma'] == 0.2
    
    # Clean up
    sample_test_cases_file.unlink()

@pytest.mark.skip(reason="Parallel processing tests temporarily disabled")
def test_single_simulation_gbm(sample_test_cases_file, sample_params_file):
    """Test running a single GBM simulation."""
    test_cases = read_test_cases(str(sample_test_cases_file))
    args = tuple(test_cases[0])  # Use first test case (GBM)
    
    mean_reward, train_fig, eval_fig = run_single_simulation(
        args=args,
        nsim=1000,
        nstep=50,
        t1=0,
        strike=100,
        num_episodes_train=100,
        num_episodes_eval=50
    )
    
    # Check outputs
    assert isinstance(mean_reward, float)
    assert mean_reward > 0
    assert train_fig is not None
    assert eval_fig is not None
    
    # Clean up
    sample_test_cases_file.unlink()

@pytest.mark.skip(reason="Parallel processing tests temporarily disabled")
def test_single_simulation_heston(sample_test_cases_file, sample_params_file):
    """Test running a single Heston simulation."""
    test_cases = read_test_cases(str(sample_test_cases_file))
    args = tuple(test_cases[2])  # Use third test case (Heston)
    
    mean_reward, train_fig, eval_fig = run_single_simulation(
        args=args,
        nsim=1000,
        nstep=50,
        t1=0,
        strike=100,
        num_episodes_train=100,
        num_episodes_eval=50
    )
    
    # Check outputs
    assert isinstance(mean_reward, float)
    assert mean_reward > 0
    assert train_fig is not None
    assert eval_fig is not None
    
    # Clean up
    sample_test_cases_file.unlink()

@pytest.mark.skip(reason="Parallel processing tests temporarily disabled")
def test_run_parallel_simulations(sample_test_cases_file, sample_params_file):
    """Test running parallel simulations."""
    # Create temporary results file
    results_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    
    # Run simulations
    run_parallel_simulations(
        params_path=str(sample_params_file),
        test_cases_path=str(sample_test_cases_file),
        results_path=results_file.name
    )
    
    # Clean up
    sample_test_cases_file.unlink()
    sample_params_file.unlink()
    Path(results_file.name).unlink()

@pytest.mark.skip(reason="Parallel processing tests temporarily disabled")
def test_invalid_params_file(sample_test_cases_file):
    """Test handling of invalid parameters file."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        f.write(b"invalid json")
        invalid_params_file = Path(f.name)
    
    with pytest.raises(json.JSONDecodeError):
        run_parallel_simulations(
            params_path=str(invalid_params_file),
            test_cases_path=str(sample_test_cases_file),
            results_path="results.pkl"
        )
    
    # Clean up
    sample_test_cases_file.unlink()
    invalid_params_file.unlink()

@pytest.mark.skip(reason="Parallel processing tests temporarily disabled")
def test_invalid_test_cases_file(sample_params_file):
    """Test handling of invalid test cases file."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        f.write(b"col1,col2\n1,2")  # Invalid CSV without required columns
        f.flush()
        invalid_cases_file = Path(f.name)
    
    with pytest.raises(ValueError) as exc_info:
        run_parallel_simulations(
            params_path=str(sample_params_file),
            test_cases_path=str(invalid_cases_file),
            results_path="results.pkl"
        )
    assert "Missing required columns" in str(exc_info.value)
    
    # Clean up
    sample_params_file.unlink()
    invalid_cases_file.unlink() 