#!/usr/bin/env python3
"""
Script to run parallel option pricing simulations.
"""

import os
import sys
import argparse
import multiprocessing as mp

from src.utils.parallel import run_parallel_simulations

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run parallel option pricing simulations.')
    parser.add_argument('--params', type=str, required=True,
                      help='Path to parameters JSON file')
    parser.add_argument('--test-cases', type=str, required=True,
                      help='Path to test cases CSV file')
    parser.add_argument('--results', type=str, required=True,
                      help='Path to save results')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method to 'spawn' on macOS
    if sys.platform == 'darwin':
        mp.set_start_method('spawn')
    
    # Run simulations
    run_parallel_simulations(
        params_path=args.params,
        test_cases_path=args.test_cases,
        results_path=args.results
    )

if __name__ == '__main__':
    main() 