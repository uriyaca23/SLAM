
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import plotly.io as pio
from datetime import datetime, timedelta
import location_plot_utils

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def generate_trajectory(n_points=100):
    lats = np.linspace(32.0, 33.0, n_points)
    lons = np.linspace(34.0, 35.0, n_points)
    alts = np.random.uniform(0, 100, n_points)
    return np.stack([lats, lons, alts], axis=1)

def test_incremental():
    print("Test Incremental...")
    plotter = location_plot_utils.LocationPlotter(title="Incremental Reverted")
    
    traj = generate_trajectory(10)
    plotter.add_points(traj, sys='lla', label="Batch 1")
    
    cov = np.eye(3).reshape(1,3,3).repeat(10, axis=0) * 1e-9
    plotter.add_covariance_2d(traj, coords_sys='lla', covariance=cov, cov_sys='lla', label="Cov")
    
    plotter.save(os.path.join(RESULTS_DIR, "test_stress_reverted.html"))

if __name__ == "__main__":
    test_incremental()
