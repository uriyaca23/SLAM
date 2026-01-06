
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import pandas as pd
import location_plot_utils

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

def test_plotting():
    print("Generating test data...")
    N = 100
    lat_start, lon_start = 32.0853, 34.7818
    lats = lat_start + np.linspace(0, 0.1, N)
    lons = lon_start + np.linspace(0, 0.1, N)
    lla = torch.tensor(np.stack([lats, lons, np.zeros(N)], axis=1))
    
    start_time = pd.Timestamp.now()
    times = np.array([start_time + pd.Timedelta(seconds=i*30) for i in range(N)])
    
    cov_lla = []
    for i in range(N):
        cov_lla.append(np.diag([1e-9, 1e-9, 1.0]))
    cov_lla = torch.tensor(np.array(cov_lla))
    
    vel_enu = torch.zeros(N, 3)
    vel_enu[:, 0] = 5.0
    vel_enu += torch.randn(N, 3)
    
    plotter = location_plot_utils.LocationPlotter(title="Reverted Functional API Test")
    
    plotter.add_points(lla, sys='lla', values=times, label="Path", colorbar_title="Times")
    
    plotter.add_covariance_2d(
        coords=lla, coords_sys='lla',
        covariance=cov_lla, cov_sys='lla',
        values=times, label="Uncertainty",
        colorbar_title="Times"
    )
    
    plotter.add_velocity_2d(
        coords=lla, coords_sys='lla',
        velocity=vel_enu, vel_sys='enu', vel_ref_lla=lla[0:1], # Using first point as Ref for all? Or explicit?
        # In functional test, let's assume vel_ref=lla[0:1] to match previous logic
        scale=20.0, values=times, label="Vel"
    )
    
    plotter.save(os.path.join(RESULTS_DIR, "test_plot_reverted.html"))

def test_categorical():
    print("Testing Categorical...")
    N = 20
    lla = torch.randn(N, 3) * 0.001
    lla[:, 0] += 32.0
    lla[:, 1] += 34.0
    cats = ["A"]*10 + ["B"]*10
    
    plotter = location_plot_utils.LocationPlotter(title="Categorical")
    plotter.add_points(lla, sys='lla', values=cats, label="Points")
    plotter.save(os.path.join(RESULTS_DIR, "test_plot_cat_reverted.html"))

if __name__ == "__main__":
    test_plotting()
    test_categorical()
    print("ALL PLOT TESTS FINISHED.")
