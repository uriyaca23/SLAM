
import sys
import os
import torch
import numpy as np
import pandas as pd
import math

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import location_plot_utils
import location_utils

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

def generate_synthetic_gsdc_data():
    """Generates synthetic data mimicking Google Smartphone Decimeter Challenge format."""
    print("Generating synthetic GSDC data...")
    
    # 1. Create Ground Truth (Smooth Trajectory)
    # Simulate a car drive in San Jose (approx)
    lat0, lon0 = 37.3382, -121.8863
    N = 200
    t = np.linspace(0, 100, N) # seconds
    
    # Simple figure-8 or curve
    lats = lat0 + 0.0001 * t + 0.0005 * np.sin(t/10)
    lons = lon0 + 0.0001 * t + 0.0005 * np.cos(t/10)
    alts = np.zeros(N) + 20.0
    
    gt_df = pd.DataFrame({
        'collectionName': ['2021-04-29-US-SJC-1'] * N,
        'phoneName': ['Pixel4'] * N,
        'millisSinceGpsEpoch': (1300000000000 + t * 1000).astype(int),
        'latDeg': lats,
        'lngDeg': lons,
        'heightAboveWgs84EllipsoidM': alts
    })
    
    # 2. Create GNSS (Noisy Trajectory)
    # Add noise to GT
    noise_lat = np.random.normal(0, 1e-5, N) # approx 1m
    noise_lon = np.random.normal(0, 1e-5, N)
    
    gnss_df = gt_df.copy()
    gnss_df['latDeg'] += noise_lat
    gnss_df['lngDeg'] += noise_lon
    
    return gt_df, gnss_df

def test_gsdc_plotting():
    # 1. Get Data
    gt_df, gnss_df = generate_synthetic_gsdc_data()
    
    print("Creating Tensors...")
    gt_coords = torch.tensor(gt_df[['latDeg', 'lngDeg', 'heightAboveWgs84EllipsoidM']].values, dtype=torch.float64)
    gnss_coords = torch.tensor(gnss_df[['latDeg', 'lngDeg', 'heightAboveWgs84EllipsoidM']].values, dtype=torch.float64)
    timestamps = pd.to_datetime(gt_df['millisSinceGpsEpoch'], unit='ms').values
    
    print("Initializing Plotter...")
    plotter = location_plot_utils.LocationPlotter(
        title="Google Decimeter Challenge (Synthetic Sample)",
        map_style="open-street-map",
        zoom=15,
        use_slider=True,
        slide_window=20
    )
    
    print("Adding GT Trajectory...")
    # Functional API usage
    plotter.add_points(
        coords=gt_coords,
        sys='lla',
        values=timestamps,
        label="Ground Truth",
        color="green",
        marker_size=5,
        opacity=1.0
    )
    
    print("Adding GNSS Trajectory...")
    plotter.add_points(
        coords=gnss_coords,
        sys='lla',
        values=timestamps,
        label="GNSS Measurements",
        color="red",
        marker_size=5,
        symbol="circle",
        opacity=0.7
    )
    
    filename = os.path.join(RESULTS_DIR, "test_gsdc_plot.html")
    print(f"Saving to {filename}...")
    plotter.save(filename)
    
    if os.path.exists(filename):
        print("SUCCESS: HTML file created.")
    else:
        print("FAILURE: File not created.")

if __name__ == "__main__":
    test_gsdc_plotting()
