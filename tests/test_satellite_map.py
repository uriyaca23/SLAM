
import sys
import os
import torch
import numpy as np
import plotly.io as pio

# Add parent directory to path to see location_plot_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import location_plot_utils

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

def test_satellite_map():
    print("Testing Satellite Map URL...")
    
    # URL for ESRI World Imagery (Global Satellite)
    # Note standard format: {z}/{y}/{x}
    # Some services use {x}/{y}/{z} or query params. ESRI uses standard path.
    satellite_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    
    # Create some points (e.g., Central Park, NY to see imagery clearly)
    lat_start, lon_start = 40.7829, -73.9654
    N = 50
    # Walk south-west
    lats = lat_start - np.linspace(0, 0.01, N)
    lons = lon_start - np.linspace(0, 0.01, N)
    lla = torch.tensor(np.stack([lats, lons, np.zeros(N)], axis=1))
    
    # Initialize Plotter with map_url
    plotter = location_plot_utils.LocationPlotter(
        title="Test: ESRI Satellite Map",
        map_url=satellite_url,
        zoom=13
    )
    
    plotter.add_points(
        coords=lla, 
        sys='lla', 
        label="Path", 
        color="red", 
        marker_size=10
    )
    
    filename = os.path.join(RESULTS_DIR, "test_satellite_map.html")
    print(f"Saving to {filename}...")
    plotter.save(filename)
    
    # Verify file exists
    if os.path.exists(filename):
        print("SUCCESS: HTML file created.")
    else:
        print("FAILURE: File not created.")

if __name__ == "__main__":
    test_satellite_map()
