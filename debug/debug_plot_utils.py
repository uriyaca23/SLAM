
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from datetime import datetime, timedelta
from location_plot_utils import LocationPlotter

def debug_trace():
    print("Initializing Plotter...")
    plotter = LocationPlotter(title="Debug Reverted", use_slider=True, slide_window=10)
    
    N = 50
    lat = 32.0 + np.linspace(0, 1, N)
    lon = 34.0 + np.linspace(0, 1, N)
    traj = np.stack([lat, lon, np.zeros(N)], axis=1)
    
    start_time = datetime(2026, 1, 4, 12, 0, 0)
    timestamps = np.array([start_time + timedelta(seconds=i*30) for i in range(N)])
    
    print("Adding points...")
    plotter.add_points(traj, sys='lla', timestep_values=timestamps, label="Points")
    
    print("Inspecting _layers...")
    layer = plotter._layers[0]
    traces = plotter._generate_trace_data(layer, start_idx=0, end_idx=10, is_frame=True)
    print("Generated traces successfully.")

if __name__ == "__main__":
    debug_trace()
