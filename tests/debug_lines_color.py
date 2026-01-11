
import sys
import os
import torch
import numpy as np
import plotly.io as pio

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import location_plot_utils

def debug_lines():
    print("DEBUG: Starting line color debug...")
    plotter = location_plot_utils.LocationPlotter(title="Debug Lines")
    
    # Fake data
    coords = torch.zeros((10, 3), dtype=torch.float64)
    vel_vec = torch.ones((10, 3), dtype=torch.float64)
    values = np.linspace(0, 100, 10)
    
    # Add Velocity (Lines) with color_by_value=False
    plotter.add_velocity_2d(
        coords=coords,
        velocity=vel_vec,
        color="lime",
        color_by_value=False,
        values=values,
        vel_ref_lla=coords # Required for ENU
    )
    
    # Manually trigger trace generation for frame 0
    traces = []
    for layer in plotter._layers:
        t = plotter._generate_trace_data(layer, 0, 5)
        traces.extend(t)
        
    print(f"Generated {len(traces)} traces.")
    for i, trace in enumerate(traces):
        print(f"Trace {i} Mode: {trace.get('mode')}")
        if 'line' in trace:
            print(f"Trace {i} Line Color: {trace['line'].get('color')}")
        if 'marker' in trace:
            print(f"Trace {i} Marker Color: {trace['marker'].get('color')}")

if __name__ == "__main__":
    debug_lines()
