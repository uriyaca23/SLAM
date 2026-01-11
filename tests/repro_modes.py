
import os
import sys
import torch
import numpy as np
import pandas as pd
# Adjust path to include the project directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from location_plot_utils import LocationPlotter

output_file = "debug/test_modes.html"
if not os.path.exists("debug"):
    os.makedirs("debug")

def test_mode_switching():
    print("Initializing Plotter...")
    plotter = LocationPlotter(title="Mode Switching Test", use_slider=True, slide_window=5)
    
    # 1. Add Points with Timestep and Categorical
    N = 20
    lats = 32.0 + np.linspace(0, 0.1, N)
    lons = 34.0 + np.linspace(0, 0.1, N)
    lla1 = np.stack([lats, lons, np.zeros(N)], axis=1)
    
    t_vals1 = np.linspace(0, 100, N) # Timestep
    cat_vals1 = np.array(['A'] * 10 + ['B'] * 10) # Categorical
    
    print("Adding Layer 1...")
    plotter.add_points(
        lla1, 
        label="Layer1", 
        color="blue", # Name color
        timestep_values=t_vals1,
        categorical_values=cat_vals1,
        colorbar_title="L1 Time"
    )
    
    # 2. Add Line (Covariance) with fallback
    # Covariance needs a lot of args, let's just make a simple line using add_points for simplicity or actually use add_covariance if I can.
    # Let's use add_velocity_2d as it creates lines.
    
    vel_data = torch.randn(N, 3)
    print("Adding Layer 2 (Velocity)...")
    plotter.add_velocity_2d(
        lla1, 
        vel_data, 
        vel_ref_lla=lla1, # ENU needs ref
        label="Layer2_Vel", 
        color="red",
        timestep_values=t_vals1 + 50, # Offset time
        categorical_values=np.array(['X']*5 + ['Y']*15)
    )

    # 3. Layer with ONLY default values (should be treated as Name or fallback?)
    # If I provide 'values' it acts as fallback for others if not provided.
    lla2 = lla1 + 0.05
    plotter.add_points(
        lla2,
        label="Layer3_Simple",
        color="green",
        timestep_values=np.arange(N) # This should map to Timestep auto-detection
    )

    print("Building and Saving Plot...")
    plotter.save(output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    test_mode_switching()
