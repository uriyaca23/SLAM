
import sys
import os
import pandas as pd
import numpy as np
import torch

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import location_plot_utils

def check_process_values():
    print("Checking _process_values logic...")
    
    # Simulate data loading
    # UnixTimeMillis typical values
    t_ms = np.array([1621900000000, 1621900001000, 1621900002000], dtype=np.int64)
    
    # Convert to datetime using pandas as in plot_real_gsdc.py
    dt_vals = pd.to_datetime(t_ms, unit='ms').values
    print(f"Original Pandas Values Dtype: {dt_vals.dtype}")
    print(f"First element type: {type(dt_vals[0])}")
    print(f"Is np.datetime64? {np.issubdtype(dt_vals.dtype, np.datetime64)}")
    
    # Logic from _process_values
    vals = np.array(dt_vals) # Copy
    print(f"After np.array conversion Dtype: {vals.dtype}")
    
    val_type = 'categorical'
    if np.issubdtype(vals.dtype, np.number): 
        val_type = 'numeric'
    elif np.issubdtype(vals.dtype, np.datetime64) or (vals.size > 0 and hasattr(vals[0], 'isoformat')):
        vals_cast = None
        try: vals_cast = vals.astype("datetime64[ns]")
        except: pass
        
        if vals_cast is not None:
             val_type = 'date'
             vals = vals_cast
    
    print(f"Detected val_type: {val_type}")
    print(f"Resulting vals dtype: {vals.dtype}")
    
    plotter = location_plot_utils.LocationPlotter()
    # Dummy add
    plotter.add_points(
         coords=torch.zeros((3,3)),
         sys='lla',
         timestep_values=dt_vals,
         label="Test"
    )
    
    # Check layer
    layer = plotter._layers[0]
    print(f"Layer val_type: {layer['timestep_type']}")
    print(f"Layer values dtype: {layer['timestep_values'].dtype}")
    
    # Check date_vals logic in _build_plot
    date_vals = None
    for l in plotter._layers:
         if l['timestep_type']=='date' and l['timestep_values'] is not None and date_vals is None: 
              date_vals = l['timestep_values']
              print("Found date_vals in layer!")
              
    if date_vals is not None:
         # Check slider label generation
         f_idx = 0
         try:
             label_txt = pd.to_datetime(date_vals[f_idx]).strftime('%Y-%m-%d %H:%M:%S')
             print(f"Generated Label: {label_txt}")
         except Exception as e:
             print(f"Label generation failed: {e}")
    else:
         print("date_vals is None!")

if __name__ == "__main__":
    check_process_values()
