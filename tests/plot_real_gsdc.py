
import sys
import os
import torch
import numpy as np
import pandas as pd
import plotly.io as pio

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import location_plot_utils
import location_utils

# Data Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "google_decimeter_example")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

def load_data():
    print("Loading data...")
    gt_path = os.path.join(DATA_DIR, "ground_truth.csv")
    gnss_path = os.path.join(DATA_DIR, "device_gnss.csv")
    
    if not os.path.exists(gt_path) or not os.path.exists(gnss_path):
        print(f"Error: Data files not found in {DATA_DIR}")
        return None, None

    # Load Ground Truth
    gt_df = pd.read_csv(gt_path)
    # Filter for 'Fix' type just in case, though usually GT is all Fix
    if 'MessageType' in gt_df.columns:
        gt_df = gt_df[gt_df['MessageType'] == 'Fix']
    
    gt_lla = gt_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].values
    gt_time = gt_df['UnixTimeMillis'].values
    
    # GT Velocity/Covariance Info
    # Check if columns exist
    gt_speed = gt_df['SpeedMps'].values if 'SpeedMps' in gt_df.columns else np.zeros(len(gt_df))
    gt_bearing = gt_df['BearingDegrees'].values if 'BearingDegrees' in gt_df.columns else np.zeros(len(gt_df))
    gt_accuracy = gt_df['AccuracyMeters'].values if 'AccuracyMeters' in gt_df.columns else np.ones(len(gt_df))*3.0
    
    # Load GNSS Device Data
    # Read only relevant columns to save memory/time
    gnss_cols = ['utcTimeMillis', 'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
    gnss_df = pd.read_csv(gnss_path, usecols=gnss_cols)
    
    # Drop duplicates to get one position per timestamp (WLS pos is repeated for all sats)
    gnss_df = gnss_df.drop_duplicates(subset=['utcTimeMillis'])
    
    # Sort just in case
    gnss_df = gnss_df.sort_values('utcTimeMillis')
    
    gnss_ecef = gnss_df[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].values
    gnss_time = gnss_df['utcTimeMillis'].values
    
    # Drop NaNs from GNSS (sometimes WLS is not computed)
    valid_idx = ~np.isnan(gnss_ecef).any(axis=1)
    gnss_ecef = gnss_ecef[valid_idx]
    gnss_time = gnss_time[valid_idx]
    
    return (gt_lla, gt_time, gt_speed, gt_bearing, gt_accuracy), (gnss_ecef, gnss_time)

def plot_real_gsdc():
    (gt_lla, gt_time, gt_speed, gt_bearing, gt_accuracy), (gnss_ecef, gnss_time) = load_data()
    
    if gt_lla is None:
        return

    print("Converting Data to Tensors...")
    gt_coords = torch.tensor(gt_lla, dtype=torch.float64)
    gnss_ecef_t = torch.tensor(gnss_ecef, dtype=torch.float64)
    
    print("Converting GNSS ECEF to LLA...")
    # Using functional API for conversion
    # Note: GNSS WLS is in ECEF, assume WGS84
    gnss_lla = location_utils.convert_coordinates(
        coords=gnss_ecef_t,
        coords_sys='ecef',
        dst_sys='lla'
    )
    
    # User requested no downsampling, but better performance
    # Strategy: Keep all points for traces, but reduce animation frames using slide_step
    
    # gt_coords = gt_coords[::DOWNSAMPLE]
    # gt_datetimes = pd.to_datetime(gt_time[::DOWNSAMPLE], unit='ms').values
    gt_datetimes = pd.to_datetime(gt_time, unit='ms').values
    gnss_datetimes = pd.to_datetime(gnss_time, unit='ms').values

    print(f"Plotting {len(gt_coords)} GT points and {len(gnss_lla)} GNSS points (Full Resolution)...")
    
    # ---------------------------------------------------------
    # Derive Velocities & Covariances
    # ---------------------------------------------------------
    
    # 1. GT Velocity (From Speed/Bearing -> ENU)
    gt_rad = np.deg2rad(gt_bearing)
    gt_vel_e = gt_speed * np.sin(gt_rad)
    gt_vel_n = gt_speed * np.cos(gt_rad)
    gt_vel_u = np.zeros_like(gt_speed)
    gt_velocity = np.stack([gt_vel_e, gt_vel_n, gt_vel_u], axis=1) # (N, 3)

    # Scale by dt for visualization (Displacement vector)
    gt_dt = np.diff(gt_time, prepend=gt_time[0]) / 1000.0 # seconds
    gt_dt[0] = 1.0 # default first element
    gt_displacement = gt_velocity * gt_dt[:, None]
    gt_displacement_tensor = torch.tensor(gt_displacement, dtype=torch.float64)

    # 2. GT Covariance
    gt_cov = torch.zeros((len(gt_accuracy), 3, 3), dtype=torch.float64)
    gt_acc_sq = torch.tensor(gt_accuracy**2, dtype=torch.float64)
    gt_cov[:, 0, 0] = gt_acc_sq
    gt_cov[:, 1, 1] = gt_acc_sq
    
    # 3. GNSS Velocity (Derived from Position Delta)
    gnss_pos_diff = gnss_ecef[1:] - gnss_ecef[:-1]
    gnss_time_diff = (gnss_time[1:] - gnss_time[:-1]) / 1000.0
    gnss_time_diff[gnss_time_diff == 0] = 1.0
    # For derived velocity, the displacement IS the position difference (approx)
    # But usually we want V = dP/dt. 
    # User wants "length on map = vel * dt". 
    # If we derived V = dP/dt, then V*dt = dP.
    # So we can just use the position diff as the vector to plot!
    # Pad last element
    gnss_disp_ecef = np.vstack([gnss_pos_diff, gnss_pos_diff[-1:]])
    gnss_displacement_tensor = torch.tensor(gnss_disp_ecef, dtype=torch.float64)
    
    # ... (Plotter init same) ...
    # Satellite Map URL from test_satellite_map.py
    # SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    SATELLITE_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

    print("Initializing Plotter...")
    plotter = location_plot_utils.LocationPlotter(
        title="Google Decimeter Challenge Real Data: Trajectory, Velocity & Covariance",
        map_url=SATELLITE_URL, # Use satellite imagery
        zoom=16,
        use_slider=True,
        slide_window=200, 
        slide_step=20 
    )
    
    print("Adding Ground Truth...")
    plotter.add_points(
        coords=gt_coords,
        sys='lla',
        values=gt_datetimes,
        label="Ground Truth",
        color="lime", # Lighter green for satellite visibility
        marker_size=6,
        opacity=0.9,
        color_by_value=False
    )
    
    # Add GT Velocity
    plotter.add_velocity_2d(
        coords=gt_coords,
        coords_sys='lla',
        velocity=gt_displacement_tensor, # Passing displacement!
        vel_sys='enu',
        vel_ref_lla=gt_coords,
        values=gt_datetimes,
        label="GT Velocity (Step Result)",
        color="lime", # Match GT color
        scale=1.0, # 1:1 scale with time-step displacement
        opacity=0.8,
        color_by_value=False
    )

    # Add GT Covariance
    plotter.add_covariance_2d(
        coords=gt_coords,
        coords_sys='lla',
        covariance=gt_cov,
        cov_sys='enu',
        cov_ref_lla=gt_coords,
        values=gt_datetimes,
        label="GT Uncertainty (90%)",
        color="lime", # Match GT color
        sigma=2.146, # 90% confidence for 2D Gaussian (Chi-square 2 dof)
        opacity=0.3,
        color_by_value=False
    )
    
    print("Adding GNSS WLS Solution...")
    plotter.add_points(
        coords=gnss_lla,
        sys='lla',
        values=gnss_datetimes,
        label="GNSS WLS",
        color="red",
        marker_size=5,
        symbol="circle",
        opacity=0.7,
        color_by_value=False
    )
    
    # Add GNSS Velocity
    plotter.add_velocity_2d(
        coords=gnss_lla,
        coords_sys='lla',
        velocity=gnss_displacement_tensor, # Passing displacement
        vel_sys='ecef',
        values=gnss_datetimes,
        label="GNSS Velocity (Step Result)",
        color="red", # Match GNSS color
        scale=1.0,
        opacity=0.6,
        color_by_value=False
    )
    
    filename = os.path.join(RESULTS_DIR, "real_gsdc_plot.html")
    print(f"Saving to {filename}...")
    plotter.save(filename)
    
    if os.path.exists(filename):
        print(f"SUCCESS: Plot saved to {filename}")
    else:
        print("FAILURE: Plot not created.")

if __name__ == "__main__":
    plot_real_gsdc()
