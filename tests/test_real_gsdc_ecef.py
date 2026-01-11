
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
    gt_path = os.path.join(DATA_DIR, "ground_truth.csv")
    gnss_path = os.path.join(DATA_DIR, "device_gnss.csv")
    
    if not os.path.exists(gt_path) or not os.path.exists(gnss_path):
        print(f"Error: Data files not found in {DATA_DIR}")
        return None, None

    # Load Ground Truth
    gt_df = pd.read_csv(gt_path)
    if 'MessageType' in gt_df.columns:
        gt_df = gt_df[gt_df['MessageType'] == 'Fix']
    
    gt_lla = gt_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].values
    gt_time = gt_df['UnixTimeMillis'].values
    
    gt_speed = gt_df['SpeedMps'].values if 'SpeedMps' in gt_df.columns else np.zeros(len(gt_df))
    gt_bearing = gt_df['BearingDegrees'].values if 'BearingDegrees' in gt_df.columns else np.zeros(len(gt_df))
    gt_accuracy = gt_df['AccuracyMeters'].values if 'AccuracyMeters' in gt_df.columns else np.ones(len(gt_df))*3.0
    
    # Load GNSS Device Data
    gnss_cols = ['utcTimeMillis', 'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
    gnss_df = pd.read_csv(gnss_path, usecols=gnss_cols)
    gnss_df = gnss_df.drop_duplicates(subset=['utcTimeMillis'])
    gnss_df = gnss_df.sort_values('utcTimeMillis')
    
    gnss_ecef = gnss_df[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].values
    gnss_time = gnss_df['utcTimeMillis'].values
    
    valid_idx = ~np.isnan(gnss_ecef).any(axis=1)
    gnss_ecef = gnss_ecef[valid_idx]
    gnss_time = gnss_time[valid_idx]
    
    return (gt_lla, gt_time, gt_speed, gt_bearing, gt_accuracy), (gnss_ecef, gnss_time)

def test_gsdc_ecef():
    (gt_lla, gt_time, gt_speed, gt_bearing, gt_accuracy), (gnss_ecef, gnss_time) = load_data()
    if gt_lla is None: return

    print("Converting Data to Tensors...")
    gt_coords_lla = torch.tensor(gt_lla, dtype=torch.float64)
    gnss_ecef_t = torch.tensor(gnss_ecef, dtype=torch.float64)
    
    print("Converting GT LLA to ECEF for testing...")
    # Convert GT LLA -> ECEF
    gt_coords_ecef = location_utils.convert_coordinates(
        gt_coords_lla, 'lla', 'ecef', output_type='tensor'
    )

    gt_datetimes = pd.to_datetime(gt_time, unit='ms').values
    gnss_datetimes = pd.to_datetime(gnss_time, unit='ms').values

    print(f"Plotting {len(gt_coords_ecef)} GT points and {len(gnss_ecef_t)} GNSS points (ECEF Inputs)...")

    # 1. GT Velocity (ENU -> ECEF)
    gt_rad = np.deg2rad(gt_bearing)
    gt_vel_e = gt_speed * np.sin(gt_rad)
    gt_vel_n = gt_speed * np.cos(gt_rad)
    gt_vel_u = np.zeros_like(gt_speed)
    gt_velocity_enu = np.stack([gt_vel_e, gt_vel_n, gt_vel_u], axis=1) # (N, 3)
    gt_velocity_enu_t = torch.tensor(gt_velocity_enu, dtype=torch.float64)

    # Convert ENU Velocity -> ECEF Velocity Vector
    # Using convert_vector(vec_sys='enu', dst_sys='ecef')
    gt_velocity_ecef = location_utils.convert_vector(
        vec=gt_velocity_enu_t,
        vec_sys='enu',
        dst_sys='ecef',
        coords=gt_coords_lla, # Need LLA for ENU ref (or could use ECEF coords if convert_vector handles it)
        coords_sys='lla',      # Let's provide LLA for robustness of ENU definition
        src_ref_lla=gt_coords_lla,
        output_type='tensor'
    )
    
    # Calculate Displacement in ECEF
    gt_dt = np.diff(gt_time, prepend=gt_time[0]) / 1000.0
    gt_dt[0] = 1.0
    gt_displacement_ecef = gt_velocity_ecef * torch.tensor(gt_dt[:, None], dtype=torch.float64)

    # 2. GT Covariance (ENU -> ECEF)
    gt_cov_enu = torch.zeros((len(gt_accuracy), 3, 3), dtype=torch.float64)
    gt_acc_sq = torch.tensor(gt_accuracy**2, dtype=torch.float64)
    gt_cov_enu[:, 0, 0] = gt_acc_sq
    gt_cov_enu[:, 1, 1] = gt_acc_sq
    
    gt_cov_ecef = location_utils.convert_covariance(
        cov=gt_cov_enu,
        cov_sys='enu',
        dst_sys='ecef',
        coords=gt_coords_lla,
        coords_sys='lla',
        src_ref_lla=gt_coords_lla,
        output_type='tensor'
    )

    # 3. GNSS Velocity (already ECEF diff)
    gnss_pos_diff = gnss_ecef[1:] - gnss_ecef[:-1]
    gnss_disp_ecef = np.vstack([gnss_pos_diff, gnss_pos_diff[-1:]])
    gnss_displacement_ecef = torch.tensor(gnss_disp_ecef, dtype=torch.float64)

    SATELLITE_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

    plotter = location_plot_utils.LocationPlotter(
        title="GSDC Data: ECEF Input Verification",
        map_url=SATELLITE_URL,
        zoom=16,
        use_slider=True,
        slide_window=200, 
        slide_step=20 
    )

    print("Adding Ground Truth (ECEF Inputs)...")
    plotter.add_points(
        coords=gt_coords_ecef, # ECEF
        sys='ecef',
        timestep_values=gt_datetimes,
        label="Ground Truth (ECEF)",
        color="lime",
        marker_size=6,
        opacity=0.9,
        color_by_value=False
    )

    # Add GT Velocity (ECEF Vectors at ECEF Coords)
    plotter.add_velocity_2d(
        coords=gt_coords_ecef,
        coords_sys='ecef',
        velocity=gt_displacement_ecef,
        vel_sys='ecef',
        timestep_values=gt_datetimes,
        label="GT Velocity (ECEF)",
        color="lime",
        scale=1.0, 
        opacity=0.8,
        color_by_value=False
    )
    
    # Add GT Covariance (ECEF Matrix at ECEF Coords)
    plotter.add_covariance_2d(
        coords=gt_coords_ecef,
        coords_sys='ecef',
        covariance=gt_cov_ecef,
        cov_sys='ecef',
        timestep_values=gt_datetimes,
        label="GT Covariance (ECEF Input)",
        color="lime",
        sigma=2.146,
        opacity=0.3,
        color_by_value=False
    )
    
    print("Adding GNSS (ECEF Inputs)...")
    plotter.add_points(
        coords=gnss_ecef_t, # ECEF
        sys='ecef',
        timestep_values=gnss_datetimes,
        label="GNSS WLS (ECEF)",
        color="red",
        marker_size=5,
        opacity=0.7,
        color_by_value=False
    )
    
    # Add GNSS Velocity (ECEF -> ECEF)
    plotter.add_velocity_2d(
        coords=gnss_ecef_t,
        coords_sys='ecef',
        velocity=gnss_displacement_ecef,
        vel_sys='ecef',
        timestep_values=gnss_datetimes,
        label="GNSS Velocity (ECEF)",
        color="red",
        scale=1.0,
        opacity=0.6,
        color_by_value=False
    )

    filename = os.path.join(RESULTS_DIR, "real_gsdc_ecef_plot.html")
    print(f"Saving to {filename}...")
    plotter.save(filename)
    if os.path.exists(filename):
        print(f"SUCCESS: Plot saved to {filename}")
    else:
        print("FAILURE: Plot not created.")

if __name__ == "__main__":
    test_gsdc_ecef()
