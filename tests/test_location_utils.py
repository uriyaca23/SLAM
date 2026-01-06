
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import location_utils

def get_test_data(device='cpu'):
    lla = torch.tensor([
        [32.0, 34.0, 100.0],
        [40.7128, -74.0060, 10.0],
        [-33.8688, 151.2093, 20.0],
    ], dtype=torch.float64, device=device)
    
    N = lla.shape[0]
    covs = []
    for i in range(N):
        A = torch.randn(3, 3, dtype=torch.float64, device=device)
        P = A @ A.T + torch.eye(3, dtype=torch.float64, device=device) * 1e-2
        covs.append(P)
    cov = torch.stack(covs)
    return lla, cov

def check_close(a, b, name="Val", tol=1e-4):
    if a.shape != b.shape:
        print(f"FAIL: Shape mismatch {a.shape} vs {b.shape}")
        return False
    err = torch.abs(a - b).max().item()
    if err > tol:
        print(f"FAIL: {name} error {err} > {tol}")
        return False
    return True

def run_roundtrip(src_sys, dst_sys, lla_data, cov_data, ref_lla):
    print(f"Testing Roundtrip: {src_sys.upper()} -> {dst_sys.upper()} -> {src_sys.upper()}")
    
    # 1. Prepare Src Data
    # Convert LLA -> Src
    # Coords
    curr_coords = location_utils.convert_coordinates(
        lla_data, 
        coords_sys='lla', 
        dst_sys=src_sys, 
        src_ref_lla=None, # LLA has no ref
        dst_ref_lla=ref_lla if src_sys=='enu' else None
    )
    
    # Covariance
    # Covariance is in LLA. Convert to Src.
    # Note: convert_covariance signature args
    curr_cov = location_utils.convert_covariance(
        cov=cov_data,
        cov_sys='lla',
        dst_sys=src_sys,
        coords=lla_data, # Use LLA coords for transformation logic
        coords_sys='lla',
        src_ref_lla=None,
        dst_ref_lla=ref_lla if src_sys=='enu' else None
    )
    
    # Capture Original
    orig_coords = curr_coords.clone()
    orig_cov = curr_cov.clone()
    
    # 2. Forward: Src -> Dst
    mid_coords = location_utils.convert_coordinates(
        curr_coords,
        coords_sys=src_sys,
        dst_sys=dst_sys,
        src_ref_lla=ref_lla if src_sys=='enu' else None,
        dst_ref_lla=ref_lla if dst_sys=='enu' else None
    )
    
    mid_cov = location_utils.convert_covariance(
        cov=curr_cov,
        cov_sys=src_sys,
        dst_sys=dst_sys,
        coords=curr_coords, # Must use current coords in current sys
        coords_sys=src_sys,
        src_ref_lla=ref_lla if src_sys=='enu' else None,
        dst_ref_lla=ref_lla if dst_sys=='enu' else None,
        coords_ref_lla=ref_lla if src_sys=='enu' else None # Needed for coords conversion inside
    )

    # 3. Backward: Dst -> Src
    end_coords = location_utils.convert_coordinates(
        mid_coords,
        coords_sys=dst_sys,
        dst_sys=src_sys,
        src_ref_lla=ref_lla if dst_sys=='enu' else None,
        dst_ref_lla=ref_lla if src_sys=='enu' else None
    )
    
    end_cov = location_utils.convert_covariance(
        cov=mid_cov,
        cov_sys=dst_sys,
        dst_sys=src_sys,
        coords=mid_coords,
        coords_sys=dst_sys,
        src_ref_lla=ref_lla if dst_sys=='enu' else None,
        dst_ref_lla=ref_lla if src_sys=='enu' else None,
        coords_ref_lla=ref_lla if dst_sys=='enu' else None
    )
    
    if not check_close(end_coords, orig_coords, f"Coords {src_sys}-{dst_sys}"): return False
    if not check_close(end_cov, orig_cov, f"Cov {src_sys}-{dst_sys}"): return False
    
    print("  PASS")
    return True

def test_mixed_frames():
    print("\n--- Testing Mixed Frames ---")
    lla_ref = torch.tensor([[45.0, 0.0, 0.0]], dtype=torch.float64)
    # Define Cov in ENU directly
    cov_enu = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    
    # Convert ENU -> ECEF
    # Coords are at LLA Ref (so 0,0,0 in ENU)
    coords_lla = lla_ref
    
    cov_ecef = location_utils.convert_covariance(
        cov_enu,
        cov_sys='enu',
        dst_sys='ecef',
        coords=coords_lla, # Logic will convert these to LLA (they are already LLA)
        coords_sys='lla',
        src_ref_lla=lla_ref, # Ref for ENU Cov
        dst_ref_lla=None
    )
    
    # Back to ENU
    cov_enu_back = location_utils.convert_covariance(
        cov_ecef,
        cov_sys='ecef',
        dst_sys='enu',
        coords=coords_lla,
        coords_sys='lla',
        src_ref_lla=None,
        dst_ref_lla=lla_ref
    )
    
    if check_close(cov_enu_back, cov_enu, "MixedFrame"):
        print("  PASS Mixed Frame")
        return True
    return False

def test_velocity():
    print("\n--- Testing Velocity ---")
    ref = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    vel_enu = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    
    vel_ecef = location_utils.convert_vector(
        vel_enu, 'enu', 'ecef',
        coords=ref, coords_sys='lla',
        src_ref_lla=ref, dst_ref_lla=None
    )
    
    expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    if check_close(vel_ecef, expected, "Vel ENU->ECEF"):
        print("  PASS Vel")
        return True
    return False

def test_all():
    lla, cov = get_test_data()
    ref_lla = lla[0].unsqueeze(0)
    systems = ['lla', 'ecef', 'enu']
    for src in systems:
        for dst in systems:
            if src == dst: continue
            if not run_roundtrip(src, dst, lla, cov, ref_lla):
                sys.exit(1)
    if not test_mixed_frames(): sys.exit(1)
    if not test_velocity(): sys.exit(1)
    print("\nALL PASSED.")

if __name__ == "__main__":
    test_all()
