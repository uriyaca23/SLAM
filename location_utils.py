
import torch
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Any, Optional

# WGS84 Constants
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 1.0 - (WGS84_B**2 / WGS84_A**2)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

def _get_device(input_device: Optional[torch.device] = None) -> torch.device:
    if input_device is not None: return input_device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor(data: Any, dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # Retain existing implementation
    metadata = {"type": type(data), "original_shape": None}
    if isinstance(data, torch.Tensor):
        metadata["original_shape"] = data.shape
        if data.ndim == 1 and data.shape[0] == 3: return data.unsqueeze(0).to(dtype=dtype), metadata
        elif data.ndim == 2 and data.shape[1] == 3: return data.to(dtype=dtype), metadata
        elif data.ndim == 3 and data.shape[1:] == (3, 3): return data.to(dtype=dtype), metadata
        return data.to(dtype=dtype), metadata
    elif isinstance(data, pd.DataFrame):
        vals = data.iloc[:, :3].values
        tensor = torch.from_numpy(vals).to(dtype=dtype)
        return tensor.to(_get_device()), metadata
    elif isinstance(data, np.ndarray):
        metadata["original_shape"] = data.shape
        tensor = torch.from_numpy(data).to(dtype=dtype)
        if tensor.ndim == 1: tensor = tensor.unsqueeze(0)
        return tensor.to(_get_device()), metadata
    elif isinstance(data, (list, tuple)):
        tensor = torch.tensor(data, dtype=dtype)
        metadata["original_shape"] = tensor.shape
        if tensor.ndim == 1: tensor = tensor.unsqueeze(0)
        return tensor.to(_get_device()), metadata
    raise TypeError(f"Unsupported data type: {type(data)}")

def _from_tensor(tensor: torch.Tensor, metadata: Dict[str, Any], output_type: str = "auto") -> Any:
    # Retain existing implementation
    original_shape = metadata.get("original_shape")
    is_singular = original_shape is not None and len(original_shape) == 1
    if output_type == "auto": target = metadata["type"]
    elif output_type == "tensor": target = torch.Tensor
    elif output_type == "numpy": target = np.ndarray
    elif output_type == "dataframe": target = pd.DataFrame
    else: target = list

    if target == torch.Tensor: return tensor.squeeze(0) if is_singular else tensor
    arr = tensor.detach().cpu().numpy()
    if target == np.ndarray: return arr.squeeze(0) if is_singular else arr
    if target == pd.DataFrame: return pd.DataFrame(arr, columns=[0,1,2])
    return arr.squeeze(0).tolist() if is_singular else arr.tolist()

# --- Internal Core Math ---
# (Keeping the helper functions I wrote/saw)

def _get_jac_lla_to_ecef(lla: torch.Tensor) -> torch.Tensor:
    lat_rad = torch.deg2rad(lla[:, 0])
    lon_rad = torch.deg2rad(lla[:, 1])
    h = lla[:, 2]
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    e2 = WGS84_E2
    N = WGS84_A / torch.sqrt(1.0 - e2 * sin_lat**2)
    dN_dlat = N * e2 * sin_lat * cos_lat / (1.0 - e2 * sin_lat**2)
    dx_dlat = dN_dlat * cos_lat * cos_lon - (N + h) * sin_lat * cos_lon
    dx_dlon = -(N + h) * cos_lat * sin_lon
    dx_dh = cos_lat * cos_lon
    dy_dlat = dN_dlat * cos_lat * sin_lon - (N + h) * sin_lat * sin_lon
    dy_dlon = (N + h) * cos_lat * cos_lon
    dy_dh = cos_lat * sin_lon
    dz_dlat = dN_dlat * (1 - e2) * sin_lat + (N * (1 - e2) + h) * cos_lat
    dz_dlon = torch.zeros_like(h)
    dz_dh = sin_lat
    J = torch.stack([
        torch.stack([dx_dlat, dx_dlon, dx_dh], dim=1),
        torch.stack([dy_dlat, dy_dlon, dy_dh], dim=1),
        torch.stack([dz_dlat, dz_dlon, dz_dh], dim=1)
    ], dim=1)
    scale_mat = torch.diag(torch.tensor([np.pi/180.0, np.pi/180.0, 1.0], device=lla.device, dtype=lla.dtype))
    return J @ scale_mat

def _get_rot_ecef_to_enu(ref_lla: torch.Tensor) -> torch.Tensor:
    lat_rad = torch.deg2rad(ref_lla[:, 0])
    lon_rad = torch.deg2rad(ref_lla[:, 1])
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    r00, r01, r02 = -sin_lon, cos_lon, torch.zeros_like(sin_lon)
    r10, r11, r12 = -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat
    r20, r21, r22 = cos_lat * cos_lon, cos_lat * sin_lon, sin_lat
    return torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1)
    ], dim=1)

def _lla_to_ecef_torch(lla: torch.Tensor) -> torch.Tensor:
    lat_rad = torch.deg2rad(lla[:, 0])
    lon_rad = torch.deg2rad(lla[:, 1])
    alt = lla[:, 2]
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    N = WGS84_A / torch.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    return torch.stack([x, y, z], dim=1)

def _ecef_to_lla_torch(ecef: torch.Tensor) -> torch.Tensor:
    x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]
    r2 = x**2 + y**2
    r = torch.sqrt(r2)
    E2 = WGS84_A**2 - WGS84_B**2
    F = 54 * WGS84_B**2 * z**2
    G = r2 + (1 - WGS84_E2) * z**2 - WGS84_E2 * E2
    C = (WGS84_E2**2 * F * r2) / (G**3)
    S = (1 + C + torch.sqrt(C**2 + 2*C))**(1/3)
    P = F / (3 * (S + 1/S + 1)**2 * G**2)
    Q = torch.sqrt(1 + 2 * WGS84_E2**2 * P)
    r_0 = -(P * WGS84_E2 * r) / (1 + Q) + torch.sqrt(
        0.5 * WGS84_A**2 * (1 + 1/Q) - 
        (P * (1 - WGS84_E2) * z**2) / (Q * (1 + Q)) - 
        0.5 * P * r2
    )
    U = torch.sqrt((r - WGS84_E2 * r_0)**2 + z**2)
    V = torch.sqrt((r - WGS84_E2 * r_0)**2 + (1 - WGS84_E2) * z**2)
    z_0 = (WGS84_B**2 * z) / (WGS84_A * V)
    alt = U * (1 - WGS84_B**2 / (WGS84_A * V))
    lat_rad = torch.atan((z + WGS84_EP2 * z_0) / r)
    lon_rad = torch.atan2(y, x)
    return torch.stack([torch.rad2deg(lat_rad), torch.rad2deg(lon_rad), alt], dim=1)

def _ecef_to_enu_torch(ecef: torch.Tensor, ref_lla: torch.Tensor) -> torch.Tensor:
    ref_ecef = _lla_to_ecef_torch(ref_lla)
    diff = ecef - ref_ecef
    R = _get_rot_ecef_to_enu(ref_lla)
    res = R @ diff.unsqueeze(-1)
    return res.squeeze(-1)

def _enu_to_ecef_torch(enu: torch.Tensor, ref_lla: torch.Tensor) -> torch.Tensor:
    ref_ecef = _lla_to_ecef_torch(ref_lla)
    R = _get_rot_ecef_to_enu(ref_lla)
    diff = R.transpose(1, 2) @ enu.unsqueeze(-1)
    return ref_ecef + diff.squeeze(-1)

# --- Public API (Reverting to Functional) ---

def convert_coordinates(
    coords: Any,
    coords_sys: str,
    dst_sys: str,
    src_ref_lla: Optional[Any] = None,
    dst_ref_lla: Optional[Any] = None,
    output_type: str = "auto"
) -> Any:
    # 1. To Tensor
    t_coords, meta = _to_tensor(coords)
    device = t_coords.device
    
    c_sys = coords_sys.lower().strip()
    d_sys = dst_sys.lower().strip()
    
    if c_sys == d_sys:
        # Same system - but strictly speaking, if it's ENU -> ENU with Diff Ref, we must convert
        if c_sys == 'enu':
             if src_ref_lla is None: raise ValueError("src_ref_lla required for ENU")
             # If refs are same, return
             pass
        else:
             return _from_tensor(t_coords, meta, output_type=output_type)

    # Convert Ref LLA
    src_ref_t = None
    if src_ref_lla is not None:
        src_ref_t, _ = _to_tensor(src_ref_lla)
        src_ref_t = src_ref_t.to(device)

    dst_ref_t = None
    if dst_ref_lla is not None:
        dst_ref_t, _ = _to_tensor(dst_ref_lla)
        dst_ref_t = dst_ref_t.to(device)

    # 1. To ECEF
    ecef = None
    if c_sys == 'lla':
        ecef = _lla_to_ecef_torch(t_coords)
    elif c_sys == 'ecef':
        ecef = t_coords
    elif c_sys == 'enu':
        if src_ref_t is None: raise ValueError("src_ref_lla required")
        ref = src_ref_t
        if ref.shape[0] == 1 and t_coords.shape[0] > 1: ref = ref.expand(t_coords.shape[0], -1)
        ecef = _enu_to_ecef_torch(t_coords, ref)
    else:
        raise ValueError(f"Unknown system {c_sys}")

    # 2. To Dest
    res = None
    if d_sys == 'lla':
        res = _ecef_to_lla_torch(ecef)
    elif d_sys == 'ecef':
        res = ecef
    elif d_sys == 'enu':
        if dst_ref_t is None: raise ValueError("dst_ref_lla required")
        ref = dst_ref_t
        if ref.shape[0] == 1 and ecef.shape[0] > 1: ref = ref.expand(ecef.shape[0], -1)
        res = _ecef_to_enu_torch(ecef, ref)
    else:
        raise ValueError(f"Unknown system {d_sys}")
        
    return _from_tensor(res, meta, output_type=output_type)

def convert_covariance(
    cov: Any,
    cov_sys: str,
    dst_sys: str,
    coords: Any,
    coords_sys: str = 'lla',
    src_ref_lla: Optional[Any] = None,
    dst_ref_lla: Optional[Any] = None,
    coords_ref_lla: Optional[Any] = None,
    output_type: str = "auto"
) -> Any:
    # Requires location of covariance to calculate rotation
    # 'coords' should match 'cov' batch size
    t_cov, meta = _to_tensor(cov)
    t_coords, _ = _to_tensor(coords)
    device = t_cov.device
    t_coords = t_coords.to(device)
    
    # 1. Get LLA Coords of the points (for jacobian calculation)
    # We use convert_coordinates to get LLA
    
    # Handling "coords" conversion 
    # If coords are ENU, we need coords_ref_lla?
    # Or src_ref_lla? 
    # Usually `coords_ref_lla` is specific for the `coords` argument if `coords_sys` is ENU.
    # While `src_ref_lla` describes the *covariance's* system if covariance is ENU (which shares origin with points usually, but maybe not?)
    # "Differentiate ENU Reference Points" in Step 163 Summary implies separate params.
    
    coords_lla = t_coords
    if coords_sys != 'lla':
        # Temporarily convert coords to LLA
        # Note: If coords_sys is ENU, we need a ref.
        # User prompt in Step 163 said: "src_ref_lla (source ENU), dst_ref_lla (dest ENU), coords_ref_lla (coords argument itself)"
        
        # Logic:
        # If coords_sys is 'enu', use coords_ref_lla
        ref_for_coords = coords_ref_lla
        
        # Wait, if cov_sys='enu', and coords_sys='enu', do they use the SAME ref? 
        # Usually yes. But the user wanted clean separation.
        # Use coords_ref_lla for coords conversion.
        
        # We can reuse our new convert_coordinates function!
        coords_lla = _to_tensor(convert_coordinates(
            t_coords, coords_sys, 'lla', 
            src_ref_lla=coords_ref_lla, dst_ref_lla=None
        ))[0].to(device)
    
    # 2. Build Rotation
    # M_src->ecef
    M1 = None
    N = t_cov.shape[0]
    I = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0).expand(N, -1, -1)
    
    c_sys = cov_sys.lower().strip()
    d_sys = dst_sys.lower().strip()
    
    if c_sys == 'lla':
        M1 = _get_jac_lla_to_ecef(coords_lla)
    elif c_sys == 'enu':
        if src_ref_lla is None: raise ValueError("src_ref_lla needed for ENU covariance")
        r, _ = _to_tensor(src_ref_lla)
        r = r.to(device)
        if r.shape[0] == 1 and N > 1: r = r.expand(N, -1)
        M1 = _get_rot_ecef_to_enu(r).transpose(1, 2)
    else: # ecef
        M1 = I
        
    # M_ecef->dst
    M2 = None
    if d_sys == 'lla':
        J = _get_jac_lla_to_ecef(coords_lla)
        M2 = torch.linalg.inv(J)
    elif d_sys == 'enu':
        if dst_ref_lla is None: raise ValueError("dst_ref_lla needed for ENU covariance")
        r, _ = _to_tensor(dst_ref_lla)
        r = r.to(device)
        if r.shape[0] == 1 and N > 1: r = r.expand(N, -1)
        M2 = _get_rot_ecef_to_enu(r)
    else:
        M2 = I
        
    M = M2 @ M1
    res = M @ t_cov @ M.transpose(1, 2)
    return _from_tensor(res, meta, output_type=output_type)

def convert_vector(
    vec: Any,
    src_sys: str,
    dst_sys: str,
    coords: Any,
    coords_sys: str = 'lla',
    src_ref_lla: Optional[Any] = None,
    dst_ref_lla: Optional[Any] = None,
    coords_ref_lla: Optional[Any] = None,
    output_type: str = "auto"
) -> Any:
    # Logic similar to covariance but for vector (N,3)
    t_vec, meta = _to_tensor(vec)
    t_coords, _ = _to_tensor(coords)
    device = t_vec.device
    t_coords = t_coords.to(device)
    
    # 1. Get LLA Coords
    coords_lla = t_coords
    if coords_sys != 'lla':
         coords_lla = _to_tensor(convert_coordinates(
            t_coords, coords_sys, 'lla', 
            src_ref_lla=coords_ref_lla, dst_ref_lla=None
        ))[0].to(device)
         
    # 2. Matrices
    N = t_vec.shape[0]
    I = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0).expand(N, -1, -1)
    
    v_sys = src_sys.lower().strip()
    d_sys = dst_sys.lower().strip()

    if v_sys == 'lla': M1 = _get_jac_lla_to_ecef(coords_lla)
    elif v_sys == 'enu':
        if src_ref_lla is None: raise ValueError("src_ref_lla required")
        r, _ = _to_tensor(src_ref_lla)
        r = r.to(device)
        if r.shape[0] == 1 and N > 1: r = r.expand(N, -1)
        M1 = _get_rot_ecef_to_enu(r).transpose(1, 2)
    else: M1 = I
    
    if d_sys == 'lla':
        J = _get_jac_lla_to_ecef(coords_lla)
        M2 = torch.linalg.inv(J)
    elif d_sys == 'enu':
        if dst_ref_lla is None: raise ValueError("dst_ref_lla required")
        r, _ = _to_tensor(dst_ref_lla)
        r = r.to(device)
        if r.shape[0] == 1 and N > 1: r = r.expand(N, -1)
        M2 = _get_rot_ecef_to_enu(r)
    else: M2 = I
    
    M = M2 @ M1
    
    # Vec Transform
    # (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
    if t_vec.ndim == 2:
        v_exp = t_vec.unsqueeze(-1)
        v_out = M @ v_exp
        res = v_out.squeeze(-1)
    else:
        # Assuming (N,3)
        res = (M @ t_vec.unsqueeze(-1)).squeeze(-1)
        
    return _from_tensor(res, meta, output_type=output_type)
