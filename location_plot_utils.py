
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors
from typing import Optional, List, Union, Any, Tuple

import location_utils

class LocationPlotter:
    def __init__(
        self, 
        title: str = "Location Plot",
        map_style: str = "open-street-map", 
        center_lla: Optional[Union[List[float], torch.Tensor]] = None, 
        zoom: int = 10,
        use_slider: bool = False,
        slide_window: int = 20,
        slide_step: int = 1,
        map_url: Optional[str] = None
    ):
        self.fig = go.Figure()
        self.use_slider = use_slider
        self.slide_window = slide_window
        self.slide_step = slide_step
        self.map_url = map_url
        self.title = title
        
        self._layers = []
        
        style = "white-bg" if self.map_url else map_style
        
        self.map_layout = dict(
            title=self.title,
            map=dict(style=style, zoom=zoom),
            margin={"r":0,"t":30,"l":0,"b":0},
            showlegend=True,
            # height=700, # Reverted: User wants responsive sizing
            # autosize=False # Reverted
        )
        if self.map_url:
            self.map_layout['map']['layers'] = [{
                "below": 'traces', "sourcetype": "raster", "source": [self.map_url]
            }]
        
        if center_lla is not None:
             # Assume center_lla is simple
             if isinstance(center_lla, torch.Tensor):
                 c = center_lla.detach().cpu().numpy().flatten().tolist()
             else: c = center_lla
             self.map_layout['map']['center'] = dict(lat=c[0], lon=c[1])
             self._center_updated = True
        else:
             self._center_updated = False

    def _to_lla(self, coords, sys, ref_lla) -> torch.Tensor:
        # Helper to get LLA from inputs using functional API
        return location_utils.convert_coordinates(
            coords, 
            coords_sys=sys, 
            dst_sys='lla', 
            src_ref_lla=ref_lla if sys.lower()=='enu' else None,
            output_type='tensor'
        )

    def _update_center(self, lla_tensor: torch.Tensor):
        if not self._center_updated and lla_tensor.numel() > 0:
            avg_lat = lla_tensor[:, 0].mean().item()
            avg_lon = lla_tensor[:, 1].mean().item()
            self.map_layout['map']['center'] = dict(lat=avg_lat, lon=avg_lon)
            self.map_layout['map']['zoom'] = 14
            self._center_updated = True

    def _process_values(self, values: Any) -> Tuple[Any, str]:
        # Return raw values and their detected type 'numeric', 'date', 'categorical', or 'none'
        if values is None: return None, 'none'
        if isinstance(values, torch.Tensor): vals = values.detach().cpu().numpy()
        else: vals = np.array(values)
        
        if vals.size == 0: return vals, 'none'

        if np.issubdtype(vals.dtype, np.number): return vals, 'numeric'
        elif np.issubdtype(vals.dtype, np.datetime64) or hasattr(vals.flat[0], 'isoformat'):
            try: 
                # Try to cast to datetime64[ns]
                return pd.to_datetime(vals).to_numpy(dtype="datetime64[ns]"), 'date'
            except: pass
        
        return vals, 'categorical'

    def add_points(
        self,
        coords: Any,
        sys: str = 'lla',
        ref_lla: Optional[Any] = None,
        timestep_values: Optional[Any] = None,
        categorical_values: Optional[Any] = None,
        label: str = "Points",
        color: str = "blue",
        cmap: str = "Viridis",
        marker_size: int = 8,
        symbol: str = "circle", 
        opacity: float = 1.0,
        colorbar_title: Optional[str] = None,
        color_by_value: bool = True
    ):
        lla = self._to_lla(coords, sys, ref_lla)
        self._update_center(lla)
        
        time_vals, time_type = self._process_values(timestep_values)
        cat_vals, cat_type = self._process_values(categorical_values)
        
        self._layers.append({
            'type': 'points',
            'lat': lla[:, 0].numpy(),
            'lon': lla[:, 1].numpy(),
            'timestep_values': time_vals,
            'timestep_type': time_type,
            'categorical_values': cat_vals,
            'categorical_type': cat_type,
            'label': label,
            'color': color,
            'cmap': cmap,
            'size': marker_size,
            'symbol': symbol,
            'opacity': opacity,
            'cbar_title': colorbar_title if colorbar_title else (label + " Value"),
            'color_by_value': color_by_value
        })

    def add_covariance_2d(
        self,
        coords: Any,
        covariance: Any,
        cov_sys: str = 'lla',
        coords_sys: str = 'lla',
        cov_ref_lla: Optional[Any] = None,
        coords_ref_lla: Optional[Any] = None,
        timestep_values: Optional[Any] = None,
        categorical_values: Optional[Any] = None,
        label: str = "Covariance",
        sigma: float = 3.0,
        color: str = "red",
        cmap: str = "Viridis",
        opacity: float = 1.0,
        colorbar_title: Optional[str] = None,
        color_by_value: bool = True
    ):
        # 1. Convert Coords to LLA for center update and plotting
        lla = self._to_lla(coords, coords_sys, coords_ref_lla)
        self._update_center(lla)
        
        # 2. Convert Covariance to ENU (Local per point)
        cov_enu = location_utils.convert_covariance(
            cov=covariance,
            cov_sys=cov_sys,
            dst_sys='enu',
            coords=coords,
            coords_sys=coords_sys,
            src_ref_lla=cov_ref_lla,
            dst_ref_lla=lla,
            coords_ref_lla=coords_ref_lla,
            output_type='tensor'
        ) # (N, 3, 3)
        
        cov_2d = cov_enu[:, 0:2, 0:2]
        ellipse_enu_2d = self._get_ellipse_points(cov_2d, sigma=sigma) # (N, P, 2)
        
        # 3. Convert Ellipses back to LLA for plotting
        zeros = torch.zeros((ellipse_enu_2d.shape[0], ellipse_enu_2d.shape[1], 1), device=ellipse_enu_2d.device, dtype=torch.float64)
        ellipse_enu = torch.cat([ellipse_enu_2d, zeros], dim=2)
        N, P, _ = ellipse_enu.shape
        ellipse_flat = ellipse_enu.reshape(-1, 3)
        
        refs_flat = lla.unsqueeze(1).expand(-1, P, -1).reshape(-1, 3)
        
        ellipse_lla = location_utils.convert_coordinates(
            ellipse_flat,
            coords_sys='enu',
            dst_sys='lla',
            src_ref_lla=refs_flat,
            output_type='tensor'
        ).reshape(N, P, 3)
        
        time_vals, time_type = self._process_values(timestep_values)
        cat_vals, cat_type = self._process_values(categorical_values)
        
        self._layers.append({
            'type': 'lines',
            'lines_lla': ellipse_lla.cpu(),
            'timestep_values': time_vals,
            'timestep_type': time_type,
            'categorical_values': cat_vals,
            'categorical_type': cat_type,
            'label': label,
            'color': color,
            'cmap': cmap,
            'opacity': opacity,
            'cbar_title': colorbar_title if colorbar_title else label,
            'color_by_value': color_by_value
        })

    def add_velocity_2d(
        self,
        coords: Any,
        velocity: Any,
        vel_sys: str = 'enu',
        coords_sys: str = 'lla',
        vel_ref_lla: Optional[Any] = None,
        coords_ref_lla: Optional[Any] = None,
        timestep_values: Optional[Any] = None,
        categorical_values: Optional[Any] = None,
        label: str = "Velocity",
        scale: float = 1.0,
        color: str = "green",
        cmap: str = "Viridis",
        opacity: float = 1.0,
        colorbar_title: Optional[str] = None,
        color_by_value: bool = True
    ):
        lla = self._to_lla(coords, coords_sys, coords_ref_lla)
        self._update_center(lla)
        
        # Convert Velocity to Local ENU
        vel_enu = location_utils.convert_vector(
            vec=velocity,
            vec_sys=vel_sys,
            dst_sys='enu',
            coords=coords,
            coords_sys=coords_sys,
            src_ref_lla=vel_ref_lla,
            dst_ref_lla=lla, 
            coords_ref_lla=coords_ref_lla,
            output_type='tensor'
        )
        
        # Geometry Generation
        end_enu = vel_enu * scale
        v2 = end_enu[:, 0:2]
        mag = torch.norm(v2, dim=1) + 1e-9
        angles = torch.atan2(v2[:, 1], v2[:, 0])
        tip_len = mag * 0.2
        angle_left = angles + 3*np.pi/4
        angle_right = angles - 3*np.pi/4
        c, s = torch.cos, torch.sin
        tip1_x = v2[:, 0] + tip_len * c(angle_left)
        tip1_y = v2[:, 1] + tip_len * s(angle_left)
        tip2_x = v2[:, 0] + tip_len * c(angle_right)
        tip2_y = v2[:, 1] + tip_len * s(angle_right)
        
        zeros = torch.zeros_like(tip1_x)
        def stack_vec(x, y): return torch.stack([x, y, zeros], dim=1)
        
        center_pts = torch.zeros_like(end_enu)
        end_pts = torch.cat([v2, zeros.unsqueeze(1)], dim=1)
        tip1_pts = stack_vec(tip1_x, tip1_y)
        tip2_pts = stack_vec(tip2_x, tip2_y)
        
        pts_enu = torch.stack([center_pts, end_pts, tip1_pts, end_pts, tip2_pts], dim=1)
        N, P, _ = pts_enu.shape
        pts_flat = pts_enu.reshape(-1, 3)
        refs_flat = lla.unsqueeze(1).expand(-1, P, -1).reshape(-1, 3)
        
        lines_lla = location_utils.convert_coordinates(
            pts_flat, 'enu', 'lla', src_ref_lla=refs_flat, output_type='tensor'
        ).reshape(N, P, 3)

        time_vals, time_type = self._process_values(timestep_values)
        cat_vals, cat_type = self._process_values(categorical_values)
        
        self._layers.append({
            'type': 'lines', 'lines_lla': lines_lla.cpu(), 
            'timestep_values': time_vals, 'timestep_type': time_type,
            'categorical_values': cat_vals, 'categorical_type': cat_type,
            'label': label, 'color': color, 'cmap': cmap, 'opacity': opacity,
            'cbar_title': colorbar_title if colorbar_title else label,
            'color_by_value': color_by_value
        })

    def _get_ellipse_points(self, cov_batch: torch.Tensor, sigma: float = 3.0, num_points: int = 30) -> torch.Tensor:
        # Same impl
        N = cov_batch.shape[0]
        L, V = torch.linalg.eigh(cov_batch) 
        scale = torch.sqrt(torch.clamp(L, min=1e-12)) * sigma
        theta = torch.linspace(0, 2*np.pi, num_points + 1, device=cov_batch.device, dtype=cov_batch.dtype)
        circle = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) 
        circle_batch = circle.unsqueeze(0).expand(N, -1, -1)
        S_sqrt = torch.diag_embed(scale) 
        T = V @ S_sqrt 
        ellipse_points = circle_batch @ T.transpose(1, 2)
        return ellipse_points

    # --- Retain Logic ---
    def _generate_trace_data(self, layer, mode, global_clim=None, start_idx=None, end_idx=None, is_frame=False):
        # Determine Values for Mode
        vals_to_use = None
        type_to_use = 'none'
        
        if mode == 'timestep':
            vals_to_use = layer.get('timestep_values')
            type_to_use = layer.get('timestep_type', 'none')
        elif mode == 'categorical':
            vals_to_use = layer.get('categorical_values')
            type_to_use = layer.get('categorical_type', 'none')
        
        # Fallback to Name mode if data missing for requested mode
        if mode != 'name' and vals_to_use is None:
             # Effectively return empty traces for this mode so it doesn't show garbage
             # But we must preserve TRACE COUNT for frames consistency?
             # E.g. if categorical has 3 categories globally, we need 3 empty traces.
             # If timestep lines, we need 20 empty traces.
             # This is tricky if 'vals_to_use' is None, we don't know categories.
             # Easier assumption: If no data, we yield "empty" traces based on GLOBAL layer properties?
             pass 

        # Slice Data
        def flatten_lines(lines_tensor):
            if lines_tensor.numel() == 0: return [], []
            arr = lines_tensor.numpy()
            N, P, _ = arr.shape
            padded = np.full((N, P+1, 2), np.nan)
            padded[:, :P, :] = arr[:, :, :2] 
            flat = padded.reshape(-1, 2)
            flat_obj = flat.astype(object)
            mask = np.isnan(flat); flat_obj[mask] = None
            return flat_obj[:, 0].tolist(), flat_obj[:, 1].tolist()
            
        def slice_arr(arr):
             if hasattr(arr, 'shape'):return arr[start_idx:end_idx] if arr.shape[0]>start_idx else arr[0:0]
             return arr

        if start_idx is not None:
             if layer['type'] == 'points': lats = slice_arr(layer['lat']); lons = slice_arr(layer['lon'])
             else: lines = slice_arr(layer['lines_lla'])
             curr_vals = slice_arr(vals_to_use) if vals_to_use is not None else None
        else:
             if layer['type'] == 'points': lats, lons = layer['lat'], layer['lon']
             else: lines = layer['lines_lla']
             curr_vals = vals_to_use
        
        traces = []
        name_prefix = layer['label'] # No suffix for Name mode
        
        # --- MODE SPECIFIC GENERATION ---
        
        if mode == 'categorical' and vals_to_use is not None:
             # Use ALL values to determine categories structure
             all_vals = layer.get('categorical_values')
             unique_cats = np.unique(all_vals) if all_vals is not None else np.array([])
             cat_colors = pcolors.sample_colorscale(layer['cmap'], np.linspace(0, 1, len(unique_cats)))
             
             for i, cat in enumerate(unique_cats):
                 idxs = np.where(curr_vals == cat)[0] if curr_vals is not None else []
                 t_lat, t_lon = [], []
                 if len(idxs) > 0:
                     if layer['type'] == 'points': t_lat, t_lon = lats[idxs], lons[idxs]
                     else: t_lat, t_lon = flatten_lines(lines[idxs])
                 
                 sym = layer.get('symbol', 'circle')
                 trace = dict(
                     type='scattermap', lat=t_lat, lon=t_lon,
                     mode='lines' if layer['type']=='lines' else ('text' if sym in ['cross','x'] else 'markers'),
                     showlegend=False, opacity=layer['opacity'],
                     name=f"{name_prefix} - {cat}", legendgroup=f"{name_prefix}_{cat}"
                 )
                 # Style
                 c = cat_colors[i]
                 if layer['type'] == 'lines': trace['line'] = dict(color=c, width=2)
                 elif sym in ['cross', 'x']: trace['text']=['x']*len(t_lat); trace['textfont']=dict(size=layer['size'], color=c)
                 else: trace['marker'] = dict(size=layer['size'], color=c, symbol=sym)
                 traces.append(trace)
        
        elif mode == 'timestep' and vals_to_use is not None:
             # Gradient Line or Color Axis Marker
             do_gradient_lines = (layer['type'] == 'lines')
             v_min, v_max = global_clim if global_clim else (0.0, 1.0)
             
             if do_gradient_lines:
                  num_bins = 20
                  norm_factor = (v_max - v_min) if v_max > v_min else 1.0
                  sample_colors = pcolors.sample_colorscale(layer['cmap'], np.linspace(0, 1, num_bins))
                  
                  v_float = curr_vals.astype("datetime64[ns]").astype(float) if (type_to_use=='date') else curr_vals.astype(float)
                  # Binning relative to Global Range
                  slice_bins = np.floor((v_float - v_min) / norm_factor * (num_bins - 1e-6)).astype(int) if (curr_vals is not None and len(curr_vals)>0) else []
                  
                  for b in range(num_bins):
                      t_lat, t_lon = [], []
                      if curr_vals is not None and len(curr_vals) > 0:
                          idxs = np.where(slice_bins == b)[0]
                          if len(idxs) > 0: t_lat, t_lon = flatten_lines(lines[idxs])
                      traces.append(dict(
                          type='scattermap', lat=t_lat, lon=t_lon, mode='lines',
                          line=dict(color=sample_colors[b], width=2), 
                          showlegend=False, opacity=layer['opacity'], hoverinfo='skip'
                      ))
             else:
                  # Scatter (Points) with ColorAxis
                  t_lat, t_lon = [], []
                  if lats.shape[0] > 0: t_lat, t_lon = lats, lons
                  
                  sym = layer.get('symbol', 'circle')
                  if sym in ['cross', 'x']:
                      trace = dict(
                          type='scattermap', lat=t_lat, lon=t_lon, mode='text', text=['x']*len(t_lat),
                          showlegend=False, opacity=layer['opacity']
                      )
                      v_float = curr_vals.astype("datetime64[ns]").astype(float) if (type_to_use=='date') else curr_vals
                      trace['textfont'] = dict(size=layer['size'], color=v_float, coloraxis='coloraxis')
                  else:
                      trace = dict(
                           type='scattermap', lat=t_lat, lon=t_lon, mode='markers',
                           showlegend=False, opacity=layer['opacity']
                      )
                      v_float = curr_vals.astype("datetime64[ns]").astype(float) if (type_to_use=='date') else curr_vals
                      trace['marker'] = dict(
                          size=layer['size'], symbol=sym,
                          color=v_float, coloraxis='coloraxis'
                      )
                  traces.append(trace)

        else:
             # Name Mode (Default fallback)
             # If "fallback" logic is triggered (e.g. asking for categorical but none exists),
             # we return Empty Traces matching structure if we tracked structure, 
             # OR we return a Name trace?
             # Frame consistency requires same NUMBER of traces.
             # If we are in 'categorical' mode but this layer has no categories, we should ideally produce 0 traces?
             # BUT if we produced 0 traces for frame 0, and 0 for frame 1, that matches.
             # The problem is mixing layers.
             # If Layer A has categories, Layer B does not.
             # In Categorical Mode: Layer A shows traces. Layer B shows nothing? 
             # User said: "timestep and categorical are optional inputs, and if inserted they should be added to a switched button"
             # This implies layers without them just don't participate or show default?
             # Let's assume they show NOTHING in that specific mode if data is missing.
             
             if mode == 'name':
                 t_lat, t_lon = [], []
                 if layer['type'] == 'lines': 
                     if lines.shape[0] > 0: t_lat, t_lon = flatten_lines(lines)
                 else: 
                     if lats.shape[0] > 0: t_lat, t_lon = lats, lons
                 
                 trace = dict(type='scattermap', lat=t_lat, lon=t_lon, showlegend=False, opacity=layer['opacity'])
                 if layer['type'] == 'lines':
                     trace['mode'] = 'lines'; trace['line'] = dict(color=layer['color'], width=2)
                 else:
                     sym = layer.get('symbol', 'circle')
                     if sym in ['cross', 'x']:
                         trace['mode'] = 'text'; trace['text'] = ['x']*len(t_lat)
                         trace['textfont'] = dict(size=layer['size'], color=layer['color'])
                     else:
                         trace['mode'] = 'markers'
                         trace['marker'] = dict(size=layer['size'], color=layer['color'], symbol=sym)
                 traces.append(trace)

        return traces

    def _get_static_traces(self):
        # Only generating "Name" style legend items (per User Req)
        traces = []
        for layer in self._layers:
            name = layer['label']
            sym = layer.get('symbol', 'circle')
            
            trace = dict(
                type='scattermap', lat=[None], lon=[None], 
                name=name, legendgroup=name, showlegend=True, opacity=1
            )
            color = layer['color']
            
            if layer['type'] == 'lines': 
                trace['mode'] = 'lines'
                trace['line'] = dict(color=color, width=2)
            elif sym in ['cross', 'x']: 
                trace['mode']='text'; trace['text']=['x']
                trace['textfont']=dict(size=layer['size'], color=color)
            else: 
                trace['mode']='markers'
                trace['marker']=dict(size=layer['size'], color=color, symbol=sym)
            traces.append(trace)
        return traces

    def _build_plot(self):
        self.fig.update_layout(self.map_layout)
        
        # 1. Calculate Global Timestep Range
        global_min, global_max = float('inf'), float('-inf')
        has_time_data = False
        is_date_axis = False
        
        for layer in self._layers:
            # Check timestep vals
            tv = layer.get('timestep_values')
            tt = layer.get('timestep_type')
            
            if tv is not None and len(tv) > 0:
                has_time_data = True
                if tt == 'date':
                    is_date_axis = True
                    # Actually datetime64[ns] cast to float gives ns. This is fine for min/max
                    vf = tv.astype("datetime64[ns]").astype(float)
                else:
                    vf = tv.astype(float)
                global_min = min(global_min, vf.min())
                global_max = max(global_max, vf.max())

        if not has_time_data: global_min, global_max = 0, 1
        
        # Setup ColorAxis
        cbar_args = dict(
            colorscale="Viridis", 
            colorbar=dict(title="Value", x=1.02, len=0.8, y=0.3),
            cmin=global_min, cmax=global_max,
            showscale=True
        )
        if is_date_axis:
             # Plotly coloraxis doesn't support date formatting automatically in the same way as tickformat?
             # It acts as numeric. We might need tickvals/text if we want pretty dates.
             # For now, let's just leave it numeric/raw or see if standard simple formatting works.
             # Ideally we pass 'tickformat' to colorbar.
             cbar_args['colorbar']['tickformat'] = '%Y-%m-%d\n%H:%M:%S'

        self.fig.update_layout(coloraxis=cbar_args)
        
        # 2. Add Static Legend Traces (Always Visible)
        for t in self._get_static_traces(): self.fig.add_trace(t)
        
        # 3. Generate Data Traces for All Modes
        # Structure: [Static Traces] + [Name Traces] + [Timestep Traces] + [Categorical Traces]
        # We need to track indices to toggle visibility.
        
        static_count = len(self._get_static_traces())
        
        traces_name = []
        traces_time = []
        traces_cat = []
        
        # Slice limits
        start_idx = 0 if self.use_slider else None
        end_idx = self.slide_window if self.use_slider else None
        
        max_N = 0
        date_vals = None # For slider labels
        
        for layer in self._layers:
            if layer['type']=='points': n=len(layer['lat'])
            else: n=layer['lines_lla'].shape[0]
            if n>max_N: max_N=n
            # Try to catch date vals for slider label from 'timestep' if available
            tv = layer.get('timestep_values')
            if tv is not None and date_vals is None: date_vals = tv
            
            # Name Mode
            traces_name.extend(self._generate_trace_data(layer, 'name', None, start_idx, end_idx))
            # Timestep Mode
            traces_time.extend(self._generate_trace_data(layer, 'timestep', (global_min, global_max), start_idx, end_idx))
            # Cat Mode
            traces_cat.extend(self._generate_trace_data(layer, 'categorical', None, start_idx, end_idx))

        # Add all to figure
        # Default Visibility: Name = True, others = False
        all_traces = traces_name + traces_time + traces_cat
        
        # Indices relative to fig.data
        # fig.data has [Static...] then we add [Name... Time... Cat...]
        base_idx = static_count
        name_range = range(base_idx, base_idx + len(traces_name))
        time_range = range(base_idx + len(traces_name), base_idx + len(traces_name) + len(traces_time))
        cat_range = range(base_idx + len(traces_name) + len(traces_time), base_idx + len(all_traces))
        
        for i, t in enumerate(traces_name): t['visible'] = True
        for t in traces_time: t['visible'] = False
        for t in traces_cat: t['visible'] = False
        
        for t in all_traces: self.fig.add_trace(t)
        
        # 4. Buttons
        # We need to construct visibility arrays for the WHOLE fig.data
        # Static traces (0 to static_count-1) -> Always True
        total_traces = len(self.fig.data)
        
        def get_vis(active_range):
            v = [True] * static_count # Static always visible
            # Dynamic part
            # Initialize all dynamic to False
            dyn = [False] * (total_traces - static_count)
            # Enable active range
            for i in active_range:
                if i >= static_count: dyn[i - static_count] = True
            return v + dyn

        buttons = [
            dict(
                label="Name",
                method="update",
                args=[{"visible": get_vis(name_range)}, {"coloraxis.showscale": False}]
            ),
            dict(
                label="Timestep",
                method="update",
                args=[{"visible": get_vis(time_range)}, {"coloraxis.showscale": True}]
            ),
            dict(
                label="Categorical",
                method="update",
                args=[{"visible": get_vis(cat_range)}, {"coloraxis.showscale": False}]
            )
        ]
        
        # 5. Frames (Slider)
        if self.use_slider and max_N > 0:
            frames = []
            steps = []
            
            # Identify indices of dynamic traces in fig.data
            # We must update ALL dynamic traces (Name, Time, Cat) in every frame 
            # so that switching mode while paused/playing works correctly.
            dynamic_indices = list(range(base_idx, total_traces))
            
            for f_idx in range(0, max_N, self.slide_step):
                # Generate data for ALL modes for this frame
                frame_traces = []
                w_start, w_end = f_idx, min(f_idx + self.slide_window, max_N)
                
                # We simply concat the mode generations exactly as we did for initial execution
                for layer in self._layers: frame_traces.extend(self._generate_trace_data(layer, 'name', None, w_start, w_end))
                for layer in self._layers: frame_traces.extend(self._generate_trace_data(layer, 'timestep', (global_min, global_max), w_start, w_end))
                for layer in self._layers: frame_traces.extend(self._generate_trace_data(layer, 'categorical', None, w_start, w_end))
                
                frames.append(go.Frame(data=frame_traces, traces=dynamic_indices, name=str(f_idx)))
                
                label_txt = str(f_idx)
                if date_vals is not None and f_idx < len(date_vals):
                    try: label_txt = pd.to_datetime(date_vals[f_idx]).strftime('%Y-%m-%d %H:%M:%S')
                    except: pass
                steps.append(dict(method="animate", args=[[str(f_idx)], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))], label=label_txt))
            
            self.fig.frames = frames
            self.fig.update_layout(
                sliders=[dict(active=0, currentvalue={"prefix": "Time: "}, pad={"t": 50}, steps=steps)],
                updatemenus=[
                    dict(type="buttons", showactive=False, x=0.0, y=1.0, xanchor="left", yanchor="top", direction="right", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]), dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])]),
                    dict(type="buttons", showactive=True, x=0.2, y=1.0, xanchor="left", yanchor="top", direction="right", buttons=buttons)
                ]
            )
        else:
            # Just mode buttons
            self.fig.update_layout(
                updatemenus=[
                    dict(type="buttons", showactive=True, x=0.0, y=1.0, xanchor="left", yanchor="top", direction="right", buttons=buttons)
                ]
            )
            # Hide coloraxis default
            self.fig.update_layout(coloraxis=dict(showscale=False))

    def save(self, filename: str):
        self._build_plot()
        self.fig.write_html(filename)
    
    def show(self):
        self._build_plot()
        self.fig.show()
