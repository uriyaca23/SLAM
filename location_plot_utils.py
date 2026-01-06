
import torch
import numpy as np
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
            margin={"r":0,"t":40,"l":0,"b":0},
            showlegend=True
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
        return torch.tensor(location_utils.convert_coordinates(
            coords, 
            coords_sys=sys, 
            dst_sys='lla', 
            src_ref_lla=ref_lla if sys.lower()=='enu' else None
        ))

    def _update_center(self, lla_tensor: torch.Tensor):
        if not self._center_updated and lla_tensor.numel() > 0:
            avg_lat = lla_tensor[:, 0].mean().item()
            avg_lon = lla_tensor[:, 1].mean().item()
            self.map_layout['map']['center'] = dict(lat=avg_lat, lon=avg_lon)
            self.map_layout['map']['zoom'] = 14
            self._center_updated = True

    def _process_values(self, values: Any) -> Tuple[np.ndarray, str]:
        if values is None: return None, 'none'
        if isinstance(values, torch.Tensor): vals = values.detach().cpu().numpy()
        else: vals = np.array(values)
        if np.issubdtype(vals.dtype, np.number): return vals, 'numeric'
        elif np.issubdtype(vals.dtype, np.datetime64) or (vals.size > 0 and hasattr(vals[0], 'isoformat')):
            try: return vals.astype("datetime64[ns]"), 'date'
            except: pass
        return vals, 'categorical'

    def add_points(
        self,
        coords: Any,
        sys: str = 'lla',
        ref_lla: Optional[Any] = None,
        values: Optional[Any] = None,
        label: str = "Points",
        color: str = "blue",
        cmap: str = "Viridis",
        marker_size: int = 8,
        symbol: str = "circle", 
        opacity: float = 1.0,
        colorbar_title: Optional[str] = None
    ):
        lla = self._to_lla(coords, sys, ref_lla)
        self._update_center(lla)
        vals, val_type = self._process_values(values)
        
        self._layers.append({
            'type': 'points',
            'lat': lla[:, 0].numpy(),
            'lon': lla[:, 1].numpy(),
            'values': vals,
            'val_type': val_type,
            'label': label,
            'color': color,
            'cmap': cmap,
            'size': marker_size,
            'symbol': symbol,
            'opacity': opacity,
            'cbar_title': colorbar_title if colorbar_title else (label + " Value")
        })

    def add_covariance_2d(
        self,
        coords: Any, # Needed for location
        covariance: Any,
        cov_sys: str = 'lla',
        coords_sys: str = 'lla',
        cov_ref_lla: Optional[Any] = None,
        coords_ref_lla: Optional[Any] = None,
        values: Optional[Any] = None,
        label: str = "Covariance",
        sigma: float = 3.0,
        color: str = "red",
        cmap: str = "Viridis",
        opacity: float = 1.0,
        colorbar_title: Optional[str] = None
    ):
        # 1. Convert Coords to LLA for center update and plotting
        lla = self._to_lla(coords, coords_sys, coords_ref_lla)
        self._update_center(lla)
        
        # 2. Convert Covariance to ENU (Local per point)
        # We need covariance at each point in a local ENU frame tangent to that point.
        # This means dst_sys='enu', dst_ref_lla = THE POINTS THEMSELVES (lla)
        # Src Ref depends on cov_sys
        
        cov_enu = torch.tensor(location_utils.convert_covariance(
            cov=covariance,
            cov_sys=cov_sys,
            dst_sys='enu',
            coords=coords,
            coords_sys=coords_sys,
            src_ref_lla=cov_ref_lla, # Ref for incoming covariance
            dst_ref_lla=lla,         # We want ENU at the point location
            coords_ref_lla=coords_ref_lla # Ref for incoming coords
        )) # (N, 3, 3)
        
        cov_2d = cov_enu[:, 0:2, 0:2]
        ellipse_enu_2d = self._get_ellipse_points(cov_2d, sigma=sigma) # (N, P, 2)
        
        # 3. Convert Ellipses back to LLA for plotting
        # Ellipse points are in local ENU.
        zeros = torch.zeros((ellipse_enu_2d.shape[0], ellipse_enu_2d.shape[1], 1), device=ellipse_enu_2d.device, dtype=torch.float64)
        ellipse_enu = torch.cat([ellipse_enu_2d, zeros], dim=2)
        N, P, _ = ellipse_enu.shape
        ellipse_flat = ellipse_enu.reshape(-1, 3)
        
        # These are offsets in ENU. We need to add them to ref (LLA).
        # OR, treat them as points in ENU frame centered at LLA.
        # So convert ENU(ref=LLA) -> LLA
        refs_flat = lla.unsqueeze(1).expand(-1, P, -1).reshape(-1, 3)
        
        ellipse_lla = torch.tensor(location_utils.convert_coordinates(
            ellipse_flat,
            coords_sys='enu',
            dst_sys='lla',
            src_ref_lla=refs_flat
        )).reshape(N, P, 3)
        
        vals, val_type = self._process_values(values)
        
        self._layers.append({
            'type': 'lines',
            'lines_lla': ellipse_lla.cpu(),
            'values': vals,
            'val_type': val_type,
            'label': label,
            'color': color,
            'cmap': cmap,
            'opacity': opacity,
            'cbar_title': colorbar_title if colorbar_title else label
        })

    def add_velocity_2d(
        self,
        coords: Any,
        velocity: Any,
        vel_sys: str = 'enu',
        coords_sys: str = 'lla',
        vel_ref_lla: Optional[Any] = None,
        coords_ref_lla: Optional[Any] = None,
        values: Optional[Any] = None,
        label: str = "Velocity",
        scale: float = 1.0,
        color: str = "green",
        cmap: str = "Viridis",
        opacity: float = 1.0,
        colorbar_title: Optional[str] = None
    ):
        lla = self._to_lla(coords, coords_sys, coords_ref_lla)
        self._update_center(lla)
        
        # Convert Velocity to Local ENU
        vel_enu = torch.tensor(location_utils.convert_vector(
            vec=velocity,
            vec_sys=vel_sys,
            dst_sys='enu',
            coords=coords,
            coords_sys=coords_sys,
            src_ref_lla=vel_ref_lla,
            dst_ref_lla=lla, # Local ENU
            coords_ref_lla=coords_ref_lla
        ))
        
        # Geometry Generation (Same as before)
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
        
        lines_lla = torch.tensor(location_utils.convert_coordinates(
            pts_flat, 'enu', 'lla', src_ref_lla=refs_flat
        )).reshape(N, P, 3)

        vals, val_type = self._process_values(values)
        self._layers.append({
            'type': 'lines', 'lines_lla': lines_lla.cpu(), 'values': vals, 'val_type': val_type,
            'label': label, 'color': color, 'cmap': cmap, 'opacity': opacity,
            'cbar_title': colorbar_title if colorbar_title else label
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
    def _generate_trace_data(self, layer, start_idx=None, end_idx=None, is_frame=False):
        # (Same as before)
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
            
        if start_idx is not None:
             def slice_arr(arr):
                 if hasattr(arr, 'shape'):return arr[start_idx:end_idx] if arr.shape[0]>start_idx else arr[0:0]
                 return arr
             if layer['type'] == 'points': lats = slice_arr(layer['lat']); lons = slice_arr(layer['lon'])
             else: lines = slice_arr(layer['lines_lla'])
             vals = layer['values']; 
             if vals is not None: vals = slice_arr(vals)
        else:
             if layer['type'] == 'points': lats, lons = layer['lat'], layer['lon']
             else: lines = layer['lines_lla']
             vals = layer['values']

        traces = []
        name_prefix = layer['label']
        
        if layer['val_type'] == 'categorical' and layer['values'] is not None:
             all_vals = layer['values']
             unique_cats = np.unique(all_vals)
             cat_colors = pcolors.sample_colorscale(layer['cmap'], np.linspace(0, 1, len(unique_cats)))
             for i, cat in enumerate(unique_cats):
                 idxs = np.where(vals == cat)[0] if vals is not None else []
                 t_lat, t_lon = [], []
                 if len(idxs) > 0:
                     if layer['type'] == 'points': t_lat, t_lon = lats[idxs], lons[idxs]
                     else: t_lat, t_lon = flatten_lines(lines[idxs])
                 sym = layer.get('symbol', 'circle')
                 trace = dict(
                     type='scattermap', lat=t_lat, lon=t_lon,
                     mode='lines' if layer['type']=='lines' else ('text' if sym in ['cross','x'] else 'markers'),
                     name=f"{name_prefix} - {cat}", legendgroup=f"{name_prefix}_{cat}", showlegend=False, opacity=layer['opacity']
                 )
                 if layer['type'] == 'lines': trace['line'] = dict(color=cat_colors[i], width=2)
                 elif sym in ['cross', 'x']: trace['text']=['x']*len(t_lat); trace['textfont']=dict(size=layer['size'], color=cat_colors[i])
                 else: trace['marker'] = dict(size=layer['size'], color=cat_colors[i], symbol=sym)
                 traces.append(trace)
        else:
             has_vals = (vals is not None)
             v_min_global, v_max_global = 0.0, 1.0
             if layer['values'] is not None:
                 if layer['val_type'] == 'date': gl_v = layer['values'].astype("datetime64[ns]").astype(float)
                 else: gl_v = layer['values']
                 v_min_global, v_max_global = gl_v.min(), gl_v.max()

             if layer['type'] == 'lines' and layer['values'] is not None:
                 num_bins = 20
                 norm_factor = (v_max_global - v_min_global) if v_max_global > v_min_global else 1.0
                 sample_colors = pcolors.sample_colorscale(layer['cmap'], np.linspace(0, 1, num_bins))
                 curr_vals = vals.astype("datetime64[ns]").astype(float) if (has_vals and layer['val_type']=='date') else vals
                 slice_bins = np.floor((curr_vals - v_min_global) / norm_factor * (num_bins - 1e-6)).astype(int) if len(curr_vals)>0 else []
                 for b in range(num_bins):
                     t_lat, t_lon = [], []
                     if has_vals and len(vals) > 0:
                         idxs = np.where(slice_bins == b)[0]
                         if len(idxs) > 0: t_lat, t_lon = flatten_lines(lines[idxs])
                     traces.append(dict(
                         type='scattermap', lat=t_lat, lon=t_lon, mode='lines',
                         line=dict(color=sample_colors[b], width=2), name=name_prefix, legendgroup=name_prefix, showlegend=False, opacity=layer['opacity']
                     ))
             else:
                 t_lat, t_lon = [], []
                 if layer['type'] == 'lines': t_lat, t_lon = flatten_lines(lines)
                 else: 
                     if lats.shape[0] > 0: t_lat, t_lon = lats, lons
                 trace = dict(type='scattermap', lat=t_lat, lon=t_lon, name=name_prefix, legendgroup=name_prefix, showlegend=False, opacity=layer['opacity'])
                 if layer['type'] == 'lines':
                     trace['mode'] = 'lines'; trace['line'] = dict(color=layer['color'], width=2)
                 else:
                     sym = layer.get('symbol', 'circle')
                     if sym in ['cross', 'x']:
                         trace['mode'] = 'text'; trace['text'] = ['x']*len(t_lat)
                         tf = dict(size=layer['size'], color=layer['color'])
                         if has_vals:
                             v = vals.astype("datetime64[ns]").astype(float) if layer['val_type']=='date' else vals
                             tf['color'] = v
                         trace['textfont'] = tf
                     else:
                         trace['mode'] = 'markers'
                         m = dict(size=layer['size'], symbol=sym, opacity=layer['opacity'])
                         if has_vals:
                             v = vals.astype("datetime64[ns]").astype(float) if layer['val_type']=='date' else vals
                             m['color'] = v; m['colorscale'] = layer['cmap']; m['showscale'] = False
                             m['cmin'], m['cmax'] = float(v_min_global), float(v_max_global)
                             m['cauto'], m['autocolorscale'] = False, False
                         else: m['color'] = layer['color']
                         trace['marker'] = m
                 traces.append(trace)
        return traces

    def _get_static_traces(self):
        # (Same as before)
        traces = []
        for layer in self._layers:
            if layer['val_type'] in ['numeric', 'date'] and layer['values'] is not None:
                cbar_dict = dict(title=layer['cbar_title'], x=1.02, len=0.8, y=0.3)
                all_vals = layer['values']
                if layer['val_type'] == 'date':
                    cbar_dict['tickformat'] = '%Y-%m-%d\n%H:%M:%S'
                    v_f = all_vals.astype("datetime64[ns]").astype(float)
                    cbar_vals = [v_f.min(), v_f.max()]
                else: cbar_vals = [all_vals.min(), all_vals.max()]
                traces.append(dict(
                    type='scattermap', lat=[None], lon=[None], mode='markers',
                    marker=dict(size=0, color=cbar_vals, colorscale=layer['cmap'], showscale=True, colorbar=cbar_dict),
                    showlegend=False, hoverinfo='skip'
                ))
            
            name = layer['label']
            sym = layer.get('symbol', 'circle')
            if layer['val_type'] == 'categorical' and layer['values'] is not None:
                unique_cats = np.unique(layer['values'])
                cat_colors = pcolors.sample_colorscale(layer['cmap'], np.linspace(0, 1, len(unique_cats)))
                for i, cat in enumerate(unique_cats):
                    leg = dict(type='scattermap', lat=[None], lon=[None], name=f"{name} - {cat}", legendgroup=f"{name}_{cat}", showlegend=True, opacity=1)
                    if layer['type'] == 'lines': leg['mode'] = 'lines'; leg['line'] = dict(color=cat_colors[i], width=2)
                    elif sym in ['cross', 'x']: leg['mode']='text'; leg['text']=['x']; leg['textfont']=dict(size=layer['size'], color=cat_colors[i])
                    else: leg['mode']='markers'; leg['marker']=dict(size=layer['size'], color=cat_colors[i], symbol=sym)
                    traces.append(leg)
            else:
                leg = dict(type='scattermap', lat=[None], lon=[None], name=name, legendgroup=name, showlegend=True, opacity=1)
                rep_color = layer['color']
                if layer['values'] is not None and layer['cmap']: rep_color = pcolors.sample_colorscale(layer['cmap'], [0.5])[0]
                if layer['type'] == 'lines': leg['mode'] = 'lines'; leg['line']=dict(color=rep_color, width=2)
                elif sym in ['cross', 'x']: leg['mode']='text'; leg['text']=['x']; leg['textfont']=dict(size=layer['size'], color=rep_color)
                else: leg['mode']='markers'; leg['marker']=dict(size=layer['size'], color=rep_color, symbol=sym)
                traces.append(leg)
        return traces

    def _build_plot(self):
        self.fig.update_layout(self.map_layout)
        for t in self._get_static_traces(): self.fig.add_trace(t)
        
        start_idx = 0 if self.use_slider else None
        end_idx = self.slide_window if self.use_slider else None
        max_N = 0
        date_vals = None
        for layer in self._layers:
            if layer['type']=='points': n=len(layer['lat'])
            else: n=layer['lines_lla'].shape[0]
            if n>max_N: max_N=n
            if layer['val_type']=='date' and layer['values'] is not None and date_vals is None: date_vals=layer['values']
            
        initial_traces = []
        for layer in self._layers:
            initial_traces.extend(self._generate_trace_data(layer, start_idx, min(end_idx, max_N) if end_idx else None))
        for t in initial_traces: self.fig.add_trace(t)
        
        if self.use_slider and max_N > 0:
            dynamic_indices = list(range(len(self.fig.data) - len(initial_traces), len(self.fig.data)))
            frames = []
            steps = []
            for f_idx in range(0, max_N, self.slide_step):
                frame_data = []
                for layer in self._layers:
                    frame_data.extend(self._generate_trace_data(layer, f_idx, min(f_idx+self.slide_window, max_N), is_frame=True))
                frames.append(go.Frame(data=frame_data, traces=dynamic_indices, name=str(f_idx)))
                label_txt = str(f_idx)
                if date_vals is not None and f_idx < len(date_vals):
                    try: label_txt = pd.to_datetime(date_vals[f_idx]).strftime('%Y-%m-%d %H:%M:%S')
                    except: pass
                steps.append(dict(method="animate", args=[[str(f_idx)], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))], label=label_txt))
            self.fig.frames = frames
            self.fig.update_layout(
                sliders=[dict(active=0, currentvalue={"prefix": "Time: "}, pad={"t": 50}, steps=steps)],
                updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=1.0, xanchor="left", yanchor="top", direction="right", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]), dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])]
            )

    def save(self, filename: str):
        self._build_plot()
        self.fig.write_html(filename)
    
    def show(self):
        self._build_plot()
        self.fig.show()
