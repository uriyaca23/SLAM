
import plotly.graph_objects as go
import numpy as np
import os

# This script uses raw Plotly and doesn't rely on location_utils, but we keep it safe.
def debug_nan_rendering():
    fig = go.Figure()
    
    lats1 = [32.0, 32.1, None, 32.2, 32.3]
    lons1 = [34.0, 34.1, None, 34.2, 34.3]
    fig.add_trace(go.Scattermap( # Updated to Scattermap
        mode='lines',
        lat=lats1, lon=lons1,
        line=dict(width=5, color='blue'),
        name='List with None'
    ))
    
    lats2 = np.array([32.0, 32.1, np.nan, 32.2, 32.3]) + 0.5
    lons2 = np.array([34.0, 34.1, np.nan, 34.2, 34.3])
    
    fig.add_trace(go.Scattermap(
        mode='lines',
        lat=lats2, lon=lons2,
        line=dict(width=5, color='red'),
        name='Numpy with NaN'
    ))

    lats3 = (np.array([32.0, 32.1, np.nan, 32.2, 32.3]) + 1.0).tolist()
    lons3 = lons2.tolist()
    
    fig.add_trace(go.Scattermap(
        mode='lines',
        lat=lats3, lon=lons3,
        line=dict(width=5, color='green'),
        name='List with NaN floats'
    ))
    
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center={"lat": 32.5, "lon": 34.2},
            zoom=8
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    fig.write_html(os.path.join(out_dir, "debug_nan_lines.html"))
    print("Saved debug_nan_lines.html")

if __name__ == "__main__":
    debug_nan_rendering()
