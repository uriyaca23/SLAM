
import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from location_plot_utils import LocationPlotter

class TestLocationPlotterModes(unittest.TestCase):
    def test_modes_structure(self):
        plotter = LocationPlotter(title="Test Modes")
        
        # Add a simple layer
        coords = np.array([[32, 34, 0], [32.1, 34.1, 0]])
        vals = np.array([10, 20])
        cats = np.array(['A', 'B'])
        
        plotter.add_points(
            coords, 
            label="L1", 
            timestep_values=vals,
            categorical_values=cats
        )
        
        plotter._build_plot()
        fig = plotter.fig
        
        # 1. Check Buttons
        updatemenus = fig.layout.updatemenus
        self.assertTrue(len(updatemenus) > 0)
        
        buttons = []
        for menu in updatemenus:
            if 'buttons' in menu:
                buttons.extend(menu['buttons'])
        
        # Look for Name, Timestep, Categorical labels
        labels = [b['label'] for b in buttons]
        self.assertIn('Name', labels)
        self.assertIn('Timestep', labels)
        self.assertIn('Categorical', labels)
        
        # 2. Check Traces
        # Expected: 
        # 1 Static Legend Trace (Name style)
        # 1 Name Trace (hidden or visible)
        # 1 Timestep Trace (hidden or visible)
        # 2 Categorical Traces (One for 'A', one for 'B')
        # Total = 1 + 1 + 1 + 2 = 5 traces
        
        self.assertEqual(len(fig.data), 5)
        
        # 3. Check Visibility Logic (Defaults)
        # Static: Visible
        # Name: Visible
        # Timestep: Hidden
        # Cat: Hidden
        
        self.assertTrue(fig.data[0].visible == True or fig.data[0].visible is None) # Static (might be None which is True default?)
        # Wait, I explicitly set opacity=1 but didn't set 'visible' in _get_static_traces, so default is True.
        
        # Check others
        # Name (idx 1)
        self.assertTrue(fig.data[1].visible) 
        # Timestep (idx 2)
        self.assertFalse(fig.data[2].visible)
        # Cat (idx 3, 4)
        self.assertFalse(fig.data[3].visible)
        self.assertFalse(fig.data[4].visible)

    def test_shared_colorbar(self):
        plotter = LocationPlotter()
        
        # Layer 1: vals 0..10
        plotter.add_points(np.zeros((1,3)), timestep_values=[0, 10], label="L1")
        # Layer 2: vals 100..110
        plotter.add_points(np.zeros((1,3)), timestep_values=[100, 110], label="L2")
        
        plotter._build_plot()
        
        caxis = plotter.fig.layout.coloraxis
        self.assertIsNotNone(caxis)
        self.assertEqual(caxis.cmin, 0)
        self.assertEqual(caxis.cmax, 110)

if __name__ == '__main__':
    unittest.main()
