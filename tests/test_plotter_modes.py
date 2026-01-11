
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
        # 1 Name Trace (Visible)
        # 1 Timestep Trace (Hidden)
        # 2 Categorical Traces (Hidden)
        # Total = 1 + 1 + 2 = 4 traces
        
        self.assertEqual(len(fig.data), 4)
        
        # 3. Check Visibility Logic (Defaults)
        # Name (idx 0): Visible
        self.assertTrue(fig.data[0].visible) 
        # Timestep (idx 1): Hidden
        self.assertFalse(fig.data[1].visible)
        # Cat (idx 2, 3): Hidden
        self.assertFalse(fig.data[2].visible)
        self.assertFalse(fig.data[3].visible)

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

    def test_buttons_visibility(self):
        # Case 1: Only Name data
        plotter = LocationPlotter(title="Name Only")
        plotter.add_points(np.zeros((3,3)), label="L1") # No timestep, no cat
        plotter._build_plot()
        
        # Check buttons
        updatemenus = plotter.fig.layout.updatemenus
        buttons = []
        if updatemenus:
            for menu in updatemenus:
                if 'buttons' in menu: buttons.extend(menu['buttons'])
        
        labels = [b['label'] for b in buttons]
        # Should contain Name, but NOT Timestep or Categorical
        self.assertIn('Name', labels)
        self.assertNotIn('Timestep', labels)
        self.assertNotIn('Categorical', labels)
        
        # Case 2: Validation of Timestep
        plotter2 = LocationPlotter(title="Time Only")
        plotter2.add_points(np.zeros((3,3)), timestep_values=[1,2,3], label="L1")
        plotter2._build_plot()
        
        buttons2 = []
        if plotter2.fig.layout.updatemenus:
             for menu in plotter2.fig.layout.updatemenus:
                  if 'buttons' in menu: buttons2.extend(menu['buttons'])
        labels2 = [b['label'] for b in buttons2]
        self.assertIn('Name', labels2)
        self.assertIn('Timestep', labels2)
        self.assertNotIn('Categorical', labels2)

if __name__ == '__main__':
    unittest.main()
