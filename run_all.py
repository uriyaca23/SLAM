
import unittest
import os
import sys
import subprocess
import time

def run_unit_tests():
    print("="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_viz_script(script_path):
    print(f"Running {script_path}...")
    try:
        # Run in separate process to ensure clean environment
        cmd = [sys.executable, script_path]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"  [PASS] {os.path.basename(script_path)}")
        return True
    except subprocess.CalledProcessError:
        print(f"  [FAIL] {os.path.basename(script_path)}")
        return False
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(script_path)}: {e}")
        return False

def run_visualizations():
    print("\n" + "="*60)
    print("RUNNING VISUALIZATION SCRIPTS")
    print("="*60)
    
    # Scripts to run
    scripts = [
        'tests/demo_gsdc_plot.py',
        'tests/plot_real_gsdc.py',
        'tests/test_real_gsdc_ecef.py',
        'tests/debug_slider_date.py',
        'tests/test_satellite_map.py',
        'tests/repro_modes.py',
        'tests/debug_lines_color.py',
        'tests/test_stress_plot_utils.py'
    ]
    
    root_dir = os.path.dirname(__file__)
    success = True
    
    for relative_path in scripts:
        full_path = os.path.join(root_dir, relative_path)
        if not run_viz_script(full_path):
            success = False
            
    return success

def main():
    start_time = time.time()
    
    print("Starting Comprehensive Test Suite...")
    
    # 1. Run Unit Tests (includes functional checks like test_plot_utils.py)
    unit_ok = run_unit_tests()
    
    # 2. Run Visualization Scripts (that generate HTMLs)
    # Note: test_plot_utils.py is redundant here if it's a unittest, but scripts like demo_gsdc_plot.py are not.
    viz_ok = run_visualizations()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Time: {duration:.2f}s")
    print(f"Unit Tests: {'PASS' if unit_ok else 'FAIL'}")
    print(f"Visualizations: {'PASS' if viz_ok else 'FAIL'}")
    
    if unit_ok and viz_ok:
        print("\nALL CHECKS PASSED. Code is ready.")
        sys.exit(0)
    else:
        print("\nSOME CHECKS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
