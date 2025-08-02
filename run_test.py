#!/usr/bin/env python3

import subprocess
import sys

def run_test():
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/performance/test_performance_integration.py::TestPerformanceIntegration::test_alert_handling', 
            '-v', '--tb=short', '--no-cov'
        ], capture_output=True, text=True, timeout=60)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Test timed out")
        return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")