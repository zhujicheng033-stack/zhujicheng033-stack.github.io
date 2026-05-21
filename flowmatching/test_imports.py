#!/usr/bin/env python3
"""Quick validation that all modules are syntactically correct."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def test_syntax():
    """Just import all modules to check syntax."""
    print("Checking module syntax...")
    
    # This will fail if there are syntax errors
    try:
        import models.components.bezier as m1
        print("  ✓ bezier.py")
        
        import models.components.losses as m2
        print("  ✓ losses.py")
        
        import models.stage1_featuremap as m3
        print("  ✓ stage1_featuremap.py")
        
        import models.stage2_cfm as m4
        print("  ✓ stage2_cfm.py")
        
        import models.stage3_attribution as m5
        print("  ✓ stage3_attribution.py")
        
        import validation.metrics as m6
        print("  ✓ metrics.py")
        
        import data.synthetic_data_generator as m7
        print("  ✓ synthetic_data_generator.py")
        
        print("\n✓ All modules load successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)
