#!/usr/bin/env python3
"""
Create a minimal dummy tensorflow module to satisfy thingsvision imports
when we only want to use PyTorch models.
"""

import os
import sys
import site

def create_dummy_tensorflow():
    """Create dummy tensorflow module structure."""
    
    # Get user site-packages directory
    user_site = site.getusersitepackages()
    if not user_site:
        # Fallback to manual construction
        user_site = os.path.expanduser(f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    
    tf_path = os.path.join(user_site, "tensorflow")
    
    # Check if real tensorflow exists
    try:
        import tensorflow
        print(f"Real tensorflow already installed (version {tensorflow.__version__})")
        return True
    except ImportError:
        pass
    
    print(f"Creating dummy tensorflow module in {tf_path}")
    
    # Create directory structure
    os.makedirs(tf_path, exist_ok=True)
    os.makedirs(os.path.join(tf_path, "keras"), exist_ok=True)
    
    # Create tensorflow/__init__.py
    with open(os.path.join(tf_path, "__init__.py"), "w") as f:
        f.write("""
__version__ = "2.0.0-dummy"

class keras:
    class applications:
        pass
    class layers:
        pass
    class models:
        pass
""")
    
    # Create tensorflow/keras/__init__.py
    with open(os.path.join(tf_path, "keras", "__init__.py"), "w") as f:
        f.write("""
class applications:
    pass

class layers:
    pass

class models:
    pass
""")
    
    # Create tensorflow/keras/applications.py
    with open(os.path.join(tf_path, "keras", "applications.py"), "w") as f:
        f.write("""
# Dummy tensorflow.keras.applications module
""")
    
    print("Dummy tensorflow module created successfully")
    
    # Test import
    try:
        import tensorflow.keras.applications
        print("✓ tensorflow.keras.applications can be imported")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

if __name__ == "__main__":
    success = create_dummy_tensorflow()
    sys.exit(0 if success else 1)