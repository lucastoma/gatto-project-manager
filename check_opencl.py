import sys
import traceback
from pathlib import Path

print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Current working directory: {Path.cwd()}")

try:
    import pyopencl as cl
    print("\nPyOpenCL is installed and importable")
    print(f"PyOpenCL version: {cl.VERSION_TEXT}")
    
    print("\nAvailable platforms and devices:")
    platforms = cl.get_platforms()
    if not platforms:
        print("  No OpenCL platforms found!")
    else:
        for i, platform in enumerate(platforms):
            print(f"\nPlatform {i}: {platform.name}")
            print(f"  Vendor: {platform.vendor}")
            print(f"  Version: {platform.version}")
            devices = platform.get_devices()
            print(f"  Devices: {len(devices)}")
            for j, device in enumerate(devices):
                print(f"    Device {j}: {device.name}")
                print(f"      Type: {cl.device_type.to_string(device.type)}")
                print(f"      Version: {device.version}")
                print(f"      Available: {'Yes' if device.available else 'No'}")
    
    # Check kernel file
    kernel_path = Path("app/algorithms/algorithm_01_palette/palette_mapping.cl")
    print(f"\nChecking kernel file at: {kernel_path.absolute()}")
    if kernel_path.exists():
        print("  Kernel file exists!")
        print(f"  Size: {kernel_path.stat().st_size} bytes")
    else:
        print("  Kernel file NOT FOUND!")
        
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nTraceback:")
    traceback.print_exc()