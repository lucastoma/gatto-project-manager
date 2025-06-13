"""
Helper script to regenerate test images in the correct format.
"""
import numpy as np
import os

def create_test_images():
    """Create test images and save them as .npy files."""
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Sample 1: Gradient image
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    sample1 = np.stack([X, Y, 100 - X], axis=-1)  # LAB-like values
    
    # Sample 2: Random noise with different distribution
    np.random.seed(42)
    sample2 = np.random.normal(loc=50, scale=30, size=(100, 100, 3)).clip(0, 100)
    
    # Save as numpy arrays without pickling
    np.save('test_images/sample1.npy', sample1, allow_pickle=False)
    np.save('test_images/sample2.npy', sample2, allow_pickle=False)
    
    print("Test images regenerated successfully!")

if __name__ == "__main__":
    create_test_images()
