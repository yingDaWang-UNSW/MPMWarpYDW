"""
Test script for USD renderer functionality.
Verifies that the USD renderer can be initialized and used.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_usd_import():
    """Test if USD Python bindings are available."""
    print("Testing USD Python bindings...")
    try:
        from pxr import Usd, UsdGeom
        print("  ✓ USD bindings available (full functionality)")
        return True
    except ImportError:
        print("  ⚠ USD bindings not available (will use fallback mode)")
        return False

def test_renderer_initialization():
    """Test if WarpUSDRenderer can be initialized."""
    print("\nTesting USD renderer initialization...")
    try:
        from utils.usdRenderer import WarpUSDRenderer
        
        output_path = "./test_output/usd_test"
        renderer = WarpUSDRenderer(output_path=output_path, fps=60, up_axis="Z")
        print(f"  ✓ Renderer initialized: {renderer.output_path}")
        return renderer
    except Exception as e:
        print(f"  ✗ Failed to initialize renderer: {e}")
        return None

def test_render_frame(renderer):
    """Test rendering a simple frame."""
    print("\nTesting frame rendering...")
    try:
        import warp as wp
        wp.init()
        
        # Create simple test data
        n_particles = 100
        positions = np.random.rand(n_particles, 3).astype(np.float32) * 10
        radii = np.ones(n_particles, dtype=np.float32) * 0.1
        colors = np.random.rand(n_particles, 3).astype(np.float32)
        velocities = np.random.rand(n_particles, 3).astype(np.float32) * 0.1
        
        # Render frame
        renderer.render_frame(
            time_code=0.0,
            particle_positions=positions,
            particle_radii=radii,
            particle_colors=colors,
            particle_velocities=velocities,
            additional_fields={
                "test_scalar": np.random.rand(n_particles).astype(np.float32),
                "test_vector": np.random.rand(n_particles, 3).astype(np.float32)
            }
        )
        print("  ✓ Frame rendered successfully")
        
        # Render a few more frames
        for i in range(1, 5):
            positions += velocities * 0.1
            renderer.render_frame(
                time_code=float(i),
                particle_positions=positions,
                particle_radii=radii,
                particle_colors=colors,
                particle_velocities=velocities
            )
        
        print(f"  ✓ Rendered {renderer.frame_count} total frames")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to render frame: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_finalization(renderer):
    """Test renderer finalization."""
    print("\nTesting renderer finalization...")
    try:
        renderer.finalize()
        print("  ✓ Renderer finalized")
        
        # Check if output file exists
        if renderer.usd_available:
            usd_file = renderer.output_path / "simulation.usd"
            if usd_file.exists():
                print(f"  ✓ USD file created: {usd_file}")
                print(f"     Size: {usd_file.stat().st_size / 1024:.2f} KB")
            else:
                print(f"  ⚠ USD file not found at: {usd_file}")
        else:
            # Check for frame files in fallback mode
            frame_files = list(renderer.output_path.glob("frame_*.usd"))
            if frame_files:
                print(f"  ✓ Created {len(frame_files)} frame files (fallback mode)")
            else:
                print(f"  ⚠ No frame files found")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to finalize: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("USD Renderer Test Suite")
    print("="*60)
    
    # Test USD availability
    has_usd = test_usd_import()
    
    # Test renderer initialization
    renderer = test_renderer_initialization()
    if renderer is None:
        print("\n" + "="*60)
        print("FAILED: Could not initialize renderer")
        print("="*60)
        return False
    
    # Test rendering
    if not test_render_frame(renderer):
        print("\n" + "="*60)
        print("FAILED: Could not render frames")
        print("="*60)
        return False
    
    # Test finalization
    if not test_finalization(renderer):
        print("\n" + "="*60)
        print("FAILED: Could not finalize renderer")
        print("="*60)
        return False
    
    # Success
    print("\n" + "="*60)
    print("SUCCESS: All tests passed!")
    print("="*60)
    
    if has_usd:
        print("\nTo view the test output, run:")
        print(f"  usdview {renderer.output_path}/simulation.usd")
    else:
        print("\nRenderer is working in fallback mode.")
        print("Install USD bindings for better performance:")
        print("  pip install usd-core")
    
    print("\nTo use in simulation, set in config JSON:")
    print('  "render": 1,')
    print('  "render_backend": "usd"')
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
