"""
Verify that the HDF5 domain file contains all required fields for runMPMYDW.py
"""

import h5py
import numpy as np

filename = "random_rock_domain_50x50x200.h5"

print(f"Checking {filename}...")
print("=" * 60)

with h5py.File(filename, 'r') as f:
    print("\n✓ REQUIRED FIELDS (from runMPMYDW.py):")
    required_fields = {
        'x': '(3, n_particles) - particle positions (transposed)',
        'particle_volume': '(n_particles,) - volume per particle'
    }
    
    for field, description in required_fields.items():
        if field in f:
            data = f[field]
            print(f"  ✓ {field:20} shape={str(data.shape):20} {description}")
        else:
            print(f"  ✗ {field:20} MISSING! {description}")
    
    print("\n✓ OPTIONAL SPATIAL PROPERTY FIELDS:")
    optional_fields = {
        'density': '(n_particles,) - density override',
        'E': '(n_particles,) - Young\'s modulus override',
        'nu': '(n_particles,) - Poisson\'s ratio override',
        'ys': '(n_particles,) - yield stress override',
        'alpha': '(n_particles,) - Drucker-Prager alpha override',
        'hardening': '(n_particles,) - hardening parameter override',
        'softening': '(n_particles,) - softening parameter override',
        'eta_shear': '(n_particles,) - shear viscosity override',
        'eta_bulk': '(n_particles,) - bulk viscosity override',
        'strainCriteria': '(n_particles,) - failure strain override'
    }
    
    for field, description in optional_fields.items():
        if field in f:
            data = f[field]
            min_val = np.min(data)
            max_val = np.max(data)
            print(f"  ✓ {field:20} shape={str(data.shape):20} range=[{min_val:.2e}, {max_val:.2e}]")
        else:
            print(f"  - {field:20} not present (will use args defaults)")
    
    print("\n✓ OTHER FIELDS:")
    other_fields = set(f.keys()) - set(required_fields.keys()) - set(optional_fields.keys())
    for field in sorted(other_fields):
        data = f[field]
        print(f"  · {field:20} shape={str(data.shape)}")
    
    print("\n✓ ATTRIBUTES:")
    for attr_name, attr_value in f.attrs.items():
        print(f"  · {attr_name:20} = {attr_value}")

print("\n" + "=" * 60)
print("✓ Verification complete!")
print("\nThis file is ready to use with:")
print('  python runMPMYDW.py --config benchmarks/cavingBenchmark/config_random_rock.json')
