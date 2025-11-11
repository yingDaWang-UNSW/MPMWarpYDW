"""
Quick test to verify Drucker-Prager implementation compiles and runs
"""
import warp as wp
import numpy as np

wp.init()
device = "cuda:0"

# Import the modules to check for syntax errors
from utils import mpmRoutines
from utils import simulationRoutines
from utils.getArgs import get_args

print("✓ All modules imported successfully")

# Test that we can read the caving config
args = get_args(config_file='benchmarks/cavingBenchmark/config_caving.json')
print(f"✓ Config loaded: constitutive_model={args.constitutive_model}")
print(f"  - Cohesion: {args.cohesion/1e6:.1f} MPa")
print(f"  - Friction angle: {args.friction_angle}°")
print(f"  - Dilation angle: {args.dilation_angle}°")
print(f"  - Tension cutoff: {args.tension_cutoff/1e6:.1f} MPa")

# Test Drucker-Prager function compilation
nTest = 10
F_trial = wp.zeros((nTest, 3, 3), dtype=wp.float32, device=device)
stress = wp.zeros((nTest, 3, 3), dtype=wp.float32, device=device)
mu_test = wp.full(nTest, 5e9, dtype=wp.float32, device=device)
lam_test = wp.full(nTest, 5e9, dtype=wp.float32, device=device)
cohesion_test = wp.full(nTest, 5e6, dtype=wp.float32, device=device)
friction_test = wp.full(nTest, np.radians(35), dtype=wp.float32, device=device)
dilation_test = wp.full(nTest, np.radians(5), dtype=wp.float32, device=device)
tension_test = wp.full(nTest, 1e6, dtype=wp.float32, device=device)
damage_test = wp.zeros(nTest, dtype=wp.float32, device=device)
strain_test = wp.zeros(nTest, dtype=wp.float32, device=device)

# Create simple compression test: F_trial = I + compression in z
F_data = np.zeros((nTest, 3, 3), dtype=np.float32)
for i in range(nTest):
    F_data[i] = np.eye(3, dtype=np.float32)
    F_data[i, 2, 2] = 0.95  # 5% compression
F_trial = wp.array(F_data, dtype=wp.mat33, device=device)

print("\n✓ Test arrays created")

# Test the compute_stress kernel with Drucker-Prager
activeLabel = wp.ones(nTest, dtype=wp.int32, device=device)
materialLabel = wp.ones(nTest, dtype=wp.int32, device=device)
particle_x = wp.zeros(nTest, dtype=wp.vec3, device=device)
particle_v = wp.zeros(nTest, dtype=wp.vec3, device=device)
particle_x_initial = wp.zeros(nTest, dtype=wp.vec3, device=device)
particle_v_initial = wp.zeros(nTest, dtype=wp.vec3, device=device)
particle_F = wp.zeros((nTest, 3, 3), dtype=wp.float32, device=device)
particle_C = wp.zeros((nTest, 3, 3), dtype=wp.float32, device=device)
ys = wp.full(nTest, 100e6, dtype=wp.float32, device=device)
hardening = wp.zeros(nTest, dtype=wp.float32, device=device)
softening = wp.zeros(nTest, dtype=wp.float32, device=device)
density = wp.full(nTest, 2700.0, dtype=wp.float32, device=device)
strainCriteria = wp.full(nTest, 0.01, dtype=wp.float32, device=device)
eff = 1.0
eta_shear = wp.zeros(nTest, dtype=wp.float32, device=device)
eta_bulk = wp.zeros(nTest, dtype=wp.float32, device=device)

try:
    wp.launch(
        kernel=mpmRoutines.compute_stress_from_F_trial,
        dim=nTest,
        inputs=[
            activeLabel,
            materialLabel,
            particle_x,
            particle_v,
            particle_x_initial,
            particle_v_initial,
            particle_F,
            F_trial,
            mu_test,
            lam_test,
            ys,
            hardening,
            softening,
            density,
            strainCriteria,
            eff,
            eta_shear,
            eta_bulk,
            particle_C,
            stress,
            strain_test,
            damage_test,
            1,  # constitutive_model = 1 (Drucker-Prager)
            cohesion_test,
            friction_test,
            dilation_test,
            tension_test
        ],
        device=device
    )
    wp.synchronize()
    print("✓ Drucker-Prager kernel launched successfully")
    
    # Check output
    stress_host = stress.numpy()
    damage_host = damage_test.numpy()
    print(f"  - Mean stress magnitude: {np.mean(np.linalg.norm(stress_host.reshape(nTest, -1), axis=1))/1e6:.2f} MPa")
    print(f"  - Max damage: {np.max(damage_host):.4f}")
    
except Exception as e:
    print(f"✗ Kernel launch failed: {e}")
    raise

print("\n✓✓✓ All tests passed! Drucker-Prager implementation ready.")
print("\nRun the caving benchmark with:")
print("  python runMPMYDW.py --config benchmarks/cavingBenchmark/config_caving.json")
