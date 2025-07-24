import argparse
import json
import os
import pprint
def print_args_in_order(args, parser):
    print("\n===== Loaded Simulation Parameters =====")
    for action in parser._actions:
        if action.dest in vars(args):
            print(f"{action.dest}: {getattr(args, action.dest)}")
    print("=======================================\n")

def get_args():
    # First pass: only parse --config so we can load it before other args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to JSON config file")
    pre_args, _ = pre_parser.parse_known_args()

    # Main parser
    parser = argparse.ArgumentParser(description="MPM-XPBD Simulation Parameters", parents=[pre_parser])

    # Simulation steps & time
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step for MPM (s)")
    parser.add_argument("--dtxpbd", type=float, default=1e-2, help="Time step for XPBD (s)")
    parser.add_argument("--nSteps", type=int, default=2500000, help="Number of simulation steps")

    # Damping & integration
    parser.add_argument("--rpic_damping", type=float, default=0.0, help="Damping for P2G transfer")
    parser.add_argument("--grid_v_damping_scale", type=float, default=1.1, help="Damping factor for grid velocities")
    parser.add_argument("--update_cov", type=int, default=1, help="Update covariance in G2P")

    # Rendering & saving
    parser.add_argument("--render", type=int, default=0, help="Enable rendering")
    parser.add_argument("--saveFlag", type=int, default=0, help="Enable saving simulation results")

    # Domain & grid
    parser.add_argument("--domainFile", type=str, default="./exampleDomains/annular_arch_particles.h5", help="Input HDF5 domain file")
    parser.add_argument("--grid_padding", type=float, default=25.0, help="Padding around min/max bounds (m)")
    parser.add_argument("--grid_particle_spacing_scale", type=float, default=4.0, help="Multiplier for particle diameter to set grid spacing")

    # Material properties
    parser.add_argument("--density", type=float, default=3000.0, help="Density of material (kg/m³)")
    parser.add_argument("--E", type=float, default=1e8, help="Young's modulus (Pa)")
    parser.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio")
    parser.add_argument("--ys", type=float, default=3e6, help="Yield stress (Pa)")
    parser.add_argument("--hardening", type=int, default=0, help="Hardening type")
    parser.add_argument("--xi", type=float, default=10.0, help="Hardening parameter")
    parser.add_argument("--softening", type=float, default=1e7, help="Softening modulus")
    parser.add_argument("--eta_shear", type=float, default=1e7, help="Shear viscosity")
    parser.add_argument("--eta_bulk", type=float, default=1e7, help="Bulk viscosity")

    # Gravity
    parser.add_argument("--gravity", type=float, default=-9.81, help="Gravity (m/s²)")

    # Boundary & friction
    parser.add_argument("--boundFriction", type=float, default=0.0, help="Bounding box friction")
    parser.add_argument("--eff", type=float, default=0.05, help="Phase change efficiency")

    # XPBD parameters
    parser.add_argument("--xpbd_relaxation", type=float, default=1.0, help="XPBD relaxation factor")
    parser.add_argument("--dynamicParticleFriction", type=float, default=0.05, help="Dynamic friction for XPBD")
    parser.add_argument("--staticVelocityThreshold", type=float, default=1e-5, help="Static velocity threshold")
    parser.add_argument("--staticParticleFriction", type=float, default=0.1, help="Static friction for XPBD")
    parser.add_argument("--xpbd_iterations", type=int, default=4, help="Number of XPBD solver iterations")
    parser.add_argument("--particle_cohesion", type=float, default=0.0, help="Cohesion for XPBD particles")
    parser.add_argument("--sleepThreshold", type=float, default=0.5, help="Sleep threshold for XPBD")

    # Swelling
    parser.add_argument("--swellingRatio", type=float, default=0.2, help="Particle swelling ratio")
    parser.add_argument("--swellingActivationFactor", type=float, default=1.0, help="Swelling activation factor")
    parser.add_argument("--swellingMaxFactor", type=float, default=2.0, help="Swelling maximum factor")

    # Velocity limits
    parser.add_argument("--particle_v_max", type=float, default=1000.0, help="Maximum particle velocity")

    # If a config file is provided, load it
    config_data = {}
    if pre_args.config and os.path.exists(pre_args.config):
        with open(pre_args.config, "r") as f:
            config_data = json.load(f)

    # Parse CLI (CLI overrides config file)
    args = parser.parse_args()
    for k, v in config_data.items():
        if getattr(args, k, None) == parser.get_default(k):  # only replace if not overridden
            setattr(args, k, v)

    # Pretty-print final merged config
    print_args_in_order(args, parser)

    return args
