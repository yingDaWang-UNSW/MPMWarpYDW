import argparse
from html import parser
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
    parser.add_argument("--dt", type=float, default=0, help="Time step for MPM (s). Set to 0 to auto-estimate from CFL condition (safety factor 0.3)")
    parser.add_argument("--dtxpbd", type=float, default=1e-2, help="Time step for XPBD (s). dt will be adjusted to ensure dtxpbd/dt is an integer.")
    parser.add_argument("--bigStepDuration", type=float, default=10.0, help="Maximum duration of combined MPM+XPBD phase in seconds")
    parser.add_argument("--xpbdOnlyDuration", type=float, default=0.0, help="Duration of XPBD-only phase after combined phase (seconds). Set to 0 to disable.")
    parser.add_argument("--bigSteps", type=int, default=100, help="Number of big steps (outer loop iterations)")
    parser.add_argument("--residualThreshold", type=float, default=5e-1, help="Residual threshold for convergence")
    parser.add_argument("--damage_stall_threshold", type=float, default=0, help="If mean damage change per step falls below this, terminate combined phase early")
    parser.add_argument("--damage_stall_steps", type=int, default=100, help="Number of consecutive steps below damage_stall_threshold to trigger early termination")
    parser.add_argument("--xpbd_sleep_termination_ratio", type=float, default=1, help="Terminate XPBD-only phase early if this fraction of XPBD particles are asleep (0-1). Set to 1.0 to disable.")

    # Damping & integration
    parser.add_argument("--rpic_damping", type=float, default=0.2, help="Damping for P2G transfer")
    parser.add_argument("--grid_v_damping_scale", type=float, default=1.1, help="Damping factor for grid velocities")
    parser.add_argument("--update_cov", type=int, default=1, help="Update covariance in G2P")

    # Rendering & saving
    parser.add_argument("--render", type=int, default=0, help="Enable OpenGL rendering")
    parser.add_argument("--render_interval", type=float, default=0.1, help="Simulation time interval (seconds) between renders")
    parser.add_argument("--color_mode", type=str, default="effective_ys", help="Color mode for rendering")
    parser.add_argument("--saveFlag", type=int, default=0, help="Enable saving simulation results")
    parser.add_argument("--outputFolder", type=str, default="./output/", help="Output folder for simulation results")
    parser.add_argument("--restart", type=str, default=None, 
                        help="Path to VTP file from which to restart simulation. Must be a bigstep-end save.")
    parser.add_argument("--open_paraview", type=int, default=0, 
                        help="Open ParaView with VTP time series after simulation completes (1=yes, 0=no)")

    # Domain & grid
    parser.add_argument("--domainFile", type=str, default="./exampleDomains/annular_arch_particles.h5", 
                        help="Input HDF5 domain file. Can contain spatial property arrays (density, E, nu, ys, alpha, hardening, softening, eta_shear, eta_bulk, strainCriteria)")
    parser.add_argument("--grid_padding", type=float, default=25.0, help="Padding around min/max bounds (m)")
    parser.add_argument("--grid_particle_spacing_scale", type=float, default=2.0, help="Multiplier for particle diameter to set grid spacing")
    parser.add_argument("--boundary_padding_mask", type=str, default="111101", 
                        help="6-digit binary mask for boundary padding: -X,+X,-Y,+Y,-Z,+Z. 1=loose (grid_padding), 0=tight (mpmPadding). Default '111101'")

    # Material properties (used as defaults if not in HDF5)
    parser.add_argument("--density", type=float, default=3000.0, help="Density of material (kg/m³) - default if not in HDF5")
    parser.add_argument("--E", type=float, default=1e8, help="Young's modulus (Pa) - default if not in HDF5")
    parser.add_argument("--nu", type=float, default=0.2, help="Poisson's ratio - default if not in HDF5")
    
    # Constitutive model selection
    parser.add_argument("--constitutive_model", type=int, default=1, help="1=Drucker-Prager")
    
    # Von Mises parameters
    parser.add_argument("--ys", type=float, default=3e8, help="Yield stress (Pa) - for Von Mises, default if not in HDF5")

    # Drucker-Prager parameters
    parser.add_argument("--alpha", type=float, default=0.3, help="Pressure sensitivity (α) - for Drucker-Prager, default if not in HDF5")

    # Hardening/softening (both models)
    parser.add_argument("--hardening", type=float, default=0, help="Hardening parameter - default if not in HDF5")
    parser.add_argument("--softening", type=float, default=0, help="Softening modulus - default if not in HDF5")
    parser.add_argument("--eta_shear", type=float, default=1e5, help="Shear viscosity - default if not in HDF5")
    parser.add_argument("--eta_bulk", type=float, default=1e5, help="Bulk viscosity - default if not in HDF5")
    
    # Volumetric locking correction
    parser.add_argument("--volumetric_locking_correction", type=int, default=1, 
                        help="Enable cell-averaged volumetric locking correction (1=on, 0=off). Prevents artificial stiffening for near-incompressible materials.")

    # Gravity
    parser.add_argument("--gravity", type=float, default=-9.81, help="Gravity (m/s²)")
    parser.add_argument("--K0", type=float, default=0.5, help="Lateral earth pressure coefficient for initial stress")
    parser.add_argument("--z_top", type=float, default=None, help="Reference height for geostatic stress (m). If None, uses max particle z-coordinate")
    parser.add_argument("--initialise_geostatic", type=int, default=1, help="initialise geo or let settle")
    
    # Seismic/gravity pulse parameters (to trigger damage propagation)
    parser.add_argument("--gravity_pulse_factor", type=float, default=1.0, 
                        help="Gravity multiplier during pulse phase at start of each big step (e.g., 2.0 = 2x gravity). Set to 1.0 to disable.")
    parser.add_argument("--gravity_pulse_duration", type=float, default=0.0, 
                        help="Duration in seconds to apply gravity pulse at start of each big step. Set to 0 to disable.")

    # Boundary & friction
    parser.add_argument("--boundFriction", type=float, default=0.2, help="Bounding box friction coefficient")
    parser.add_argument("--boundaryCondition", type=str, default="friction", choices=["friction", "friction_gradual", "restitution", "absorbing"],
                        help="Boundary condition type: 'friction' (Coulomb, abrupt), 'friction_gradual' (Coulomb with smooth transition), 'restitution' (elastic bounce), 'absorbing' (damping)")
    parser.add_argument("--boundRestitution", type=float, default=0, help="Coefficient of restitution for boundary collisions (0=inelastic, 1=elastic)")
    parser.add_argument("--eff", type=float, default=0.05, help="Phase change efficiency")
    parser.add_argument("--strainCriteria", type=float, default=0.05, help="Critical accumulated strain for phase change - default if not in HDF5")

    # XPBD parameters
    parser.add_argument("--xpbd_relaxation", type=float, default=1.0, help="XPBD relaxation factor")
    parser.add_argument("--xpbd_mpm_coupling_strength", type=float, default=0.25, 
                        help="MPM-XPBD coupling strength: scales XPBD displacement to MPM velocity impulse (empirically tuned, typically 0.2-0.3)")
    parser.add_argument("--xpbd_deactivation_z_datum", type=float, default=None, 
                        help="Z coordinate below which XPBD particles are deactivated and teleported. If None, uses minBounds[2] (bottom of domain).")
    parser.add_argument("--mpm_contact_transition_lock", type=int, default=0,
                        help="If 1, MPM particles in contact with XPBD particles cannot transition (materialLabel=0). If 0, all MPM particles can transition.")
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

    # Parse CLI arguments first to detect which were explicitly provided
    # Use parse_known_args to handle unknown keys in JSON gracefully
    args, unknown = parser.parse_known_args()
    
    # Track which arguments were explicitly set on command line
    # by comparing sys.argv to see what was actually provided
    import sys
    cli_args_set = set()
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            # Extract the argument name (remove leading --)
            arg_name = arg.lstrip('-').replace('-', '_')
            cli_args_set.add(arg_name)
    
    # Apply config file values only if NOT overridden by CLI
    for k, v in config_data.items():
        if k not in cli_args_set and hasattr(args, k):
            setattr(args, k, v)

    # Pretty-print final merged config
    print_args_in_order(args, parser)

    return args
