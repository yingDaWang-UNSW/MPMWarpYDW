import numpy as np
from scipy.spatial.transform import Rotation
from pyglet.math import Vec3 as PyVec3
import matplotlib.pyplot as plt
def values_to_rgb(values, min_val=2000, max_val=4000):
    normalized_values = (values - min_val) / (max_val - min_val)
    normalized_values = np.clip(normalized_values, 0, 1)
    rgb_colors = plt.cm.jet(normalized_values)[:, :3]  
    return rgb_colors
def look_at_centroid(particles, renderer, fov_degrees=60):
    centroid = np.mean(particles[:, :3], axis=0)
    distances = np.linalg.norm(particles[:, :3] - centroid, axis=1)
    max_distance = np.max(distances)
    fov_radians = np.radians(fov_degrees)
    camera_distance = max_distance / np.sin(fov_radians / 2)
    camera_pos = centroid + np.array([camera_distance, camera_distance, camera_distance])
    renderer._camera_pos = PyVec3(camera_pos[0], camera_pos[1], camera_pos[2])
    camera_direction = centroid - camera_pos
    camera_direction /= np.linalg.norm(camera_direction)  
    renderer._camera_front = PyVec3(camera_direction[0], camera_direction[1], camera_direction[2])
    renderer._yaw = np.arctan2(camera_direction[0], camera_direction[2]) * 180 / np.pi
    renderer._pitch = -np.arcsin(camera_direction[1]) * 180 / np.pi
    return renderer
def look_at_centroid_z_up(particles, renderer, fov_degrees=60):
    import numpy as np

    # Compute centroid and spread
    centroid = np.mean(particles[:, :3], axis=0)
    distances = np.linalg.norm(particles[:, :3] - centroid, axis=1)
    max_distance = np.max(distances)

    # FOV and distance to frame the object
    fov_radians = np.radians(fov_degrees)
    camera_distance = max_distance / np.sin(fov_radians / 2)

    # Place camera in +Y direction (XY plane), Z is up
    camera_pos = centroid + np.array([0, camera_distance, 0])
    camera_direction = centroid - camera_pos
    camera_direction /= np.linalg.norm(camera_direction)

    # Set camera
    renderer._camera_pos = PyVec3(*camera_pos)
    renderer._camera_front = PyVec3(*camera_direction)

    # Compute yaw (in XY plane) and pitch (vertical)
    dx, dy, dz = camera_direction

    # Yaw: rotation around Z axis (in XY plane)
    renderer._yaw = np.degrees(np.arctan2(dx, dy))  # yaw = arctan(x / y)

    # Pitch: rotation from XY plane toward Z
    renderer._pitch = np.degrees(np.arcsin(dz))  # pitch = arcsin(z)

    return renderer



def transform_nodes(nodes, body_coordinates):
    position = body_coordinates[0][:3]
    quaternion = body_coordinates[0][3:]
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    homogeneous_nodes = np.hstack((nodes, np.ones((nodes.shape[0], 1))))
    transformed_nodes = homogeneous_nodes @ transformation_matrix.T
    return transformed_nodes[:, :3]

from mpl_toolkits.mplot3d import Axes3D  # Although not directly used, this import is necessary for 3D plotting.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d_box_and_particles(center, half_edges, particles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Unpack the center and half-edge lengths
    center_x, center_y, center_z = center
    half_width, half_height, half_depth = half_edges
    
    # Generate the vertices of the box
    box_verts = np.array([[center_x - half_width, center_y - half_height, center_z - half_depth],
                          [center_x + half_width, center_y - half_height, center_z - half_depth],
                          [center_x + half_width, center_y + half_height, center_z - half_depth],
                          [center_x - half_width, center_y + half_height, center_z - half_depth],
                          [center_x - half_width, center_y - half_height, center_z + half_depth],
                          [center_x + half_width, center_y - half_height, center_z + half_depth],
                          [center_x + half_width, center_y + half_height, center_z + half_depth],
                          [center_x - half_width, center_y + half_height, center_z + half_depth]])

    # List of sides' polygons
    verts = [[box_verts[0], box_verts[1], box_verts[2], box_verts[3]],
             [box_verts[4], box_verts[5], box_verts[6], box_verts[7]], 
             [box_verts[0], box_verts[1], box_verts[5], box_verts[4]], 
             [box_verts[2], box_verts[3], box_verts[7], box_verts[6]], 
             [box_verts[1], box_verts[2], box_verts[6], box_verts[5]],
             [box_verts[4], box_verts[7], box_verts[3], box_verts[0]]]

    # Plot sides
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    
    # Plot particles
    for particle in particles:
        ax.scatter(*particle, color='b')

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Box and Particles')

    # Show the plot
    plt.show()
import h5py
import numpy as np
from lxml import etree
import os

def save_particles_and_grid_to_h5_xdmf(
    filename_prefix,
    particle_positions,        # shape (N, 3), float32
    particle_field,            # shape (N,), float32 or (N,3)
    particle_field_name,
    grid_field,                # shape (nx,ny,nz) or (nx,ny,nz,3), float32
    grid_origin,               # (x0, y0, z0)
    grid_spacing               # dx
):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

    h5_filename = f"{filename_prefix}.h5"
    xdmf_filename = f"{filename_prefix}.xdmf"

    with h5py.File(h5_filename, "w") as h5f:
        # Particle data
        h5f.create_dataset("Particle/Position", data=particle_positions.astype(np.float32))
        h5f.create_dataset(f"Particle/{particle_field_name}", data=particle_field.astype(np.float32))

        # Grid data
        h5f.create_dataset("Grid/Data", data=grid_field.astype(np.float32))

    # XDMF
    root = etree.Element("Xdmf", Version="3.0")
    domain = etree.SubElement(root, "Domain")

    ######################
    # Particle Grid Block
    ######################
    grid_particles = etree.SubElement(domain, "Grid", Name="Particles", GridType="Uniform")

    geometry = etree.SubElement(grid_particles, "Geometry", GeometryType="XYZ")
    data_item = etree.SubElement(geometry, "DataItem",
                                  Dimensions=f"{len(particle_positions)} 3",
                                  NumberType="Float",
                                  Precision="4",
                                  Format="HDF")
    data_item.text = f"{os.path.basename(h5_filename)}:/Particle/Position"

    topology = etree.SubElement(grid_particles, "Topology", TopologyType="Polyvertex", NumberOfElements=str(len(particle_positions)))

    # Particle field
    attr = etree.SubElement(grid_particles, "Attribute", Name=particle_field_name,
                            AttributeType="Scalar" if particle_field.ndim == 1 else "Vector", Center="Node")
    data_item = etree.SubElement(attr, "DataItem",
                                  Dimensions=f"{len(particle_positions)} {1 if particle_field.ndim == 1 else 3}",
                                  NumberType="Float",
                                  Precision="4",
                                  Format="HDF")
    data_item.text = f"{os.path.basename(h5_filename)}:/Particle/{particle_field_name}"

    ######################
    # Grid Block
    ######################
    grid_shape = grid_field.shape[:3]
    grid_ncomp = grid_field.shape[3] if grid_field.ndim == 4 else 1

    grid_struct = etree.SubElement(domain, "Grid", Name="Grid", GridType="Uniform")

    topology = etree.SubElement(grid_struct, "Topology", TopologyType="3DRectMesh",
                                 Dimensions=f"{grid_shape[2]} {grid_shape[1]} {grid_shape[0]}")  # ZYX ordering

    geometry = etree.SubElement(grid_struct, "Geometry", GeometryType="ORIGIN_DXDYDZ")
    origin_elem = etree.SubElement(geometry, "DataItem", Dimensions="3", NumberType="Float", Format="XML")
    origin_elem.text = f"{grid_origin[0]} {grid_origin[1]} {grid_origin[2]}"
    spacing_elem = etree.SubElement(geometry, "DataItem", Dimensions="3", NumberType="Float", Format="XML")
    spacing_elem.text = f"{grid_spacing} {grid_spacing} {grid_spacing}"

    attr = etree.SubElement(grid_struct, "Attribute", Name="grid_field",
                            AttributeType="Scalar" if grid_ncomp == 1 else "Vector", Center="Cell")
    data_item = etree.SubElement(attr, "DataItem",
                                  Dimensions=f"{grid_shape[2]} {grid_shape[1]} {grid_shape[0]} {grid_ncomp}",
                                  NumberType="Float",
                                  Precision="4",
                                  Format="HDF")
    data_item.text = f"{os.path.basename(h5_filename)}:/Grid/Data"

    # Write XDMF
    with open(xdmf_filename, "wb") as f:
        f.write(etree.tostring(root, pretty_print=True))


def check_grid_particle_alignment(grid, dx, minBounds, particle_positions, threshold=1e-8):
    """
    grid: ndarray of shape (nx, ny, nz) or (nx, ny, nz, 3)
    dx: grid spacing (scalar)
    minBounds: (3,) array-like, origin of the grid
    particle_positions: (N, 3) ndarray of particle positions
    threshold: minimum value to consider a grid cell "non-zero"
    """

    grid = np.asarray(grid)
    particle_positions = np.asarray(particle_positions)
    minBounds = np.asarray(minBounds)

    if grid.ndim == 4:
        grid_mag = np.linalg.norm(grid, axis=-1)
    else:
        grid_mag = grid

    nonzero_idx = np.argwhere(grid_mag > threshold)
    nonzero_world = nonzero_idx * dx + minBounds  # convert to world coordinates

    # Convert particle positions to grid indices
    particle_grid_idx = np.floor((particle_positions - minBounds) / dx).astype(int)

    # Unique grid cells touched by particles
    particle_cells = set(map(tuple, particle_grid_idx))

    # Unique non-zero grid indices
    grid_cells = set(map(tuple, nonzero_idx))

    # Intersection
    matched = grid_cells & particle_cells
    missed = grid_cells - particle_cells
    excess = particle_cells - grid_cells

    print(f"Non-zero grid cells: {len(grid_cells)}")
    print(f"Matched particle grid cells: {len(matched)}")
    print(f"Grid cells with values but no nearby particles: {len(missed)}")
    print(f"Particle locations with no corresponding grid activity: {len(excess)}")

    # return {
    #     "matched_cells": matched,
    #     "missed_cells": missed,
    #     "excess_particle_cells": excess,
    # }
import numpy as np
import vtk
from vtk.util import numpy_support
import os


def load_restart_from_vtp(vtp_path):
    """
    Load restart data from a VTP file saved at end of bigstep.
    
    Parameters
    ----------
    vtp_path : str
        Path to the VTP file containing restart data
    
    Returns
    -------
    dict
        Dictionary containing all restart data needed to continue simulation:
        - particle_x: (N, 3) positions
        - particle_v: (N, 3) velocities
        - particle_radius: (N,) radii
        - particle_F: (N, 3, 3) deformation gradient
        - particle_C: (N, 3, 3) APIC affine matrix
        - particle_stress: (N, 3, 3) stress tensor
        - particle_damage: (N,) damage values
        - particle_accumulated_strain: (N,) accumulated plastic strain
        - materialLabel: (N,) material labels (0,1=MPM, 2=XPBD)
        - activeLabel: (N,) active labels
        - ys: (N,) current yield stress
        - ys_base: (N,) base yield stress
        - mu, lam, bulk, alpha, hardening, softening, eta_shear, eta_bulk, strainCriteria: (N,) material properties
        - particle_vol, particle_mass, particle_density: (N,) physical properties
        - particle_x_initial_xpbd: (N, 3) XPBD initial positions
        - particle_cumDist_xpbd: (N,) XPBD cumulative distance
        - sim_time: float, simulation time
        - bigStep: int, completed bigstep number
        - dx: float, grid spacing
    """
    print(f"\n{'='*60}")
    print(f"LOADING RESTART DATA FROM: {vtp_path}")
    print(f"{'='*60}")
    
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"Failed to read VTP file or file is empty: {vtp_path}")
    
    nPoints = polydata.GetNumberOfPoints()
    print(f"  Number of particles: {nPoints}")
    
    restart_data = {}
    
    # === Load particle positions ===
    points = polydata.GetPoints()
    positions = np.zeros((nPoints, 3), dtype=np.float32)
    for i in range(nPoints):
        positions[i] = points.GetPoint(i)
    restart_data['particle_x'] = positions
    
    # === Load point data arrays ===
    point_data = polydata.GetPointData()
    
    def get_array(name, components=1, required=True):
        arr = point_data.GetArray(name)
        if arr is None:
            if required:
                raise ValueError(f"Required array '{name}' not found in VTP file")
            return None
        data = numpy_support.vtk_to_numpy(arr)
        if components > 1 and data.ndim == 1:
            data = data.reshape(-1, components)
        return data
    
    # Basic visualization arrays (always present)
    restart_data['particle_radius'] = get_array("radius")
    restart_data['particle_v'] = get_array("velocity", components=3, required=False)
    if restart_data['particle_v'] is None:
        restart_data['particle_v'] = np.zeros((nPoints, 3), dtype=np.float32)
    
    restart_data['ys'] = get_array("yield_stress", required=False)
    restart_data['particle_damage'] = get_array("damage", required=False)
    restart_data['materialLabel'] = get_array("material_label", required=False)
    restart_data['activeLabel'] = get_array("active_label", required=False)
    
    # Stress tensor (6 symmetric components → 3x3)
    stress_6 = get_array("stress_tensor", components=6, required=False)
    if stress_6 is not None:
        restart_data['particle_stress'] = np.zeros((nPoints, 3, 3), dtype=np.float32)
        restart_data['particle_stress'][:, 0, 0] = stress_6[:, 0]  # xx
        restart_data['particle_stress'][:, 1, 1] = stress_6[:, 1]  # yy
        restart_data['particle_stress'][:, 2, 2] = stress_6[:, 2]  # zz
        restart_data['particle_stress'][:, 0, 1] = stress_6[:, 3]  # xy
        restart_data['particle_stress'][:, 1, 0] = stress_6[:, 3]  # xy (symmetric)
        restart_data['particle_stress'][:, 0, 2] = stress_6[:, 4]  # xz
        restart_data['particle_stress'][:, 2, 0] = stress_6[:, 4]  # xz (symmetric)
        restart_data['particle_stress'][:, 1, 2] = stress_6[:, 5]  # yz
        restart_data['particle_stress'][:, 2, 1] = stress_6[:, 5]  # yz (symmetric)
    
    # === Restart-specific arrays (only in checkpoint files) ===
    # Deformation gradient F (9 components → 3x3)
    F_flat = get_array("deformation_gradient_F", components=9, required=False)
    if F_flat is not None:
        restart_data['particle_F'] = F_flat.reshape(-1, 3, 3)
        print(f"  Loaded deformation gradient F")
    else:
        print(f"  WARNING: No deformation gradient F found - will initialize to identity")
        restart_data['particle_F'] = np.tile(np.eye(3, dtype=np.float32), (nPoints, 1, 1))
    
    # APIC affine matrix C
    C_flat = get_array("apic_C", components=9, required=False)
    if C_flat is not None:
        restart_data['particle_C'] = C_flat.reshape(-1, 3, 3)
        print(f"  Loaded APIC affine matrix C")
    else:
        restart_data['particle_C'] = np.zeros((nPoints, 3, 3), dtype=np.float32)
    
    # Accumulated strain
    restart_data['particle_accumulated_strain'] = get_array("accumulated_strain", required=False)
    if restart_data['particle_accumulated_strain'] is None:
        restart_data['particle_accumulated_strain'] = np.zeros(nPoints, dtype=np.float32)
    
    # Base yield stress
    restart_data['ys_base'] = get_array("yield_stress_base", required=False)
    if restart_data['ys_base'] is None and restart_data['ys'] is not None:
        restart_data['ys_base'] = restart_data['ys'].copy()
    
    # Material properties
    for prop_name in ['mu', 'lam', 'bulk', 'alpha', 'hardening', 'softening', 
                      'eta_shear', 'eta_bulk', 'strainCriteria']:
        arr = get_array(f"mpm_{prop_name}", required=False)
        if arr is not None:
            restart_data[prop_name] = arr
            print(f"  Loaded {prop_name}")
    
    # Physical properties
    restart_data['particle_vol'] = get_array("particle_volume", required=False)
    restart_data['particle_mass'] = get_array("particle_mass", required=False)
    restart_data['particle_density'] = get_array("particle_density", required=False)
    
    # XPBD state
    xpbd_init = get_array("xpbd_initial_position", components=3, required=False)
    if xpbd_init is not None:
        restart_data['particle_x_initial_xpbd'] = xpbd_init
        print(f"  Loaded XPBD initial positions")
    
    restart_data['particle_cumDist_xpbd'] = get_array("xpbd_cumulative_distance", required=False)
    
    # === Load field data (metadata) ===
    field_data = polydata.GetFieldData()
    
    def get_field_scalar(name, default=None):
        arr = field_data.GetArray(name)
        if arr is None:
            return default
        return arr.GetValue(0)
    
    restart_data['sim_time'] = get_field_scalar("sim_time", 0.0)
    restart_data['bigStep'] = int(get_field_scalar("bigStep", 0))
    restart_data['dx'] = get_field_scalar("dx", None)
    
    print(f"\n  Restart state:")
    print(f"    Simulation time: {restart_data['sim_time']:.4f}s")
    print(f"    Completed bigStep: {restart_data['bigStep']}")
    if restart_data['dx'] is not None:
        print(f"    Grid spacing (dx): {restart_data['dx']:.4f}m")
    
    # Count material states
    if restart_data['materialLabel'] is not None:
        n_mpm_locked = np.sum(restart_data['materialLabel'] == 0)
        n_mpm_trans = np.sum(restart_data['materialLabel'] == 1)
        n_xpbd = np.sum(restart_data['materialLabel'] == 2)
        print(f"    MPM (locked): {n_mpm_locked}, MPM (transitionable): {n_mpm_trans}, XPBD: {n_xpbd}")
    
    if restart_data['particle_damage'] is not None:
        mean_damage = np.mean(restart_data['particle_damage'])
        max_damage = np.max(restart_data['particle_damage'])
        print(f"    Mean damage: {mean_damage:.6f}, Max damage: {max_damage:.6f}")
    
    print(f"{'='*60}\n")
    
    return restart_data

def save_grid_and_particles_vti_vtp(
    output_prefix,
    grid_mass,           # shape (nx, ny, nz)
    minBounds,           # (x0, y0, z0)
    dx,                  # scalar spacing
    particle_positions,  # shape (N, 3)
    particle_radius=0.5, # optional radius field or scalar
    particle_velocity=None,  # shape (N, 3)
    particle_ys=None,    # shape (N,)
    particle_effective_ys=None,  # shape (N,) - for Drucker-Prager: ys - alpha*p
    particle_damage=None, # shape (N,)
    particle_mean_stress=None,  # shape (N,)
    particle_von_mises=None,     # shape (N,)
    particle_stress_tensor=None,  # shape (N, 3, 3) or (N, 9)
    particle_active_label=None,   # shape (N,) - active/inactive flag
    particle_material_label=None,  # shape (N,) - 0=MPM(locked), 1=MPM(transitionable), 2=XPBD
    # === Restart fields (only saved for bigstep-end checkpoints) ===
    restart_data=None,  # dict with all restart fields, if None only saves visualization data
):
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    ### Save grid as .vti
    nx, ny, nz = grid_mass.shape
    grid = vtk.vtkImageData()
    grid.SetDimensions(nx, ny, nz)
    grid.SetSpacing(dx, dx, dx)
    grid.SetOrigin(*minBounds)

    mass_flat = grid_mass.ravel(order='F')  # VTK expects Fortran order
    mass_vtk = numpy_support.numpy_to_vtk(mass_flat.astype(np.float32))
    mass_vtk.SetName("grid_mass")

    grid.GetPointData().SetScalars(mass_vtk)

    writer_vti = vtk.vtkXMLImageDataWriter()
    writer_vti.SetFileName(f"{output_prefix}_grid.vti")
    writer_vti.SetInputData(grid)
    writer_vti.Write()

    ### Save particles as .vtp
    points = vtk.vtkPoints()
    positions = particle_positions.astype(np.float32)
    for pos in positions:
        points.InsertNextPoint(pos.tolist())

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Add radius
    if np.isscalar(particle_radius):
        radius_array = np.full((positions.shape[0],), particle_radius, dtype=np.float32)
    else:
        radius_array = particle_radius.astype(np.float32)
    vtk_radius = numpy_support.numpy_to_vtk(radius_array)
    vtk_radius.SetName("radius")
    polydata.GetPointData().AddArray(vtk_radius)

    # Add velocity (vector)
    if particle_velocity is not None:
        vtk_velocity = numpy_support.numpy_to_vtk(particle_velocity.astype(np.float32))
        vtk_velocity.SetName("velocity")
        vtk_velocity.SetNumberOfComponents(3)
        polydata.GetPointData().AddArray(vtk_velocity)

    # Add yield stress (scalar)
    if particle_ys is not None:
        vtk_ys = numpy_support.numpy_to_vtk(particle_ys.astype(np.float32))
        vtk_ys.SetName("yield_stress")
        polydata.GetPointData().AddArray(vtk_ys)

    # Add effective yield stress (scalar) - for Drucker-Prager: (ys - alpha*p)*(1-D)
    if particle_effective_ys is not None:
        vtk_effective_ys = numpy_support.numpy_to_vtk(particle_effective_ys.astype(np.float32))
        vtk_effective_ys.SetName("effective_yield_stress")
        polydata.GetPointData().AddArray(vtk_effective_ys)

    # Add damage (scalar)
    if particle_damage is not None:
        vtk_damage = numpy_support.numpy_to_vtk(particle_damage.astype(np.float32))
        vtk_damage.SetName("damage")
        polydata.GetPointData().AddArray(vtk_damage)

    # Add mean stress (scalar)
    if particle_mean_stress is not None:
        vtk_mean_stress = numpy_support.numpy_to_vtk(particle_mean_stress.astype(np.float32))
        vtk_mean_stress.SetName("mean_stress")
        polydata.GetPointData().AddArray(vtk_mean_stress)

    # Add von Mises stress (scalar)
    if particle_von_mises is not None:
        vtk_von_mises = numpy_support.numpy_to_vtk(particle_von_mises.astype(np.float32))
        vtk_von_mises.SetName("von_mises_stress")
        polydata.GetPointData().AddArray(vtk_von_mises)

    # Add full stress tensor (6 components: xx, yy, zz, xy, xz, yz)
    if particle_stress_tensor is not None:
        # Ensure it's (N, 3, 3) shape
        if particle_stress_tensor.ndim == 3:
            N = particle_stress_tensor.shape[0]
            # Extract symmetric components (xx, yy, zz, xy, xz, yz)
            stress_components = np.zeros((N, 6), dtype=np.float32)
            stress_components[:, 0] = particle_stress_tensor[:, 0, 0]  # xx
            stress_components[:, 1] = particle_stress_tensor[:, 1, 1]  # yy
            stress_components[:, 2] = particle_stress_tensor[:, 2, 2]  # zz
            stress_components[:, 3] = particle_stress_tensor[:, 0, 1]  # xy
            stress_components[:, 4] = particle_stress_tensor[:, 0, 2]  # xz
            stress_components[:, 5] = particle_stress_tensor[:, 1, 2]  # yz
            
            vtk_stress = numpy_support.numpy_to_vtk(stress_components)
            vtk_stress.SetName("stress_tensor")
            vtk_stress.SetNumberOfComponents(6)
            polydata.GetPointData().AddArray(vtk_stress)

    # Add active label (integer)
    if particle_active_label is not None:
        vtk_active_label = numpy_support.numpy_to_vtk(particle_active_label.astype(np.int32))
        vtk_active_label.SetName("active_label")
        polydata.GetPointData().AddArray(vtk_active_label)

    # Add material label (integer): 0=MPM(locked), 1=MPM(transitionable), 2=XPBD
    if particle_material_label is not None:
        vtk_material_label = numpy_support.numpy_to_vtk(particle_material_label.astype(np.int32))
        vtk_material_label.SetName("material_label")
        polydata.GetPointData().AddArray(vtk_material_label)

    # === Add restart data if provided (for bigstep-end checkpoints) ===
    if restart_data is not None:
        # Simulation metadata (stored as field data)
        field_data = polydata.GetFieldData()
        
        # Scalar metadata
        for key in ['sim_time', 'bigStep', 'dx']:
            if key in restart_data:
                arr = vtk.vtkFloatArray()
                arr.SetName(key)
                arr.SetNumberOfTuples(1)
                arr.SetValue(0, float(restart_data[key]))
                field_data.AddArray(arr)
        
        # Deformation gradient F (9 components per particle, stored as 3x3 flattened)
        if 'particle_F' in restart_data:
            F = restart_data['particle_F']  # (N, 3, 3)
            F_flat = F.reshape(-1, 9).astype(np.float32)
            vtk_F = numpy_support.numpy_to_vtk(F_flat)
            vtk_F.SetName("deformation_gradient_F")
            vtk_F.SetNumberOfComponents(9)
            polydata.GetPointData().AddArray(vtk_F)
        
        # APIC affine matrix C (9 components per particle)
        if 'particle_C' in restart_data:
            C = restart_data['particle_C']  # (N, 3, 3)
            C_flat = C.reshape(-1, 9).astype(np.float32)
            vtk_C = numpy_support.numpy_to_vtk(C_flat)
            vtk_C.SetName("apic_C")
            vtk_C.SetNumberOfComponents(9)
            polydata.GetPointData().AddArray(vtk_C)
        
        # Accumulated plastic strain
        if 'particle_accumulated_strain' in restart_data:
            vtk_acc_strain = numpy_support.numpy_to_vtk(restart_data['particle_accumulated_strain'].astype(np.float32))
            vtk_acc_strain.SetName("accumulated_strain")
            polydata.GetPointData().AddArray(vtk_acc_strain)
        
        # Base yield stress (before creep)
        if 'ys_base' in restart_data:
            vtk_ys_base = numpy_support.numpy_to_vtk(restart_data['ys_base'].astype(np.float32))
            vtk_ys_base.SetName("yield_stress_base")
            polydata.GetPointData().AddArray(vtk_ys_base)
        
        # Material properties that may have evolved
        for prop_name in ['mu', 'lam', 'bulk', 'alpha', 'hardening', 'softening', 
                          'eta_shear', 'eta_bulk', 'strainCriteria']:
            if prop_name in restart_data:
                vtk_arr = numpy_support.numpy_to_vtk(restart_data[prop_name].astype(np.float32))
                vtk_arr.SetName(f"mpm_{prop_name}")
                polydata.GetPointData().AddArray(vtk_arr)
        
        # Particle volume and mass (may change with swelling)
        if 'particle_vol' in restart_data:
            vtk_vol = numpy_support.numpy_to_vtk(restart_data['particle_vol'].astype(np.float32))
            vtk_vol.SetName("particle_volume")
            polydata.GetPointData().AddArray(vtk_vol)
        
        if 'particle_mass' in restart_data:
            vtk_mass = numpy_support.numpy_to_vtk(restart_data['particle_mass'].astype(np.float32))
            vtk_mass.SetName("particle_mass")
            polydata.GetPointData().AddArray(vtk_mass)
        
        if 'particle_density' in restart_data:
            vtk_density = numpy_support.numpy_to_vtk(restart_data['particle_density'].astype(np.float32))
            vtk_density.SetName("particle_density")
            polydata.GetPointData().AddArray(vtk_density)
        
        # XPBD-specific state
        if 'particle_x_initial_xpbd' in restart_data:
            xpbd_init = restart_data['particle_x_initial_xpbd']  # (N, 3)
            vtk_xpbd_init = numpy_support.numpy_to_vtk(xpbd_init.astype(np.float32))
            vtk_xpbd_init.SetName("xpbd_initial_position")
            vtk_xpbd_init.SetNumberOfComponents(3)
            polydata.GetPointData().AddArray(vtk_xpbd_init)
        
        if 'particle_cumDist_xpbd' in restart_data:
            vtk_cumDist = numpy_support.numpy_to_vtk(restart_data['particle_cumDist_xpbd'].astype(np.float32))
            vtk_cumDist.SetName("xpbd_cumulative_distance")
            polydata.GetPointData().AddArray(vtk_cumDist)
        
        print(f"  (includes restart data for checkpoint)")

    writer_vtp = vtk.vtkXMLPolyDataWriter()
    writer_vtp.SetFileName(f"{output_prefix}_particles.vtp")
    writer_vtp.SetInputData(polydata)
    writer_vtp.Write()

    print(f"Exported: {output_prefix}_grid.vti and {output_prefix}_particles.vtp")


def render_mpm(
    renderer,
    sim,
    mpm,
    bigStep,
    counter,
    maxStress=None
):
    """
    Render particles and optionally save grid/particle state.

    Parameters
    ----------
    renderer : OpenGLRenderer
        Active renderer instance.
    sim : SimState
        Global simulation state
    mpm : MPMState
        MPM-specific state
    bigStep : int
    counter : int
    maxStress : float, optional
        Maximum stress for color scaling
    """

    # --- Choose coloring ---
    if sim.color_mode == "yield_ratio":
        colors = values_to_rgb(
            (mpm.ys.numpy() / mpm.ys_base.numpy()),
            min_val=0.0,
            max_val=1.0
        )

    elif sim.color_mode == "damage":
        colors = values_to_rgb(
            mpm.particle_damage.numpy(),
            min_val=0.0,
            max_val=1.0
        )

    elif sim.color_mode == "vz":
        vz = sim.particle_v.numpy()[:, 2]
        colors = values_to_rgb(
            vz,
            min_val=vz.min(),
            max_val=vz.max()
        )

    elif sim.color_mode == "von_mises":
        sigma = mpm.particle_stress.numpy().astype(np.float64)  # (N,3,3)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        identity = np.eye(3)
        s = sigma - mean_stress[:, None, None] * identity
        von_mises = np.sqrt(1.5 * np.sum(s**2, axis=(1, 2)))

        if maxStress is None:
            maxStress = np.quantile(von_mises, 0.95)

        colors = values_to_rgb(
            von_mises,
            min_val=0.0,
            max_val=maxStress
        )

    elif sim.color_mode == "effective_ys":
        sigma = mpm.particle_stress.numpy().astype(np.float64)  # (N,3,3)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        effective_ys = (mpm.ys.numpy() - mpm.alpha.numpy() * mean_stress)*(1-mpm.particle_damage.numpy())
        colors = values_to_rgb(
            effective_ys,
            min_val=np.quantile(effective_ys, 0.1),
            max_val=np.quantile(effective_ys, 0.9)+1
        )

    elif sim.color_mode == "ys":
        sigma = mpm.particle_stress.numpy().astype(np.float64)  # (N,3,3)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        effective_ys = mpm.ys.numpy()
        colors = values_to_rgb(
            effective_ys,
            min_val=np.quantile(effective_ys, 0.1),
            max_val=np.quantile(effective_ys, 0.9)+1
        )

    elif sim.color_mode == "stress":
        # 0 stress should always be the middle color
        sigma = mpm.particle_stress.numpy().astype(np.float64)  # (N,3,3)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        # Scale symmetrically around 0 based on larger magnitude
        stress_min = np.quantile(mean_stress, 0.01)
        stress_max = np.quantile(mean_stress, 0.99)
        max_magnitude = max(abs(stress_min), abs(stress_max))
        colors = values_to_rgb(
            mean_stress,
            min_val=-max_magnitude,  # Tension
            max_val=max_magnitude    # Compression
        )
    
    elif sim.color_mode == "sigma_zz":
        # Vertical stress component - shows geostatic gradient
        # Use absolute value: larger compression magnitude → hotter color
        sigma = mpm.particle_stress.numpy().astype(np.float64)  # (N,3,3)
        sigma_zz = sigma[:, 2, 2]  # Vertical component (negative = compression)
        sigma_zz_abs = np.abs(sigma_zz)
        colors = values_to_rgb(
            sigma_zz_abs,
            min_val=0.0,  # No stress (surface)
            max_val=np.quantile(sigma_zz_abs, 0.99)  # Maximum compression (bottom)
        )
    
    elif sim.color_mode == "state":
        sigma = sim.materialLabel.numpy().astype(np.float64)  # (N,3,3)
        colors = values_to_rgb(
            sigma,
            min_val=0.0,
            max_val=np.max(sigma)
        )
    else:
        raise ValueError(f"Unknown color_mode: {sim.color_mode}")

    # --- Render ---
    renderer.begin_frame()
    renderer.render_points(
        points=sim.particle_x,
        name="points",
        radius=sim.particle_radius.numpy(),
        colors=colors,
        dynamic=True
    )
    renderer.end_frame()
    renderer.update_view_matrix()

    return maxStress  # so you can keep track of running maximum for von Mises



def save_mpm(
    sim,
    mpm,
    bigStep,
    counter,
    include_restart_data=False,
    xpbd=None
):
    """
    Save particle and grid state to VTK files.
    
    Parameters
    ----------
    sim : SimState
        Global simulation state
    mpm : MPMState
        MPM-specific state
    bigStep : int
    counter : int
    include_restart_data : bool
        If True, include all data needed for simulation restart (checkpoint)
    xpbd : XPBDState, optional
        XPBD state (required if include_restart_data=True)
    """
    # Compute mean stress and von Mises stress
    sigma = mpm.particle_stress.numpy().astype(np.float64)  # (N, 3, 3)
    mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0  # (N,)
    
    # Deviatoric stress
    identity = np.eye(3)
    s = sigma - mean_stress[:, None, None] * identity  # (N, 3, 3)
    von_mises = np.sqrt(1.5 * np.sum(s**2, axis=(1, 2)))  # (N,)

    # Compute effective yield stress for Drucker-Prager: (ys - alpha*p)*(1-D)
    # where p = -mean_stress (pressure, positive in compression)
    ys = mpm.ys.numpy()
    alpha = mpm.alpha.numpy()
    damage = mpm.particle_damage.numpy()
    effective_ys = (ys - alpha * mean_stress) * (1.0 - damage)

    # Prepare restart data if requested
    restart_data = None
    if include_restart_data:
        restart_data = {
            # Simulation metadata
            'sim_time': sim.t,
            'bigStep': bigStep,
            'dx': sim.dx,
            
            # MPM state
            'particle_F': mpm.particle_F.numpy(),
            'particle_C': mpm.particle_C.numpy(),
            'particle_accumulated_strain': mpm.particle_accumulated_strain.numpy(),
            'ys_base': mpm.ys_base.numpy(),
            
            # Material properties
            'mu': mpm.mu.numpy(),
            'lam': mpm.lam.numpy(),
            'bulk': mpm.bulk.numpy(),
            'alpha': mpm.alpha.numpy(),
            'hardening': mpm.hardening.numpy(),
            'softening': mpm.softening.numpy(),
            'eta_shear': mpm.eta_shear.numpy(),
            'eta_bulk': mpm.eta_bulk.numpy(),
            'strainCriteria': mpm.strainCriteria.numpy(),
            
            # Physical properties
            'particle_vol': sim.particle_vol.numpy(),
            'particle_mass': sim.particle_mass.numpy(),
            'particle_density': sim.particle_density.numpy(),
        }
        
        # XPBD state (if provided)
        if xpbd is not None:
            restart_data['particle_x_initial_xpbd'] = xpbd.particle_x_initial_xpbd.numpy()
            restart_data['particle_cumDist_xpbd'] = xpbd.particle_cumDist_xpbd.numpy()

    # --- Save if requested ---
    # Filename format: sim_step_BBBB_CCCCCC_tTTTTTTTT where B=bigStep, C=counter, T=time in seconds (8 digits, 4 decimals)
    save_grid_and_particles_vti_vtp(
        output_prefix=f"{sim.outputFolder}/sim_step_{bigStep:04d}{counter:06d}{int(np.ceil(sim.t*1e8)):016d}",
        grid_mass=mpm.grid_m.numpy(),
        minBounds=sim.minBounds,
        dx=sim.dx,
        particle_positions=sim.particle_x.numpy(),
        particle_radius=sim.particle_radius.numpy(),
        particle_velocity=sim.particle_v.numpy(),
        particle_ys=ys,
        particle_effective_ys=effective_ys,
        particle_damage=damage,
        particle_mean_stress=mean_stress,
        particle_von_mises=von_mises,
        particle_stress_tensor=sigma,  # Add full stress tensor
        particle_active_label=sim.activeLabel.numpy(),
        particle_material_label=sim.materialLabel.numpy(),
        restart_data=restart_data
    )


def open_paraview_with_vtp_series(output_folder):
    """
    Open ParaView and load the VTP particle time series from the output folder.
    
    Args:
        output_folder: Path to the folder containing *_particles.vtp files
    """
    import subprocess
    import glob
    import os
    
    # Find all particle VTP files
    vtp_pattern = os.path.join(output_folder, "*_particles.vtp")
    vtp_files = sorted(glob.glob(vtp_pattern))
    
    if not vtp_files:
        print(f"Warning: No VTP particle files found in {output_folder}")
        return False
    
    if len(vtp_files) < 2:
        print(f"Warning: Only {len(vtp_files)} VTP file found. Need at least 2 for a time series.")
        print(f"VTP files are located at: {output_folder}")
        return False
    
    print(f"Found {len(vtp_files)} VTP particle files in {output_folder}")
    
    # ParaView can load a time series by opening any file in the series
    # It auto-detects the pattern and loads all files
    first_file = vtp_files[0]
    
    # Try to find ParaView executable
    paraview_candidates = [
        "paraview",  # If in PATH
        r"C:\Program Files\ParaView 5.12.0\bin\paraview.exe",
        r"C:\Program Files\ParaView 5.11.2\bin\paraview.exe",
        r"C:\Program Files\ParaView 5.11.1\bin\paraview.exe",
        r"C:\Program Files\ParaView 5.11.0\bin\paraview.exe",
        r"C:\Program Files\ParaView 5.10.1\bin\paraview.exe",
        r"C:\Program Files (x86)\ParaView 5.12.0\bin\paraview.exe",
        r"C:\Program Files (x86)\ParaView 5.11.2\bin\paraview.exe",
    ]
    
    paraview_exe = None
    for candidate in paraview_candidates:
        # Check if it's in PATH or exists as a file
        if os.path.isfile(candidate):
            paraview_exe = candidate
            break
        # Try running it to see if it's in PATH
        try:
            result = subprocess.run([candidate, "--version"], 
                                    capture_output=True, timeout=5)
            if result.returncode == 0:
                paraview_exe = candidate
                break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    
    if paraview_exe is None:
        print("Warning: Could not find ParaView executable.")
        print("Please install ParaView or add it to your PATH.")
        print(f"VTP files are located at: {output_folder}")
        print(f"First file: {first_file}")
        return False
    
    print(f"Opening ParaView: {paraview_exe}")
    print(f"Loading file: {first_file}")
    
    # Launch ParaView with the first file (it will auto-detect the series)
    # Use Popen to not block the Python process
    try:
        subprocess.Popen([paraview_exe, first_file])
        print("ParaView launched successfully.")
        print("Tip: In ParaView, use the 'Play' button or animation controls to view the time series.")
        return True
    except Exception as e:
        print(f"Error launching ParaView: {e}")
        print(f"VTP files are located at: {output_folder}")
        return False

