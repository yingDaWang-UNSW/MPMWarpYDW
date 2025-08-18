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

def save_grid_and_particles_vti_vtp(
    output_prefix,
    grid_mass,           # shape (nx, ny, nz)
    minBounds,           # (x0, y0, z0)
    dx,                  # scalar spacing
    particle_positions,  # shape (N, 3)
    particle_radius=0.5  # optional radius field or scalar
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

    # Optional: add radius or other scalar fields
    radius_array = np.full((positions.shape[0],), particle_radius, dtype=np.float32)
    vtk_radius = numpy_support.numpy_to_vtk(radius_array)
    vtk_radius.SetName("radius")
    polydata.GetPointData().AddArray(vtk_radius)

    writer_vtp = vtk.vtkXMLPolyDataWriter()
    writer_vtp.SetFileName(f"{output_prefix}_particles.vtp")
    writer_vtp.SetInputData(polydata)
    writer_vtp.Write()

    print(f"Exported: {output_prefix}_grid.vti and {output_prefix}_particles.vtp")


def render_and_save(
    renderer,
    particle_x,
    particle_v,
    particle_radius,
    ys,
    ys_base,
    particle_damage,
    particle_stress,
    grid_m,
    minBounds,
    dx,
    bigStep,
    counter,
    nPoints,
    saveFlag=False,
    color_mode="yield_ratio",
    maxStress=None
):
    """
    Render particles and optionally save grid/particle state.

    Parameters
    ----------
    renderer : OpenGLRenderer
        Active renderer instance.
    particle_x : wp.array (N,3)
    particle_radius : wp.array (N,)
    ys : wp.array (N,)
    ys_base : wp.array (N,)
    particle_damage : wp.array (N,)
    particle_stress : wp.array (N,3,3)
    grid_m : wp.array (nx,ny,nz)
    minBounds : wp.vec3 or np.array(3,)
    dx : float
    bigStep : int
    counter : int
    nPoints : int
    saveFlag : bool
    color_mode : str
        Options: "yield_ratio", "damage", "vz", "von_mises"
    maxStress : float or None
        If color_mode="von_mises", this is the running max for normalization.
    """

    # --- Choose coloring ---
    if color_mode == "yield_ratio":
        colors = values_to_rgb(
            (ys.numpy() / ys_base.numpy()),
            min_val=0.0,
            max_val=1.0
        )

    elif color_mode == "damage":
        colors = values_to_rgb(
            particle_damage.numpy(),
            min_val=0.0,
            max_val=1.0
        )

    elif color_mode == "vz":
        vz = particle_v.numpy()[:, 2]
        colors = values_to_rgb(
            vz,
            min_val=vz.min(),
            max_val=vz.max()
        )

    elif color_mode == "von_mises":
        sigma = particle_stress.numpy().astype(np.float64)  # (N,3,3)
        mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
        identity = np.eye(3)
        s = sigma - mean_stress[:, None, None] * identity
        von_mises = np.sqrt(1.5 * np.sum(s**2, axis=(1, 2)))

        if maxStress is None:
            maxStress = np.max(von_mises)

        colors = values_to_rgb(
            von_mises,
            min_val=0.0,
            max_val=maxStress
        )
    else:
        raise ValueError(f"Unknown color_mode: {color_mode}")

    # --- Render ---
    renderer.begin_frame()
    renderer.render_points(
        points=particle_x,
        name="points",
        radius=particle_radius.numpy(),
        colors=colors,
        dynamic=True
    )
    renderer.end_frame()
    renderer.update_view_matrix()

    # --- Save if requested ---
    if saveFlag:
        save_grid_and_particles_vti_vtp(
            output_prefix=f"./output/sim_step_{bigStep}_{counter:06d}",
            grid_mass=grid_m.numpy(),
            minBounds=minBounds,
            dx=dx,
            particle_positions=particle_x.numpy(),
            particle_radius=np.arange(0, nPoints, 1)  # Point Gaussian trick
        )

    return maxStress  # so you can keep track of running maximum for von Mises
