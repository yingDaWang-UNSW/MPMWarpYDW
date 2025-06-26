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
